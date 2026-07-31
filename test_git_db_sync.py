import json
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from git_db_sync import GitSnapshotPublisher, maybe_sync_database


def run_git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def create_database(path: Path, row_count: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE simulation_records (
                sample_id TEXT PRIMARY KEY,
                value REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO simulation_records VALUES (?, ?)",
            [(f"sample-{index}", float(index)) for index in range(row_count)],
        )
        connection.commit()
    finally:
        connection.close()


def add_rows(path: Path, start: int, count: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executemany(
            "INSERT INTO simulation_records VALUES (?, ?)",
            [
                (f"sample-{index}", float(index))
                for index in range(start, start + count)
            ],
        )
        connection.commit()
    finally:
        connection.close()


def count_rows(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM simulation_records"
            ).fetchone()[0]
        )
    finally:
        connection.close()


class FakePublisher:
    def __init__(self, fail=False):
        self.fail = fail
        self.snapshots = []

    def publish(self, snapshot_path, row_count):
        if self.fail:
            raise RuntimeError("simulated git push failure")
        captured = Path(snapshot_path).with_name(
            f"captured-{len(self.snapshots)}.db"
        )
        captured.write_bytes(Path(snapshot_path).read_bytes())
        self.snapshots.append((captured, row_count))


class GitDatabaseSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "quantum_simulation.db"
        self.snapshot_path = (
            self.root / "database_backup" / "quantum_simulation.db"
        )
        self.state_path = self.root / "git_sync_state.json"
        self.config_path = self.root / "git_sync_config.json"
        self.db_config_path = self.root / "db_config.json"

        self.db_config_path.write_text(
            json.dumps(
                {
                    "database": {
                        "db_name": str(self.database_path),
                        "table_name": "simulation_records",
                    }
                }
            ),
            encoding="utf-8",
        )
        self.config_path.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "batch_size": 100,
                    "db_config": str(self.db_config_path),
                    "snapshot_path": str(self.snapshot_path),
                    "state_file": str(self.state_path),
                    "git_repository": str(self.root),
                    "git_remote": "origin",
                    "git_branch": "main",
                    "commit_message": "Update: 更新模擬資料庫快照",
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_first_hundred_rows_publish_a_complete_snapshot(self):
        create_database(self.database_path, 100)
        publisher = FakePublisher()

        result = maybe_sync_database(
            self.config_path,
            publisher_factory=lambda _configuration: publisher,
        )

        self.assertTrue(result.uploaded)
        self.assertEqual(result.row_count, 100)
        self.assertEqual(count_rows(publisher.snapshots[0][0]), 100)

    def test_second_snapshot_contains_previous_and_new_rows(self):
        create_database(self.database_path, 100)
        publisher = FakePublisher()
        self.assertTrue(
            maybe_sync_database(
                self.config_path,
                publisher_factory=lambda _configuration: publisher,
            ).uploaded
        )

        add_rows(self.database_path, 100, 99)
        skipped = maybe_sync_database(
            self.config_path,
            publisher_factory=lambda _configuration: publisher,
        )
        self.assertFalse(skipped.uploaded)
        self.assertEqual(skipped.pending_rows, 99)

        add_rows(self.database_path, 199, 1)
        second = maybe_sync_database(
            self.config_path,
            publisher_factory=lambda _configuration: publisher,
        )
        self.assertTrue(second.uploaded)
        self.assertEqual(count_rows(publisher.snapshots[1][0]), 200)

    def test_push_failure_does_not_advance_state(self):
        create_database(self.database_path, 100)
        failed = maybe_sync_database(
            self.config_path,
            publisher_factory=lambda _configuration: FakePublisher(fail=True),
        )

        self.assertFalse(failed.uploaded)
        self.assertIsNotNone(failed.error)
        self.assertFalse(self.state_path.exists())

    def test_force_publishes_partial_tail_batch(self):
        create_database(self.database_path, 37)
        publisher = FakePublisher()

        result = maybe_sync_database(
            self.config_path,
            force=True,
            publisher_factory=lambda _configuration: publisher,
        )

        self.assertTrue(result.uploaded)
        self.assertEqual(count_rows(publisher.snapshots[0][0]), 37)

    def test_real_git_publisher_commits_only_database_snapshot(self):
        repository = self.root / "repository"
        remote = self.root / "remote.git"
        repository.mkdir()

        run_git("init", "--bare", str(remote), cwd=self.root)
        run_git("init", "-b", "main", cwd=repository)
        run_git("config", "user.name", "Test User", cwd=repository)
        run_git("config", "user.email", "test@example.com", cwd=repository)
        run_git("remote", "add", "origin", str(remote), cwd=repository)

        readme = repository / "README.md"
        readme.write_text("test repository\n", encoding="utf-8")
        run_git("add", "README.md", cwd=repository)
        run_git("commit", "-m", "Initial commit", cwd=repository)
        run_git("push", "-u", "origin", "main", cwd=repository)

        source_db = repository / "quantum_simulation.db"
        create_database(source_db, 100)
        snapshot = repository / "database_backup" / "quantum_simulation.db"
        unrelated = repository / "do_not_upload.txt"
        unrelated.write_text("must remain untracked\n", encoding="utf-8")
        already_staged = repository / "already_staged.txt"
        already_staged.write_text("must remain staged\n", encoding="utf-8")
        run_git("add", "already_staged.txt", cwd=repository)

        publisher = GitSnapshotPublisher(
            repository=repository,
            snapshot_path=snapshot,
            remote="origin",
            branch="main",
            commit_message="Update: 更新模擬資料庫快照",
        )
        publisher.publish(source_db, row_count=100)

        tracked_files = set(
            run_git("ls-tree", "-r", "--name-only", "HEAD", cwd=repository)
            .stdout.splitlines()
        )
        self.assertIn("database_backup/quantum_simulation.db", tracked_files)
        self.assertNotIn("do_not_upload.txt", tracked_files)
        self.assertNotIn("already_staged.txt", tracked_files)
        staged_after_commit = set(
            run_git(
                "diff",
                "--cached",
                "--name-only",
                cwd=repository,
            ).stdout.splitlines()
        )
        self.assertIn("already_staged.txt", staged_after_commit)
        self.assertEqual(count_rows(snapshot), 100)

        remote_commit = run_git(
            "--git-dir",
            str(remote),
            "rev-parse",
            "refs/heads/main",
            cwd=self.root,
        ).stdout.strip()
        local_commit = run_git("rev-parse", "HEAD", cwd=repository).stdout.strip()
        self.assertEqual(remote_commit, local_commit)


if __name__ == "__main__":
    unittest.main()