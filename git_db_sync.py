"""Publish cumulative SQLite snapshots with Git.

Only the configured snapshot path is staged and committed. Other modified,
untracked, or already-staged project files are intentionally excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence


_SAFE_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SyncError(RuntimeError):
    """A recoverable Git/SQLite synchronization failure."""


@dataclass(frozen=True)
class SyncConfiguration:
    enabled: bool
    batch_size: int
    database_path: Path
    table_name: str
    snapshot_path: Path
    state_path: Path
    work_directory: Path
    git_repository: Path
    git_remote: str
    git_branch: str
    commit_message: str


@dataclass(frozen=True)
class DatabaseInfo:
    row_count: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SyncResult:
    uploaded: bool
    row_count: int
    pending_rows: int
    reason: str
    error: Optional[str] = None


PublisherFactory = Callable[[SyncConfiguration], "GitSnapshotPublisher"]


def _resolve_path(base_directory: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise SyncError(f"{path.name} 最外層必須是 JSON 物件。")
    return loaded


def load_configuration(
    config_path: os.PathLike[str] | str = "git_sync_config.json",
) -> SyncConfiguration:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise SyncError(f"找不到 Git 同步設定檔：{config_file}")

    raw = _load_json(config_file)
    base_directory = config_file.parent

    batch_size = int(raw.get("batch_size", 100))
    if batch_size <= 0:
        raise SyncError("git_sync_config.json 的 batch_size 必須大於 0。")

    db_config_path = _resolve_path(
        base_directory,
        str(raw.get("db_config", "db_config.json")),
    )
    if not db_config_path.exists():
        raise SyncError(f"找不到資料庫設定檔：{db_config_path}")

    db_config = _load_json(db_config_path)
    database_section = db_config.get("database", {})
    if not isinstance(database_section, dict):
        raise SyncError("db_config.json 的 database 必須是 JSON 物件。")

    database_name = str(
        database_section.get("db_name", "quantum_simulation.db")
    )
    table_name = str(
        database_section.get("table_name", "simulation_records")
    )
    if not _SAFE_SQL_IDENTIFIER.fullmatch(table_name):
        raise SyncError(f"不安全的 SQLite 資料表名稱：{table_name}")

    git_repository = _resolve_path(
        base_directory,
        str(raw.get("git_repository", ".")),
    )
    snapshot_path = _resolve_path(
        git_repository,
        str(
            raw.get(
                "snapshot_path",
                "database_backup/quantum_simulation.db",
            )
        ),
    )

    git_remote = str(raw.get("git_remote", "origin")).strip()
    git_branch = str(raw.get("git_branch", "main")).strip()
    commit_message = str(
        raw.get("commit_message", "Update: 更新模擬資料庫快照")
    ).strip()
    if not git_remote or not git_branch or not commit_message:
        raise SyncError("Git remote、branch 與 commit_message 都不可為空。")

    return SyncConfiguration(
        enabled=bool(raw.get("enabled", True)),
        batch_size=batch_size,
        database_path=_resolve_path(db_config_path.parent, database_name),
        table_name=table_name,
        snapshot_path=snapshot_path,
        state_path=_resolve_path(
            base_directory,
            str(raw.get("state_file", "git_sync_state.json")),
        ),
        work_directory=_resolve_path(
            base_directory,
            str(raw.get("work_directory", "git_sync_work")),
        ),
        git_repository=git_repository,
        git_remote=git_remote,
        git_branch=git_branch,
        commit_message=commit_message,
    )


def _read_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SyncError(f"同步狀態檔損壞：{path}（{exc}）") from exc


def _write_state_atomic(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(state, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _database_row_count(database_path: Path, table_name: str) -> int:
    if not database_path.exists():
        raise SyncError(f"找不到模擬資料庫：{database_path}")
    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro",
        uri=True,
        timeout=30,
    )
    try:
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]
        )
    except sqlite3.Error as exc:
        raise SyncError(f"讀取 SQLite 資料筆數失敗：{exc}") from exc
    finally:
        connection.close()


def inspect_database(database_path: Path, table_name: str) -> DatabaseInfo:
    connection = sqlite3.connect(
        f"file:{database_path}?mode=ro",
        uri=True,
        timeout=30,
    )
    try:
        integrity_messages = [
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check").fetchall()
        ]
        if integrity_messages != ["ok"]:
            raise SyncError(
                "SQLite 完整性檢查失敗：" + "; ".join(integrity_messages)
            )
        row_count = int(
            connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]
        )
    except sqlite3.Error as exc:
        raise SyncError(f"驗證 SQLite 快照失敗：{exc}") from exc
    finally:
        connection.close()

    return DatabaseInfo(
        row_count=row_count,
        size_bytes=database_path.stat().st_size,
        sha256=_sha256(database_path),
    )


def create_verified_snapshot(
    source_database: Path,
    destination: Path,
    table_name: str,
) -> DatabaseInfo:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)

    source = sqlite3.connect(
        f"file:{source_database}?mode=ro",
        uri=True,
        timeout=60,
    )
    target = sqlite3.connect(destination, timeout=60)
    try:
        source.backup(target)
        target.commit()
    except sqlite3.Error as exc:
        destination.unlink(missing_ok=True)
        raise SyncError(f"建立 SQLite 安全快照失敗：{exc}") from exc
    finally:
        target.close()
        source.close()

    return inspect_database(destination, table_name)


class GitSnapshotPublisher:
    """Atomically place and commit only one configured SQLite snapshot."""

    def __init__(
        self,
        repository: Path,
        snapshot_path: Path,
        remote: str,
        branch: str,
        commit_message: str,
    ):
        self.repository = Path(repository).resolve()
        self.snapshot_path = Path(snapshot_path).resolve()
        self.remote = remote
        self.branch = branch
        self.commit_message = commit_message

    @classmethod
    def from_configuration(
        cls,
        configuration: SyncConfiguration,
    ) -> "GitSnapshotPublisher":
        return cls(
            repository=configuration.git_repository,
            snapshot_path=configuration.snapshot_path,
            remote=configuration.git_remote,
            branch=configuration.git_branch,
            commit_message=configuration.commit_message,
        )

    def _git(
        self,
        *arguments: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=self.repository,
                check=check,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise SyncError(
                "找不到 Git。請先安裝 Git for Windows 並重新啟動 VS Code。"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise SyncError(
                f"Git 指令失敗：git {' '.join(arguments)}\n{detail}"
            ) from exc

    def _snapshot_relative_path(self) -> Path:
        try:
            return self.snapshot_path.relative_to(self.repository)
        except ValueError as exc:
            raise SyncError(
                "snapshot_path 必須位於 git_repository 資料夾內。"
            ) from exc

    def _verify_repository(self) -> None:
        result = self._git("rev-parse", "--show-toplevel")
        actual_root = Path(result.stdout.strip()).resolve()
        if actual_root != self.repository:
            raise SyncError(
                f"git_repository 不是目前 Git 根目錄：{self.repository}"
            )
        self._git("remote", "get-url", self.remote)

    def _replace_snapshot_atomically(self, source_snapshot: Path) -> None:
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.snapshot_path.with_name(
            f"{self.snapshot_path.name}.tmp-{uuid.uuid4().hex}"
        )
        try:
            shutil.copyfile(source_snapshot, temporary_path)
            os.replace(temporary_path, self.snapshot_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def publish(
        self,
        source_snapshot: os.PathLike[str] | str,
        row_count: int,
    ) -> None:
        self._verify_repository()
        source_path = Path(source_snapshot).resolve()
        self._replace_snapshot_atomically(source_path)

        relative_path = self._snapshot_relative_path().as_posix()

        # Status is diagnostic only. The pathspec on add/commit is the safety
        # boundary that prevents unrelated files from entering this commit.
        self._git("status", "--short", "--", relative_path)
        self._git("add", "--", relative_path)

        staged_change = self._git(
            "diff",
            "--cached",
            "--quiet",
            "--",
            relative_path,
            check=False,
        )
        if staged_change.returncode not in (0, 1):
            raise SyncError("無法判斷 SQLite 快照是否有 Git 變更。")

        if staged_change.returncode == 1:
            message = f"{self.commit_message} ({row_count} rows)"
            self._git(
                "commit",
                "-m",
                message,
                "--",
                relative_path,
            )

        # Push also runs when no new commit was needed, allowing recovery from
        # an earlier commit-success/push-failure situation.
        self._git("push", self.remote, self.branch)


def _default_publisher_factory(
    configuration: SyncConfiguration,
) -> GitSnapshotPublisher:
    return GitSnapshotPublisher.from_configuration(configuration)


def _pending_rows(current_count: int, uploaded_count: int) -> int:
    if current_count >= uploaded_count:
        return current_count - uploaded_count
    return current_count


def maybe_sync_database(
    config_path: os.PathLike[str] | str = "git_sync_config.json",
    *,
    force: bool = False,
    publisher_factory: PublisherFactory = _default_publisher_factory,
) -> SyncResult:
    """Commit and push a complete snapshot after the configured row interval."""
    try:
        configuration = load_configuration(config_path)
        if not configuration.enabled:
            return SyncResult(False, 0, 0, "Git 自動同步目前已停用。")

        row_count = _database_row_count(
            configuration.database_path,
            configuration.table_name,
        )
        state = _read_state(configuration.state_path)
        uploaded_count = int(state.get("last_uploaded_row_count", 0))
        pending_rows = _pending_rows(row_count, uploaded_count)

        if row_count == 0:
            return SyncResult(False, 0, pending_rows, "資料庫目前沒有資料。")
        if not force and pending_rows < configuration.batch_size:
            return SyncResult(
                False,
                row_count,
                pending_rows,
                (
                    f"尚未累積滿 {configuration.batch_size} 筆，"
                    "本輪不需提交。"
                ),
            )

        configuration.work_directory.mkdir(parents=True, exist_ok=True)
        temporary_snapshot = (
            configuration.work_directory
            / f"quantum_simulation.db.snapshot-{uuid.uuid4().hex}"
        )
        try:
            snapshot_info = create_verified_snapshot(
                configuration.database_path,
                temporary_snapshot,
                configuration.table_name,
            )
            if snapshot_info.row_count != row_count:
                raise SyncError(
                    "建立快照期間資料筆數改變，本輪不提交並於下一筆重試。"
                )

            publisher = publisher_factory(configuration)
            publisher.publish(temporary_snapshot, snapshot_info.row_count)

            _write_state_atomic(
                configuration.state_path,
                {
                    "last_uploaded_row_count": snapshot_info.row_count,
                    "snapshot_sha256": snapshot_info.sha256,
                    "snapshot_size_bytes": snapshot_info.size_bytes,
                    "snapshot_path": str(configuration.snapshot_path),
                    "git_remote": configuration.git_remote,
                    "git_branch": configuration.git_branch,
                    "synced_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ",
                        time.gmtime(),
                    ),
                },
            )
        finally:
            temporary_snapshot.unlink(missing_ok=True)

        return SyncResult(
            True,
            snapshot_info.row_count,
            0,
            (
                f"已提交並推送包含 {snapshot_info.row_count} 筆資料的"
                "完整 SQLite 快照。"
            ),
        )
    except Exception as exc:
        return SyncResult(
            False,
            locals().get("row_count", 0),
            locals().get("pending_rows", 0),
            "Git 同步失敗，模擬可繼續執行，下一筆會再重試。",
            error=str(exc),
        )


def synchronization_status(
    config_path: os.PathLike[str] | str = "git_sync_config.json",
) -> SyncResult:
    configuration = load_configuration(config_path)
    if not configuration.enabled:
        return SyncResult(False, 0, 0, "Git 自動同步目前已停用。")
    row_count = _database_row_count(
        configuration.database_path,
        configuration.table_name,
    )
    state = _read_state(configuration.state_path)
    uploaded_count = int(state.get("last_uploaded_row_count", 0))
    pending = _pending_rows(row_count, uploaded_count)
    return SyncResult(
        False,
        row_count,
        pending,
        (
            f"資料庫共 {row_count} 筆，上次成功 Push {uploaded_count} 筆，"
            f"待同步 {pending} 筆。"
        ),
    )


def _print_result(result: SyncResult) -> None:
    icon = "✅" if result.uploaded else ("⚠️" if result.error else "ℹ️")
    print(f"{icon} {result.reason}")
    if result.error:
        print(f"   原因：{result.error}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="以 Git 提交並 Push 最新的完整 SQLite 快照。"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["sync", "force", "status"],
        default="sync",
    )
    parser.add_argument(
        "--config",
        default="git_sync_config.json",
        help="Git 同步設定檔路徑。",
    )
    args = parser.parse_args()

    if args.command == "status":
        try:
            result = synchronization_status(args.config)
        except Exception as exc:
            print(f"❌ 讀取同步狀態失敗：{exc}")
            return 1
    else:
        result = maybe_sync_database(
            args.config,
            force=args.command == "force",
        )

    _print_result(result)
    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())