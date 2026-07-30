import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT_DIR, "runs")

# =========================================================================
# 智慧型流水線開關
# =========================================================================
PIPELINE_SWITCHES = {
    "drc_check": True,
    "gds_generator": True,
    "q3d_extraction": True,
    "lom_bridge": True,
    # 資料產生階段先關閉 Spec，避免規格不通過時中斷 Q3D/LOM 資料收集。
    # 完成 spec_results.json 與連續指標輸出後可再改回 True。
    "spec_check": False,
    "db_manager": True,
}

PIPELINE_CONFIGS = [
    ("drc_check", "drc_checker.py", "1. 執行 DRC 與參數限制檢查"),
    ("gds_generator", "gds_generator.py", "2. 繪製多圖層 GDS 版圖"),
    (
        "q3d_extraction",
        "q3d_auto_extraction.py",
        "3. Ansys Q3D 靜電場模擬與矩陣萃取",
    ),
    ("lom_bridge", "lom_bridge.py", "4. LOM 模型量子參數計算"),
    ("spec_check", "spec_checker.py", "4.5 物理指標過濾 (Detuning & Geff)"),
]

EXPECTED_OUTPUTS = {
    "gds_generator": "quantum_chip_final.gds",
    "q3d_extraction": "capacitance_matrix_results.json",
    "lom_bridge": "lom_results.json",
}

# 各階段最大執行時間。Q3D 超時時只讓該樣本失敗，不讓整個批次永久卡住。
STAGE_TIMEOUT_SECONDS = {
    "drc_checker.py": 20,
    "gds_generator.py": 40,
    "q3d_auto_extraction.py": 240,  # 4分鐘
    "lom_bridge.py": 20,
    "spec_checker.py": 20,
}
DEFAULT_STAGE_TIMEOUT_SECONDS = 600


def calculate_param_hash(json_path: str) -> str:
    """根據幾何、材料、模擬與 LOM 設定計算唯一 MD5。"""
    if not os.path.exists(json_path):
        return "UNKNOWN_HASH"

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    target_keys = [
        "toggles",
        "global",
        "qubit",
        "coupler",
        "t_coupler",
        "h_coupler",
        "substrate",
        "Qubit_pra",
        "simulation",
        "project_settings",
        "lom_settings",
    ]
    filtered_data = {key: data.get(key, {}) for key in target_keys}
    dump_string = json.dumps(
        filtered_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.md5(dump_string.encode("utf-8")).hexdigest()


def _write_stage_log(
    work_dir: str,
    script_name: str,
    command: List[str],
    stdout: str,
    stderr: str,
    return_code: int,
) -> str:
    """將每一階段的 stdout/stderr 完整保存，方便失敗追蹤。"""
    log_name = f"{os.path.splitext(os.path.basename(script_name))[0]}.log"
    log_path = os.path.join(work_dir, log_name)

    with open(log_path, "w", encoding="utf-8") as file:
        file.write(f"COMMAND: {' '.join(command)}\n")
        file.write(f"RETURN_CODE: {return_code}\n")
        file.write("\n========== STDOUT ==========\n")
        file.write(stdout or "")
        file.write("\n========== STDERR ==========\n")
        file.write(stderr or "")

    return log_path


def _normalize_subprocess_output(output: Any) -> str:
    """將 TimeoutExpired 可能回傳的 bytes/str 統一轉成字串。"""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return str(output)


def execute_sub_script(
    script_name: str,
    work_dir: str,
    arg_input: Optional[str] = None,
) -> Tuple[bool, str]:
    """在指定 sample 資料夾內執行子腳本，並保存完整 log。"""
    script_path = os.path.join(ROOT_DIR, script_name)
    if not os.path.exists(script_path):
        return False, f"找不到子腳本：{script_path}"

    command = [sys.executable, script_path]
    if arg_input:
        command.extend(["-i", arg_input])

    timeout_seconds = STAGE_TIMEOUT_SECONDS.get(
        script_name,
        DEFAULT_STAGE_TIMEOUT_SECONDS,
    )

    try:
        result = subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )

        log_path = _write_stage_log(
            work_dir=work_dir,
            script_name=script_name,
            command=command,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )
        combined_output = "\n".join(
            text.strip()
            for text in [result.stdout, result.stderr]
            if text and text.strip()
        )

        if result.returncode != 0:
            if not combined_output:
                combined_output = "子程式沒有輸出錯誤訊息。"
            return (
                False,
                f"Return Code {result.returncode}. "
                f"Output: {combined_output} | Log: {log_path}",
            )

        return True, f"Success | Log: {log_path}"

    except subprocess.TimeoutExpired as exc:
        stdout = _normalize_subprocess_output(exc.stdout)
        stderr = _normalize_subprocess_output(exc.stderr)
        timeout_message = f"TIMEOUT AFTER {timeout_seconds} SECONDS"
        stderr = f"{stderr}\n{timeout_message}" if stderr else timeout_message

        log_path = _write_stage_log(
            work_dir=work_dir,
            script_name=script_name,
            command=command,
            stdout=stdout,
            stderr=stderr,
            return_code=-1,
        )
        return (
            False,
            f"{script_name} 執行超過 {timeout_seconds} 秒，"
            f"已停止該筆樣本。Log: {log_path}",
        )

    except Exception as exc:
        error_message = f"執行 {script_name} 發生例外：{exc}"
        try:
            log_path = _write_stage_log(
                work_dir=work_dir,
                script_name=script_name,
                command=command,
                stdout="",
                stderr=error_message,
                return_code=-2,
            )
            error_message = f"{error_message} | Log: {log_path}"
        except Exception:
            # 即使 log 寫入也失敗，仍回傳原始錯誤，不遮蔽真正例外。
            pass
        return False, error_message


def _build_auto_sample_id() -> str:
    """單獨執行 main_pipeline.py 時使用高精度時間建立不重複 ID。"""
    return datetime.now().strftime("sample_%Y%m%d_%H%M%S_%f")


def _create_new_sample_directory(sample_id: str) -> str:
    """
    原子性建立新的 sample 資料夾。

    已存在時直接拒絕，絕不使用 exist_ok=True，避免覆蓋前一筆結果。
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    sample_dir = os.path.join(RUNS_DIR, sample_id)
    try:
        os.mkdir(sample_dir)
    except FileExistsError as exc:
        raise FileExistsError(
            "樣本資料夾已存在，為避免舊檔污染已停止執行："
            f"{sample_dir}"
        ) from exc

    return sample_dir


def main_pipeline_entry(
    sample_id: Optional[str] = None,
    run_name: str = "LHS_Sweep",
) -> Optional[Dict[str, Any]]:
    """
    全自動單次流水線入口。

    回傳 trace_record 給 batch_runner；不因 DRC/Q3D/LOM/Spec 失敗而直接
    結束整個批次，且不論成功失敗都嘗試歸檔。
    """
    if sample_id is None:
        sample_id = _build_auto_sample_id()

    source_json = os.path.join(ROOT_DIR, "layout_parameters.json")
    if not os.path.exists(source_json):
        print(f"❌ 根目錄找不到輸入參數：{source_json}")
        return None

    try:
        sample_dir = _create_new_sample_directory(sample_id)
    except FileExistsError as exc:
        print(f"❌ {exc}")
        raise

    destination_json = os.path.join(sample_dir, "layout_parameters.json")
    shutil.copy2(source_json, destination_json)
    parameter_hash = calculate_param_hash(destination_json)

    trace_record: Dict[str, Any] = {
        "sample_id": sample_id,
        "parameter_hash": parameter_hash,
        "run_name": run_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "drc_pass": 0,
        "q3d_success": 0,
        "lom_success": 0,
        "spec_pass": 0,
        "failure_stage": "None",
        "failure_reason": "None",
        "sample_dir": sample_dir,
    }

    print(
        f"\n🚀 啟動獨立流水線 -> "
        f"[ID: {sample_id}] [Hash: {parameter_hash}]"
    )

    for key, script_name, description in PIPELINE_CONFIGS:
        if not PIPELINE_SWITCHES.get(key, False):
            continue

        print(f" -> 正在執行：{description}")
        argument_input = (
            "layout_parameters.json"
            if key in {"gds_generator", "q3d_extraction"}
            else None
        )
        success, message = execute_sub_script(
            script_name=script_name,
            work_dir=sample_dir,
            arg_input=argument_input,
        )

        # 子程式即使錯誤地回傳 0，也必須確認必要輸出檔真的存在且非空。
        if success and key in EXPECTED_OUTPUTS:
            expected_path = os.path.join(sample_dir, EXPECTED_OUTPUTS[key])
            if (
                not os.path.isfile(expected_path)
                or os.path.getsize(expected_path) == 0
            ):
                success = False
                message = f"必要輸出不存在或為空檔：{expected_path}"

        if not success:
            print(f"❌ 階段攔截於 {key}：{message}")
            trace_record["failure_stage"] = key
            trace_record["failure_reason"] = message
            break

        if key == "drc_check":
            trace_record["drc_pass"] = 1
        elif key == "q3d_extraction":
            trace_record["q3d_success"] = 1
        elif key == "lom_bridge":
            trace_record["lom_success"] = 1
        elif key == "spec_check":
            trace_record["spec_pass"] = 1

    if PIPELINE_SWITCHES.get("db_manager", True):
        print(" -> 正在執行：5. 模擬軌跡與結果歸檔")
        try:
            # 延遲 import，避免主流程啟動時不必要地建立資料庫連線。
            import db_manager

            db_manager.archive_sample_folder(trace_record)
            print("✅ 資料庫歸檔完成。")
        except Exception as exc:
            print(f"❌ 資料庫歸檔發生異常：{exc}")
            if trace_record["failure_stage"] == "None":
                trace_record["failure_stage"] = "db_manager"
                trace_record["failure_reason"] = str(exc)

    return trace_record


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QCQ 單筆自動化流水線")
    parser.add_argument(
        "--sample-id",
        default=None,
        help="指定唯一 sample ID；省略時自動產生高精度時間 ID。",
    )
    parser.add_argument(
        "--run-name",
        default="Manual_Run",
        help="寫入資料庫的批次名稱。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    try:
        trace = main_pipeline_entry(
            sample_id=args.sample_id,
            run_name=args.run_name,
        )
    except FileExistsError:
        sys.exit(2)

    if trace is None:
        sys.exit(1)

    # 單獨執行時，以流水線是否中途失敗決定 shell return code。
    sys.exit(0 if trace["failure_stage"] == "None" else 1)