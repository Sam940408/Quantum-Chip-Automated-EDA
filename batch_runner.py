import copy
import json
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from scipy.stats import qmc

try:
    import gds_json_
    import main_pipeline
except ImportError as exc:
    print(
        "❌ 找不到 gds_json_.py 或 main_pipeline.py，"
        "請確認它們與 batch_runner.py 位於同一資料夾。"
    )
    raise SystemExit(1) from exc


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_JSON_FILE = os.path.join(ROOT_DIR, "layout_parameters.json")
RUNS_DIR = os.path.join(ROOT_DIR, "runs")
SURROGATE_DB = os.path.join(ROOT_DIR, "quantum_simulation_surrogate.db")
SAMPLE_ID_PATTERN = re.compile(r"^sample_(\d+)$")

# =========================================================================
# 參數範圍限制設定
# =========================================================================
PARAM_LIMITS = {
    "qubit": {
        "gap_size": {"min": 30.0, "max": 50.0},
        "radius": {"min": 50.0, "max": 500.0},
        "slit_width": {"min": 50.0, "max": 70.0},
        "rect_length": {"min": 50.0, "max": 1000.0},
        "rect_width": {"min": 50.0, "max": 1000.0},
        "q_c_dis": {"min": 70.0, "max": 120.0},
    },
    "coupler": {
        "arc_width": {"min": 10.0, "max": 100.0},
        "gap_size": {"min": 30.0, "max": 50.0},
        "center_dis": {"min": 5.0, "max": 15.0},
        "length": {"min": 50.0, "max": 1000.0},
        "round_radius": {"min": 0.0, "max": 20.0},
    },
    "t_coupler": {
        "arm_length": {"min": 50.0, "max": 500.0},
        "arm_width": {"min": 5.0, "max": 100.0},
        "head_width": {"min": 10.0, "max": 200.0},
        "gap_size": {"min": 30.0, "max": 50.0},
        "center_dis": {"min": 5.0, "max": 15.0},
        "round_radius": {"min": 0.0, "max": 20.0},
    },
    "h_coupler": {
        "arm_length": {"min": 50.0, "max": 500.0},
        "head1_width": {"min": 10.0, "max": 200.0},
        "head2_width": {"min": 10.0, "max": 200.0},
        "gap_size": {"min": 30.0, "max": 50.0},
        "center_dis": {"min": 25.0, "max": 35.0},
        # min == max 代表固定參數，不參與 LHS。
        "round_radius": {"min": 10.0, "max": 10.0},
    },
}


def get_pruned_limits(base_json_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """依目前 qubit/coupler 形狀修剪參數空間。"""
    current_limits = copy.deepcopy(PARAM_LIMITS)
    toggles = base_json_data.get("toggles", {})
    qubit_type = toggles.get("qubit_type", "rect")
    coupler_type = toggles.get("coupler_type", "arc")

    if "qubit" in current_limits:
        if qubit_type == "rect":
            current_limits["qubit"].pop("radius", None)
        elif qubit_type == "circle":
            current_limits["qubit"].pop("rect_length", None)
            current_limits["qubit"].pop("rect_width", None)

    if coupler_type == "arc":
        current_limits.pop("t_coupler", None)
        current_limits.pop("h_coupler", None)
    elif coupler_type == "t_shape":
        current_limits.pop("coupler", None)
        current_limits.pop("h_coupler", None)
    elif coupler_type == "h_shape":
        current_limits.pop("coupler", None)
        current_limits.pop("t_coupler", None)
    else:
        raise ValueError(f"不支援的 coupler_type：{coupler_type}")

    flat_limits: Dict[str, Dict[str, float]] = {}

    def flatten(data: Dict[str, Any], prefix: str = "") -> None:
        for key, value in data.items():
            full_key = f"{prefix}{key}"
            if isinstance(value, dict) and {"min", "max"}.issubset(value):
                minimum = float(value["min"])
                maximum = float(value["max"])
                if minimum < maximum:
                    flat_limits[full_key] = value
                else:
                    print(
                        f"⚠️ [{full_key}] min={minimum} >= max={maximum}，"
                        "視為固定值，不加入 LHS。"
                    )
            elif isinstance(value, dict):
                flatten(value, prefix=f"{full_key}.")

    flatten(current_limits)
    return flat_limits


def generate_lhs_samples(
    num_samples: int,
    flat_limits: Dict[str, Dict[str, float]],
    seed: Optional[int] = None,
) -> Tuple[List[str], Any]:
    """建立可重現的 Latin Hypercube 樣本。"""
    dimension = len(flat_limits)
    if dimension == 0:
        return [], []

    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    unit_samples = sampler.random(n=num_samples)

    keys = list(flat_limits.keys())
    lower_bounds = [float(flat_limits[key]["min"]) for key in keys]
    upper_bounds = [float(flat_limits[key]["max"]) for key in keys]
    scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)
    return keys, scaled_samples


def apply_single_lhs_sample(
    target_dict: Dict[str, Any],
    keys: List[str],
    sample_row: Any,
    flat_limits: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """將一列 LHS 數值注入巢狀 layout JSON。"""
    updated_records: Dict[str, Any] = {}

    for index, key_path in enumerate(keys):
        value = sample_row[index]
        if flat_limits[key_path].get("type") == "int":
            value = int(round(value))
        else:
            value = round(float(value), 3)

        path_parts = key_path.split(".")
        current = target_dict
        for path_part in path_parts[:-1]:
            current = current.setdefault(path_part, {})
        current[path_parts[-1]] = value
        updated_records[key_path] = value

    return updated_records


def find_next_sample_index() -> int:
    """
    同時掃描 runs 資料夾與 SQLite，找出下一個可用的流水號。

    即使使用者刪除了 runs 資料夾但保留資料庫，也不會重用舊 sample_id。
    """
    used_indices = set()

    if os.path.isdir(RUNS_DIR):
        for name in os.listdir(RUNS_DIR):
            match = SAMPLE_ID_PATTERN.fullmatch(name)
            if match:
                used_indices.add(int(match.group(1)))

    if os.path.exists(SURROGATE_DB):
        try:
            connection = sqlite3.connect(SURROGATE_DB)
            try:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='simulation_records'"
                )
                if cursor.fetchone():
                    cursor.execute("SELECT sample_id FROM simulation_records")
                    for (sample_id,) in cursor.fetchall():
                        match = SAMPLE_ID_PATTERN.fullmatch(str(sample_id))
                        if match:
                            used_indices.add(int(match.group(1)))
            finally:
                connection.close()
        except sqlite3.Error as exc:
            print(f"⚠️ 無法掃描既有代理模型資料庫：{exc}")

    return max(used_indices, default=0) + 1


def run_batch(
    max_iterations: int = 10_000,
    run_name: str = "LHS_Sweep",
    seed: Optional[int] = 940408,
    failure_pause_seconds: float = 2.0,
) -> None:
    """執行 LHS 批次探索，每一筆都使用唯一 sample_id。"""
    if max_iterations <= 0:
        raise ValueError("max_iterations 必須大於 0。")

    os.chdir(ROOT_DIR)

    print(
        f"🔄 啟動 LHS 參數空間探索，預計執行 {max_iterations} 筆；"
        f"seed={seed}"
    )

    gds_json_.generate_layout_json(TARGET_JSON_FILE, print_content=False)
    with open(TARGET_JSON_FILE, "r", encoding="utf-8") as file:
        base_config = json.load(file)

    flat_limits = get_pruned_limits(base_config)
    keys, lhs_matrix = generate_lhs_samples(
        max_iterations,
        flat_limits,
        seed=seed,
    )

    if not keys:
        print("❌ 找不到有效參數範圍，請檢查 PARAM_LIMITS。")
        return

    first_sample_index = find_next_sample_index()
    print(
        f"📐 LHS矩陣維度：{lhs_matrix.shape}；"
        f"第一筆 ID：sample_{first_sample_index:06d}"
    )

    success_count = 0
    failure_count = 0

    try:
        for local_index in range(max_iterations):
            sample_number = first_sample_index + local_index
            sample_id = f"sample_{sample_number:06d}"

            print("\n" + "=" * 80)
            print(
                f"▶️ 第 {local_index + 1}/{max_iterations} 筆 | "
                f"{sample_id}"
            )
            print("=" * 80)

            # 每次都重新生成乾淨的預設 JSON，避免上一筆參數殘留。
            gds_json_.generate_layout_json(TARGET_JSON_FILE, print_content=False)
            with open(TARGET_JSON_FILE, "r", encoding="utf-8") as file:
                config_data = json.load(file)

            updated_parameters = apply_single_lhs_sample(
                config_data,
                keys,
                lhs_matrix[local_index],
                flat_limits,
            )

            print("🎲 本次注入參數：")
            for key, value in updated_parameters.items():
                print(f"   - {key}: {value}")

            with open(TARGET_JSON_FILE, "w", encoding="utf-8") as file:
                json.dump(config_data, file, indent=4, ensure_ascii=False)

            start_time = time.time()
            try:
                trace = main_pipeline.main_pipeline_entry(
                    sample_id=sample_id,
                    run_name=run_name,
                )
            except FileExistsError as exc:
                # 正常情況不應發生；發生時停止，避免流水號邏輯失控。
                print(f"❌ {exc}")
                raise RuntimeError(
                    "偵測到 sample_id 衝突，批次已停止以避免覆蓋資料。"
                ) from exc

            elapsed = time.time() - start_time

            if trace is None:
                failure_count += 1
                print(f"❌ {sample_id} 無法啟動流水線。耗時 {elapsed:.2f} 秒")
            elif trace["failure_stage"] == "None":
                success_count += 1
                print(f"✅ {sample_id} 全流程完成。耗時 {elapsed:.2f} 秒")
            else:
                failure_count += 1
                print(
                    f"⚠️ {sample_id} 失敗於 {trace['failure_stage']}："
                    f"{trace['failure_reason']}"
                )
                print(f"   該失敗樣本仍已嘗試歸檔。耗時 {elapsed:.2f} 秒")
                if failure_pause_seconds > 0:
                    time.sleep(failure_pause_seconds)

    except KeyboardInterrupt:
        print("\n🛑 使用者中止批次。已完成的樣本不會被刪除。")

    finally:
        completed = success_count + failure_count
        print("\n" + "=" * 80)
        print("📊 批次執行摘要")
        print(f"   已處理：{completed}/{max_iterations}")
        print(f"   全流程通過：{success_count}")
        print(f"   中途失敗但已歸檔：{failure_count}")
        print("=" * 80)


if __name__ == "__main__":
    # 先用 2 筆做小型驗證；確認資料夾與 SQLite 正常後再改成 5000。
    run_batch(max_iterations=2, run_name="LHS_Sweep_Test", seed=940408)