import json
import os
import re
import sqlite3
import sys
import time
import uuid
from typing import Any, Dict, Optional


# 固定標準矩陣節點順序：6 個浮接導體 + 1 個 GND。
CANONICAL_NODES = [
    "qubit1_left",
    "qubit1_right",
    "coupler1_left",
    "coupler1_right",
    "qubit2_left",
    "qubit2_right",
    "GND",
]

_MATRIX_PATTERN = re.compile(r"C\(([^,]+),([^)]+)\)")
_SAFE_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def normalize_q3d_node(raw_name: str) -> Optional[str]:
    """
    將 AEDT/Q3D 自動產生的節點名稱轉成固定名稱。

    例如：
        0_qubit1_left_1  -> qubit1_left
        6_GND_7          -> GND

    找不到對應節點時回傳 None，避免把未知導體誤存到資料庫。
    """
    raw_lower = str(raw_name).strip().lower()

    # 先匹配較長、較明確的導體名稱。
    for canonical in CANONICAL_NODES[:-1]:
        if canonical.lower() in raw_lower:
            return canonical

    if "gnd" in raw_lower:
        return "GND"

    return None


def _empty_matrix_elements() -> Dict[str, Optional[float]]:
    """建立 7x7 對稱矩陣的 28 個獨立上三角欄位；缺值使用 None。"""
    result: Dict[str, Optional[float]] = {}
    for i, node_1 in enumerate(CANONICAL_NODES):
        for j in range(i, len(CANONICAL_NODES)):
            node_2 = CANONICAL_NODES[j]
            result[f"C_{node_1}_vs_{node_2}"] = None
    return result


def parse_full_maxwell_elements(q3d_json_path: str) -> Dict[str, Any]:
    """
    將 Q3D JSON 對齊至固定的 7 節點 Maxwell 電容矩陣。

    回傳內容包含：
      - 28 個獨立矩陣元素，單位沿用 Q3D JSON（目前為 fF）
      - q3d_matrix_complete：是否取得全部 28 個元素
      - q3d_matrix_element_count：實際取得的元素數
      - q3d_node_count：成功辨識的標準節點數
      - q3d_unmapped_nodes_json：無法辨識的原始節點名稱
    """
    matrix_elements: Dict[str, Any] = _empty_matrix_elements()
    mapped_nodes = set()
    unmapped_nodes = set()

    if not os.path.exists(q3d_json_path):
        matrix_elements.update(
            {
                "q3d_matrix_complete": 0,
                "q3d_matrix_element_count": 0,
                "q3d_node_count": 0,
                "q3d_unmapped_nodes_json": "[]",
            }
        )
        return matrix_elements

    try:
        with open(q3d_json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        matrix_data = data.get("matrix_data")
        if not isinstance(matrix_data, dict):
            raise ValueError("Q3D JSON 缺少有效的 matrix_data 字典。")

        for raw_key, info in matrix_data.items():
            match = _MATRIX_PATTERN.fullmatch(str(raw_key).replace(" ", ""))
            if not match:
                continue

            raw_a, raw_b = match.group(1), match.group(2)
            node_a = normalize_q3d_node(raw_a)
            node_b = normalize_q3d_node(raw_b)

            if node_a is None:
                unmapped_nodes.add(raw_a)
            else:
                mapped_nodes.add(node_a)

            if node_b is None:
                unmapped_nodes.add(raw_b)
            else:
                mapped_nodes.add(node_b)

            if node_a is None or node_b is None:
                continue

            value = info.get("value") if isinstance(info, dict) else None
            if value is None:
                continue

            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue

            ordered_nodes = sorted(
                [node_a, node_b],
                key=CANONICAL_NODES.index,
            )
            field_name = f"C_{ordered_nodes[0]}_vs_{ordered_nodes[1]}"

            old_value = matrix_elements[field_name]
            if old_value is not None and abs(float(old_value) - value_float) > 1e-6:
                print(
                    "⚠️ Q3D 正反向矩陣元素不一致："
                    f"{field_name} 原值={old_value}, 新值={value_float}；採用新值。"
                )

            matrix_elements[field_name] = value_float

    except Exception as exc:
        print(f"⚠️ 解析原始 Q3D 矩陣時發生異常：{exc}")

    element_count = sum(
        value is not None
        for key, value in matrix_elements.items()
        if key.startswith("C_")
    )
    expected_count = len(CANONICAL_NODES) * (len(CANONICAL_NODES) + 1) // 2

    matrix_elements.update(
        {
            "q3d_matrix_complete": int(
                element_count == expected_count
                and len(mapped_nodes) == len(CANONICAL_NODES)
            ),
            "q3d_matrix_element_count": element_count,
            "q3d_node_count": len(mapped_nodes),
            "q3d_unmapped_nodes_json": json.dumps(
                sorted(unmapped_nodes), ensure_ascii=False
            ),
        }
    )
    return matrix_elements


def _flatten_numeric_sections(data: Dict[str, Any]) -> Dict[str, Any]:
    """保存目前代理模型會使用的幾何與固定模擬條件。"""
    result: Dict[str, Any] = {}
    sections = [
        "toggles",
        "global",
        "qubit",
        "coupler",
        "t_coupler",
        "h_coupler",
        "substrate",
        "Qubit_pra",
        "simulation",
        "lom_settings",
    ]

    def flatten(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(child, f"{prefix}_{key}" if prefix else str(key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[f"in_{prefix}"] = value

    for section in sections:
        if section in data:
            flatten(data[section], section)

    return result


def _sqlite_type(column_name: str, value: Any) -> str:
    if column_name.startswith("C_"):
        return "REAL"
    if column_name in {
        "drc_pass",
        "q3d_success",
        "lom_success",
        "spec_pass",
        "q3d_matrix_complete",
        "q3d_matrix_element_count",
        "q3d_node_count",
    }:
        return "INTEGER"
    if isinstance(value, bool):
        return "INTEGER"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        return "REAL"
    return "TEXT"


def _validate_identifier(identifier: str) -> str:
    if not _SAFE_SQL_IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"不安全的 SQLite 欄位名稱：{identifier}")
    return identifier


_MISSING = object()


def _get_nested_value(data: Dict[str, Any], path: list) -> Any:
    """依照 db_config.json 的鍵路徑讀取巢狀 JSON 值。"""
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _extract_selected_fields(
    data: Dict[str, Any],
    selections: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    依照 db_config.json 的 selected_inputs/selected_outputs 擷取欄位。

    設定中的 alias 會轉為 SQLite 欄位名稱，例如：
      qubit_gap -> in_qubit_gap
      EC1_MHz   -> out_EC1_MHz
    """
    result: Dict[str, Any] = {}

    for alias, path in selections.items():
        if (
            not isinstance(path, list)
            or not path
            or not all(isinstance(key, str) and key for key in path)
        ):
            raise ValueError(
                f"db_config.json 欄位 {alias!r} 的路徑必須是非空字串陣列。"
            )

        column_name = _validate_identifier(f"{prefix}_{alias}")
        value = _get_nested_value(data, path)

        if value is _MISSING:
            print(
                f"⚠️ 選定欄位 {column_name} 找不到路徑 "
                f"{'.'.join(path)}，本筆將寫入 NULL。"
            )
            result[column_name] = None
        elif isinstance(value, (dict, list, tuple)):
            result[column_name] = json.dumps(value, ensure_ascii=False)
        else:
            result[column_name] = value

    return result


def archive_sample_folder(trace_record: Dict[str, Any]) -> bool:
    """
    讀取單一 sample 工作資料夾內的成果物，並將完整資料寫入 SQLite。
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(module_dir, "db_config.json")

    config_data: Dict[str, Any] = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                loaded_config = json.load(file)
            if not isinstance(loaded_config, dict):
                raise ValueError("設定檔最外層必須是 JSON 物件。")
            config_data = loaded_config
        except Exception as exc:
            print(f"⚠️ 讀取 db_config.json 失敗，採用預設設定：{exc}")

    db_cfg = config_data.get("database", {})
    if not isinstance(db_cfg, dict):
        raise ValueError("db_config.json 的 database 必須是 JSON 物件。")

    db_name = str(db_cfg.get("db_name", "quantum_simulation.db"))
    table_name = _validate_identifier(
        str(db_cfg.get("table_name", "simulation_records"))
    )

    selected_inputs = config_data.get("selected_inputs", {})
    selected_outputs = config_data.get("selected_outputs", {})
    if not isinstance(selected_inputs, dict):
        raise ValueError("db_config.json 的 selected_inputs 必須是 JSON 物件。")
    if not isinstance(selected_outputs, dict):
        raise ValueError("db_config.json 的 selected_outputs 必須是 JSON 物件。")

    sample_dir = os.path.abspath(trace_record["sample_dir"])
    layout_path = os.path.join(sample_dir, "layout_parameters.json")
    q3d_path = os.path.join(sample_dir, "capacitance_matrix_results.json")
    lom_path = os.path.join(sample_dir, "lom_results.json")
    spec_path = os.path.join(sample_dir, "spec_results.json")

    input_flat: Dict[str, Any] = {}
    if os.path.exists(layout_path):
        with open(layout_path, "r", encoding="utf-8") as file:
            layout_data = json.load(file)

        if selected_inputs:
            input_flat = _extract_selected_fields(
                layout_data,
                selected_inputs,
                "in",
            )
        else:
            input_flat = _flatten_numeric_sections(layout_data)

    q3d_elements = parse_full_maxwell_elements(q3d_path)

    lom_flat: Dict[str, Any] = {}
    if os.path.exists(lom_path):
        try:
            with open(lom_path, "r", encoding="utf-8") as file:
                lom_data = json.load(file)

            if selected_outputs:
                lom_flat = _extract_selected_fields(
                    lom_data,
                    selected_outputs,
                    "out",
                )
            else:
                qubit_params = lom_data.get("qubits_parameters", {})
                coupling_params = lom_data.get("coupling_parameters_MHz", {})

                for mode_name in ["Qubit_1", "Coupler", "Qubit_2"]:
                    mode_data = qubit_params.get(mode_name)
                    if isinstance(mode_data, dict):
                        lom_flat[f"lom_{mode_name}_C_eff"] = mode_data.get(
                            "C_eff_fF"
                        )
                        lom_flat[f"lom_{mode_name}_Ec"] = mode_data.get("Ec_MHz")
                        lom_flat[f"lom_{mode_name}_Ej"] = mode_data.get("Ej_GHz")
                        lom_flat[f"lom_{mode_name}_Ej_Ec_ratio"] = mode_data.get(
                            "Ej_Ec_ratio"
                        )

                lom_flat["lom_g12"] = coupling_params.get("g_12")
                lom_flat["lom_g1c"] = coupling_params.get("g_1c")
                lom_flat["lom_g2c"] = coupling_params.get("g_2c")
        except Exception as exc:
            print(f"⚠️ 讀取 LOM 結果失敗：{exc}")

    spec_flat: Dict[str, Any] = {}
    if os.path.exists(spec_path):
        try:
            with open(spec_path, "r", encoding="utf-8") as file:
                spec_data = json.load(file)
            spec_flat = {
                f"spec_{key}": value
                for key, value in spec_data.items()
                if key != "spec_pass"
            }
            spec_flat["spec_pass"] = int(spec_data.get("spec_pass", 0))
        except Exception as exc:
            print(f"⚠️ 讀取 Spec 結果失敗：{exc}")

    master_row: Dict[str, Any] = {}
    master_row.update(trace_record)
    master_row.update(input_flat)
    master_row.update(q3d_elements)
    master_row.update(lom_flat)
    master_row.update(spec_flat)
    master_row.pop("sample_dir", None)

    for key, value in list(master_row.items()):
        if isinstance(value, (dict, list, tuple)):
            master_row[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            master_row[key] = int(value)

    db_path = os.path.join(module_dir, db_name)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                sample_id TEXT PRIMARY KEY,
                parameter_hash TEXT,
                run_name TEXT,
                timestamp TEXT,
                drc_pass INTEGER,
                q3d_success INTEGER,
                lom_success INTEGER,
                spec_pass INTEGER,
                failure_stage TEXT,
                failure_reason TEXT
            )
            """
        )

        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {column[1] for column in cursor.fetchall()}

        for column_name, value in master_row.items():
            _validate_identifier(column_name)
            if column_name not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE {table_name} "
                    f"ADD COLUMN {column_name} {_sqlite_type(column_name, value)}"
                )
                existing_columns.add(column_name)

        column_names = list(master_row.keys())
        for column_name in column_names:
            _validate_identifier(column_name)

        columns_sql = ", ".join(column_names)
        placeholders = ", ".join("?" for _ in column_names)
        insert_sql = (
            f"INSERT INTO {table_name} ({columns_sql}) "
            f"VALUES ({placeholders})"
        )

        cursor.execute(insert_sql, [master_row[name] for name in column_names])
        conn.commit()

        print(
            "💾 樣本已成功寫入資料庫：\n"
            f"   - 資料庫路徑: {db_path}\n"
            f"   - 資料表: {table_name}\n"
            f"   - Sample ID: {trace_record.get('sample_id')}\n"
            f"   - Q3D矩陣完整: {q3d_elements['q3d_matrix_complete']}"
        )
        return True

    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise RuntimeError(
            "資料庫已存在相同 sample_id，已拒絕寫入："
            f"{trace_record.get('sample_id')}"
        ) from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    current_dir = os.getcwd()
    trace_record = {
        "sample_id": f"LHS_{int(time.time())}_{uuid.uuid4().hex[:6]}",
        "sample_dir": current_dir,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": "Auto_Pipeline_Run",
        "spec_pass": 0,
        "failure_stage": None,
        "failure_reason": None,
    }

    try:
        success = archive_sample_folder(trace_record)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ 資料庫歸檔失敗: {e}")
        sys.exit(1)