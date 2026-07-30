import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


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


def normalize_q3d_node(raw_name: str) -> Optional[str]:
    """將 Q3D 自動節點名稱正規化為固定 QCQ 節點名稱。"""
    raw_lower = str(raw_name).strip().lower()
    for canonical in CANONICAL_NODES[:-1]:
        if canonical.lower() in raw_lower:
            return canonical
    if "gnd" in raw_lower:
        return "GND"
    return None


def parse_q3d_json(json_path: str):
    """
    解析 Q3D JSON，回傳完整 7x7 Maxwell 電容矩陣（單位 F）與固定節點順序。

    不再使用 0 補齊缺失元素；只要節點或 28 個獨立元素不完整就直接拋出錯誤，
    防止 LOM 使用殘缺矩陣卻被誤判為成功。
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"找不到 Q3D 結果檔：{json_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    matrix_data = data.get("matrix_data")
    if not isinstance(matrix_data, dict) or not matrix_data:
        raise ValueError("Q3D JSON 缺少有效的 matrix_data。")

    pair_values: Dict[Tuple[int, int], List[float]] = {}
    recognized_nodes = set()
    unmapped_nodes = set()

    for trace_name, info in matrix_data.items():
        match = _MATRIX_PATTERN.fullmatch(str(trace_name).replace(" ", ""))
        if not match:
            continue

        raw_node_1, raw_node_2 = match.group(1), match.group(2)
        node_1 = normalize_q3d_node(raw_node_1)
        node_2 = normalize_q3d_node(raw_node_2)

        if node_1 is None:
            unmapped_nodes.add(raw_node_1)
        else:
            recognized_nodes.add(node_1)

        if node_2 is None:
            unmapped_nodes.add(raw_node_2)
        else:
            recognized_nodes.add(node_2)

        if node_1 is None or node_2 is None:
            continue

        if not isinstance(info, dict) or "value" not in info:
            continue

        try:
            value_ff = float(info["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Q3D trace {trace_name} 的 value 不是有效數字。"
            ) from exc

        if not np.isfinite(value_ff):
            raise ValueError(f"Q3D trace {trace_name} 含有 NaN 或 Inf。")

        index_1 = CANONICAL_NODES.index(node_1)
        index_2 = CANONICAL_NODES.index(node_2)
        pair_key = tuple(sorted((index_1, index_2)))
        pair_values.setdefault(pair_key, []).append(value_ff * 1e-15)

    expected_pair_count = len(CANONICAL_NODES) * (len(CANONICAL_NODES) + 1) // 2

    if len(recognized_nodes) != len(CANONICAL_NODES):
        missing_nodes = sorted(set(CANONICAL_NODES) - recognized_nodes)
        raise ValueError(
            "Q3D節點不完整："
            f"辨識到 {len(recognized_nodes)}/7；"
            f"缺少={missing_nodes}；未知={sorted(unmapped_nodes)}"
        )

    if len(pair_values) != expected_pair_count:
        missing_pairs = []
        for i, node_1 in enumerate(CANONICAL_NODES):
            for j in range(i, len(CANONICAL_NODES)):
                if (i, j) not in pair_values:
                    missing_pairs.append(f"C({node_1},{CANONICAL_NODES[j]})")
        raise ValueError(
            "Q3D矩陣元素不完整："
            f"取得 {len(pair_values)}/{expected_pair_count}；"
            f"缺少={missing_pairs}"
        )

    capacitance_matrix = np.empty(
        (len(CANONICAL_NODES), len(CANONICAL_NODES)),
        dtype=float,
    )

    for (index_1, index_2), values in pair_values.items():
        # 若 Q3D 同時輸出 C(a,b) 與 C(b,a)，使用平均值降低微小數值差異。
        mean_value = float(np.mean(values))
        if len(values) > 1 and np.ptp(values) > 1e-21:
            print(
                "⚠️ Q3D 正反向矩陣元素略有差異，採用平均值："
                f"{CANONICAL_NODES[index_1]} vs "
                f"{CANONICAL_NODES[index_2]}"
            )
        capacitance_matrix[index_1, index_2] = mean_value
        capacitance_matrix[index_2, index_1] = mean_value

    if not np.all(np.isfinite(capacitance_matrix)):
        raise ValueError("Q3D矩陣仍含有 NaN 或 Inf。")

    if not np.allclose(
        capacitance_matrix,
        capacitance_matrix.T,
        rtol=1e-10,
        atol=1e-24,
    ):
        raise ValueError("Q3D電容矩陣不是對稱矩陣。")

    return capacitance_matrix, CANONICAL_NODES.copy()


def calculate_lom_parameters(CM, nodes, w_vector, h_bar, e):
    """執行 LOM 差模轉換、Schur complement、Ec/Ej/g 計算。"""
    CM = np.asarray(CM, dtype=float)
    w_vector = np.asarray(w_vector, dtype=float)

    if CM.shape != (7, 7):
        raise ValueError(f"LOM需要 7x7 電容矩陣，目前形狀為 {CM.shape}。")
    if list(nodes) != CANONICAL_NODES:
        raise ValueError(
            f"LOM節點順序不正確。預期={CANONICAL_NODES}，實際={nodes}"
        )
    if w_vector.shape != (3,) or not np.all(np.isfinite(w_vector)):
        raise ValueError("w_vector 必須包含有限的 [w1, wc, w2] 三個頻率。")
    if np.any(w_vector <= 0):
        raise ValueError("w1、wc、w2 必須全部大於 0 GHz。")
    if h_bar <= 0 or e <= 0:
        raise ValueError("h_bar 與 e 必須為正值。")

    print(f"🔍 LOM固定節點順序：{nodes}")

    # GND固定放在最後一列/欄，先刪除 GND。
    reduced_matrix = CM[:-1, :-1]

    id_q1_physical = [0, 1]
    id_c_physical = [2, 3]
    id_q2_physical = [4, 5]

    transform_matrix = np.eye(6)
    differential_common_transform = np.array([[1.0, -1.0], [1.0, 1.0]])

    transform_matrix[
        np.ix_(id_q1_physical, id_q1_physical)
    ] = differential_common_transform
    transform_matrix[
        np.ix_(id_c_physical, id_c_physical)
    ] = differential_common_transform
    transform_matrix[
        np.ix_(id_q2_physical, id_q2_physical)
    ] = differential_common_transform

    try:
        transform_inverse = np.linalg.inv(transform_matrix)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("差模/共模轉換矩陣不可逆。") from exc

    mode_matrix = transform_inverse.T @ reduced_matrix @ transform_inverse

    # 每一對節點的第一個 index 為目前定義下的差模。
    differential_indices = [
        id_q1_physical[0],
        id_c_physical[0],
        id_q2_physical[0],
    ]
    floating_common_indices = [
        index for index in range(6) if index not in differential_indices
    ]

    reorder_indices = differential_indices + floating_common_indices
    reordered_matrix = mode_matrix[np.ix_(reorder_indices, reorder_indices)]

    mode_count = len(differential_indices)
    Cqq = reordered_matrix[:mode_count, :mode_count]
    Cqx = reordered_matrix[:mode_count, mode_count:]
    Cxq = reordered_matrix[mode_count:, :mode_count]
    Cxx = reordered_matrix[mode_count:, mode_count:]

    try:
        effective_capacitance = Cqq - Cqx @ np.linalg.solve(Cxx, Cxq)
        capacitance_inverse = np.linalg.inv(effective_capacitance)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Schur complement 或有效電容矩陣求逆失敗；"
            "請檢查Q3D矩陣是否完整且物理合理。"
        ) from exc

    if not np.all(np.isfinite(effective_capacitance)):
        raise RuntimeError("有效電容矩陣含有 NaN 或 Inf。")
    if not np.allclose(
        effective_capacitance,
        effective_capacitance.T,
        rtol=1e-9,
        atol=1e-24,
    ):
        raise RuntimeError("Schur complement 後的有效電容矩陣不對稱。")

    effective_eigenvalues = np.linalg.eigvalsh(effective_capacitance)
    if np.any(effective_eigenvalues <= 0):
        raise RuntimeError(
            "有效電容矩陣不是正定矩陣；"
            f"特徵值={effective_eigenvalues.tolist()}"
        )

    inverse_diagonal = np.diag(capacitance_inverse)
    if np.any(inverse_diagonal <= 0):
        raise RuntimeError("有效電容逆矩陣的對角元素必須為正。")

    mode_capacitances = 1.0 / inverse_diagonal
    if np.any(mode_capacitances <= 0):
        raise RuntimeError("計算得到的 mode capacitance 必須為正。")

    output_data = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "node_order": nodes,
            "system_frequencies_GHz": {
                "w1": float(w_vector[0]),
                "wc": float(w_vector[1]),
                "w2": float(w_vector[2]),
            },
        },
        "effective_capacitance_matrix_fF": (
            effective_capacitance / 1e-15
        ).tolist(),
        "qubits_parameters": {},
        "coupling_parameters_MHz": {},
    }

    print("\n" + "=" * 50)
    print("📊 LOM 量子參數計算結果")
    print("=" * 50)

    mode_names = ["Qubit_1", "Coupler", "Qubit_2"]

    for index, mode_name in enumerate(mode_names):
        capacitance_value = mode_capacitances[index]
        ec_ghz = (
            0.5
            * e**2
            / capacitance_value
            / h_bar
            / (2.0 * np.pi)
            / 1e9
        )

        if not np.isfinite(ec_ghz) or ec_ghz <= 0:
            raise RuntimeError(f"{mode_name} 的 Ec 無效：{ec_ghz}")

        target_frequency = w_vector[index]
        ej_ghz = ((target_frequency + ec_ghz) ** 2) / (8.0 * ec_ghz)
        ej_ec_ratio = ej_ghz / ec_ghz

        output_data["qubits_parameters"][mode_name] = {
            "C_eff_fF": round(float(capacitance_value / 1e-15), 6),
            "Ec_MHz": round(float(ec_ghz * 1000.0), 6),
            "Ej_GHz": round(float(ej_ghz), 6),
            "Ej_Ec_ratio": round(float(ej_ec_ratio), 4),
        }

        print(
            f"[{mode_name:<9}] "
            f"C_eff={capacitance_value / 1e-15:>9.4f} fF | "
            f"Ec={ec_ghz * 1000.0:>9.4f} MHz | "
            f"Ej={ej_ghz:>9.5f} GHz | "
            f"Ej/Ec={ej_ec_ratio:>8.3f}"
        )

    coupling_matrix = np.zeros_like(capacitance_inverse)
    for row in range(mode_count):
        for column in range(mode_count):
            if row == column:
                continue

            denominator = np.sqrt(
                capacitance_inverse[row, row]
                * capacitance_inverse[column, column]
            )
            coupling_matrix[row, column] = (
                0.5
                * capacitance_inverse[row, column]
                / denominator
                * np.sqrt(w_vector[row] * w_vector[column])
            )

    g_12_mhz = float(coupling_matrix[0, 2] * 1000.0)
    g_1c_mhz = float(coupling_matrix[0, 1] * 1000.0)
    g_2c_mhz = float(coupling_matrix[2, 1] * 1000.0)

    if not np.all(np.isfinite([g_12_mhz, g_1c_mhz, g_2c_mhz])):
        raise RuntimeError("計算出的耦合強度含有 NaN 或 Inf。")

    output_data["coupling_parameters_MHz"] = {
        "g_12": round(g_12_mhz, 6),
        "g_1c": round(g_1c_mhz, 6),
        "g_2c": round(g_2c_mhz, 6),
    }

    print("-" * 50)
    print(f"g_12 (Q1-Q2) = {g_12_mhz:.6f} MHz")
    print(f"g_1c (Q1-C)  = {g_1c_mhz:.6f} MHz")
    print(f"g_2c (Q2-C)  = {g_2c_mhz:.6f} MHz")
    print("=" * 50)

    return output_data


def run_lom_bridge(
    layout_json_path: str = "layout_parameters.json",
    q3d_json_path: str = "capacitance_matrix_results.json",
    output_json_path: str = "lom_results.json",
) -> int:
    """
    執行完整 LOM 計算並以非零 return code 正確回報失敗。

    Return code：
        0 成功
        1 layout_parameters.json 缺失或無效
        2 Q3D JSON 缺失或矩陣不完整
        3 LOM矩陣/物理計算失敗
        4 lom_results.json 寫出失敗
    """
    layout_json_path = os.path.abspath(layout_json_path)
    q3d_json_path = os.path.abspath(q3d_json_path)
    output_json_path = os.path.abspath(output_json_path)

    # 禁止使用前一次殘留的 LOM 結果。
    if os.path.exists(output_json_path):
        os.remove(output_json_path)

    try:
        with open(layout_json_path, "r", encoding="utf-8") as file:
            layout_data = json.load(file)

        lom_settings = layout_data.get("lom_settings", {})
        constants = lom_settings.get("constants", {})
        frequencies = lom_settings.get("frequencies", {})

        h_bar = float(constants.get("h_bar", 1.05457180013e-34))
        electron_charge = float(constants.get("e", 1.602176620898e-19))
        w_1 = float(frequencies.get("w1", 4.58))
        w_c = float(frequencies.get("wc", 6.14))
        w_2 = float(frequencies.get("w2", 4.64))
        frequency_vector = [w_1, w_c, w_2]

    except FileNotFoundError:
        print(
            f"❌ 找不到 layout 參數檔：{layout_json_path}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"❌ 讀取 layout/LOM 設定失敗：{exc}", file=sys.stderr)
        return 1

    try:
        capacitance_matrix, nodes = parse_q3d_json(q3d_json_path)
    except Exception as exc:
        print(f"❌ Q3D矩陣解析失敗：{exc}", file=sys.stderr)
        return 2

    try:
        final_results = calculate_lom_parameters(
            capacitance_matrix,
            nodes,
            frequency_vector,
            h_bar,
            electron_charge,
        )
    except Exception as exc:
        print(f"❌ LOM計算失敗：{exc}", file=sys.stderr)
        return 3

    if not isinstance(final_results, dict) or not final_results:
        print("❌ LOM沒有產生有效結果。", file=sys.stderr)
        return 3

    try:
        with open(output_json_path, "w", encoding="utf-8") as file:
            json.dump(final_results, file, indent=4, ensure_ascii=False)

        if not os.path.isfile(output_json_path) or os.path.getsize(output_json_path) == 0:
            raise IOError("輸出檔不存在或為空檔。")

    except Exception as exc:
        print(f"❌ 寫出 LOM JSON 失敗：{exc}", file=sys.stderr)
        return 4

    print(f"\n🎉 LOM量子參數已成功輸出至：{output_json_path}")
    return 0


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QCQ LOM 計算橋接器")
    parser.add_argument(
        "-i",
        "--input",
        default="layout_parameters.json",
        help="layout_parameters.json 路徑。",
    )
    parser.add_argument(
        "-q",
        "--q3d",
        default="capacitance_matrix_results.json",
        help="Q3D矩陣JSON路徑。",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="lom_results.json",
        help="LOM輸出JSON路徑。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    sys.exit(
        run_lom_bridge(
            layout_json_path=args.input,
            q3d_json_path=args.q3d,
            output_json_path=args.output,
        )
    )