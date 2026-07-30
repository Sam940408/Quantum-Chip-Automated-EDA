import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import List, Optional, Tuple

from pyaedt import Q3d


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
    """將 AEDT 自動節點名稱轉為固定 QCQ 節點名稱。"""
    raw_lower = str(raw_name).strip().lower()
    for canonical in CANONICAL_NODES[:-1]:
        if canonical.lower() in raw_lower:
            return canonical
    if "gnd" in raw_lower:
        return "GND"
    return None


def _validate_exported_matrix(export_data: dict) -> Tuple[bool, int, int, List[str]]:
    """驗證 7 節點 Maxwell 矩陣是否具有完整的 28 個獨立元素。"""
    recognized_nodes = set()
    independent_pairs = set()
    unmapped_nodes = set()

    for trace_name in export_data.get("matrix_data", {}):
        match = _MATRIX_PATTERN.fullmatch(str(trace_name).replace(" ", ""))
        if not match:
            continue

        raw_a, raw_b = match.group(1), match.group(2)
        node_a = normalize_q3d_node(raw_a)
        node_b = normalize_q3d_node(raw_b)

        if node_a is None:
            unmapped_nodes.add(raw_a)
        else:
            recognized_nodes.add(node_a)

        if node_b is None:
            unmapped_nodes.add(raw_b)
        else:
            recognized_nodes.add(node_b)

        if node_a is None or node_b is None:
            continue

        index_a = CANONICAL_NODES.index(node_a)
        index_b = CANONICAL_NODES.index(node_b)
        independent_pairs.add(tuple(sorted((index_a, index_b))))

    expected_elements = len(CANONICAL_NODES) * (len(CANONICAL_NODES) + 1) // 2
    is_complete = (
        len(recognized_nodes) == len(CANONICAL_NODES)
        and len(independent_pairs) == expected_elements
    )

    return (
        is_complete,
        len(recognized_nodes),
        len(independent_pairs),
        sorted(unmapped_nodes),
    )


def run_q3d_extraction(json_file: str) -> int:
    """
    匯入 GDS、執行 Q3D 並輸出 Maxwell 電容矩陣。

    Return code：
        0  成功且矩陣完整
        1  輸入 JSON 或 GDS 不存在/無法讀取
        2  缺少 layer_mapping
        3  Q3D/AEDT 初始化或建模失敗
        4  沒有取得任何電容 trace
        5  無法取得 solution data
        6  輸出的 7 節點矩陣不完整
        7  trace 數值提取失敗
        8  JSON 寫出失敗
        9  未預期例外
    """
    json_path = os.path.abspath(json_file)
    matrix_path = os.path.abspath("capacitance_matrix_results.json")
    gds_file = os.path.abspath("quantum_chip_final.gds")

    # 即使使用者手動重跑同一資料夾，也不允許沿用舊矩陣。
    if os.path.exists(matrix_path):
        os.remove(matrix_path)

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            params = json.load(file)
        print(f"✅ 成功讀取參數庫：{json_path}")
    except Exception as exc:
        print(f"❌ 無法讀取 JSON 檔案：{exc}", file=sys.stderr)
        return 1

    if not os.path.isfile(gds_file) or os.path.getsize(gds_file) == 0:
        print(
            f"❌ 找不到有效 GDS 檔案：{gds_file}，"
            "請先執行 gds_generator.py。",
            file=sys.stderr,
        )
        return 1

    project_params = params.get("project_settings", {})
    project_name = project_params.get("project_name", "Quantum_Chip_Auto")
    design_name = project_params.get("design_name", "Q3D_Extraction")
    setup_name = project_params.get("setup_name", "Setup_Q3D")
    aedt_version = project_params.get("aedt_version", "2021.2")
    non_graphical = bool(project_params.get("non_graphical", False))

    frequency_ghz = params.get("simulation", {}).get("adaptive_freq_ghz", 5.0)

    qubit_boundary = params.get("Qubit_pra", {})
    thin_conductor_nm = qubit_boundary.get("thin_cond_thickness_nm", 1.0)
    qubit_material = qubit_boundary.get("material_Q", "aluminum")

    substrate_params = params.get("substrate", {})
    substrate_thickness_um = float(substrate_params.get("thickness", 500.0))
    substrate_material = substrate_params.get("material_S", "silicon")
    substrate_name = substrate_params.get("name", "Silicon_Substrate")

    chip_length_um = float(params.get("global", {}).get("gnd_length", 5000.0))
    chip_width_um = float(params.get("global", {}).get("gnd_width", 2000.0))

    layer_mapping_data = params.get("layer_mapping", {})
    if not isinstance(layer_mapping_data, dict) or not layer_mapping_data:
        print("❌ JSON 內沒有有效的 layer_mapping。", file=sys.stderr)
        return 2

    app = None
    result_code = 9

    try:
        print("⏳ 正在啟動 Ansys Q3D Modeler...")
        app = Q3d(
            projectname=project_name,
            designname=design_name,
            specified_version=aedt_version,
            non_graphical=non_graphical,
        )
        app.modeler.model_units = "um"

        print("🧹 正在清除前一次 AEDT 模型與 Setup...")
        try:
            old_objects = (
                app.modeler.object_names
                if hasattr(app.modeler, "object_names")
                else app.modeler.primitives.object_names
            )
            if old_objects:
                app.modeler.delete(old_objects)

            if app.setup_names:
                for old_setup_name in list(app.setup_names):
                    app.delete_setup(old_setup_name)
        except Exception as exc:
            print(f"⚠️ 清除舊 AEDT 內容時發生警告：{exc}")

        print(f"⏳ 正在導入多圖層版圖：{gds_file}")
        layer_map_info = ["NAME:LayerMap"]
        order_entry = []
        sorted_layers = sorted(layer_mapping_data.items(), key=lambda item: item[1])

        for index, (layer_name, gds_layer_number) in enumerate(sorted_layers):
            q3d_layer_name = f"{index}_{layer_name}"
            layer_map_info.append(
                [
                    "NAME:LayerMapInfo",
                    "LayerNum:=",
                    gds_layer_number,
                    "DestLayer:=",
                    q3d_layer_name,
                    "layer_type:=",
                    "signal",
                ]
            )
            order_entry.extend(
                ["entry:=", ["order:=", index, "layer:=", q3d_layer_name]]
            )

        import_options = [
            "NAME:options",
            "FileName:=",
            gds_file,
            "FlattenHierarchy:=",
            True,
            "ImportMethod:=",
            1,
            layer_map_info,
            "OrderMap:=",
            order_entry,
        ]
        app.modeler.oeditor.ImportGDSII(import_options)
        print("✨ 所有訊號層與 GND 已成功匯入。")

        x_start = -(chip_length_um / 2.0)
        y_start = -(chip_width_um / 2.0)
        app.modeler.create_box(
            position=[f"{x_start}um", f"{y_start}um", "0um"],
            dimensions_list=[
                f"{chip_length_um}um",
                f"{chip_width_um}um",
                f"-{substrate_thickness_um}um",
            ],
            name=substrate_name,
            matname=substrate_material,
        )
        print(
            f"✨ 已建立 {chip_length_um} x {chip_width_um} x "
            f"{substrate_thickness_um} um 基板 [{substrate_name}]。"
        )

        all_objects = (
            app.modeler.object_names
            if hasattr(app.modeler, "object_names")
            else app.modeler.primitives.object_names
        )
        signal_objects = [obj for obj in all_objects if obj != substrate_name]
        if not signal_objects:
            print("❌ GDS 匯入後沒有可用的訊號導體。", file=sys.stderr)
            return 3

        app.oboundary.AssignThinConductor(
            [
                "NAME:Superconductor_Films",
                "Objects:=",
                signal_objects,
                "Material:=",
                qubit_material,
                "Thickness:=",
                f"{thin_conductor_nm}nm",
            ]
        )
        print("✨ 超導薄膜邊界條件設定完成。")

        app.oboundary.AutoIdentifyNets()
        print("✨ 網路與節點識別完成。")

        print("⏳ 正在建立並執行 Q3D Setup...")
        setup = app.create_setup(setupname=setup_name)
        setup.props["AdaptiveFreq"] = f"{frequency_ghz}GHz"
        setup.props["Cap"]["PerError"] = 1.0
        setup.props["DC"] = False
        setup.props["AC"] = False
        setup.update()

        app.analyze_setup(setup_name)
        print("✅ Q3D 模擬完成，開始提取電容矩陣。")

        all_capacitance_traces = app.get_traces_for_plot(
            get_self_terms=True,
            get_mutual_terms=True,
            category="C",
        )
        if not all_capacitance_traces:
            print("❌ Q3D 沒有取得任何電容 trace。", file=sys.stderr)
            result_code = 4
            return result_code

        solution = app.post.get_solution_data(expressions=all_capacitance_traces)
        if solution is None:
            print("❌ Q3D 無法取得 solution data。", file=sys.stderr)
            result_code = 5
            return result_code

        export_data = {
            "metadata": {
                "project_name": app.project_name,
                "design_name": design_name,
                "setup_name": setup_name,
                "aedt_version": aedt_version,
                "adaptive_frequency_GHz": float(frequency_ghz),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "matrix_type": "Maxwell Capacitance Matrix",
                "matrix_unit": "fF",
            },
            "matrix_data": {},
        }

        print("\n" + "=" * 60)
        print(f"{'電容元素':<38} | {'數值 (fF)':>15}")
        print("-" * 60)

        for trace_name in all_capacitance_traces:
            try:
                trace_values = solution.data_real(trace_name)
                if trace_values is None or len(trace_values) == 0:
                    raise ValueError("trace 沒有數值")

                value_real = float(trace_values[0])
                units_data = getattr(solution, "units_data", {}) or {}
                unit = units_data.get(trace_name, "")

                if unit == "pF":
                    value_real *= 1_000.0
                elif unit == "F":
                    value_real *= 1e15
                elif unit in {"fF", ""}:
                    pass
                else:
                    print(
                        f"⚠️ 未知電容單位 {unit!r}，"
                        f"{trace_name} 將暫按 fF 保存。"
                    )

                print(f"{trace_name:<38} | {value_real:>12.6f} fF")
                export_data["matrix_data"][trace_name] = {
                    "value": value_real,
                    "unit": "fF",
                }
            except Exception as exc:
                print(
                    f"❌ 提取 trace {trace_name} 失敗：{exc}",
                    file=sys.stderr,
                )
                result_code = 7
                return result_code

        print("=" * 60)

        (
            matrix_complete,
            recognized_node_count,
            independent_element_count,
            unmapped_nodes,
        ) = _validate_exported_matrix(export_data)

        export_data["metadata"].update(
            {
                "recognized_node_count": recognized_node_count,
                "independent_element_count": independent_element_count,
                "expected_independent_element_count": 28,
                "matrix_complete": matrix_complete,
                "unmapped_nodes": unmapped_nodes,
            }
        )

        try:
            with open(matrix_path, "w", encoding="utf-8") as file:
                json.dump(export_data, file, indent=4, ensure_ascii=False)
        except Exception as exc:
            print(f"❌ 寫出 Q3D JSON 失敗：{exc}", file=sys.stderr)
            result_code = 8
            return result_code

        if not os.path.isfile(matrix_path) or os.path.getsize(matrix_path) == 0:
            print("❌ Q3D JSON 沒有成功建立。", file=sys.stderr)
            result_code = 8
            return result_code

        if not matrix_complete:
            print(
                "❌ Q3D矩陣不完整："
                f"節點 {recognized_node_count}/7，"
                f"獨立元素 {independent_element_count}/28，"
                f"未知節點={unmapped_nodes}",
                file=sys.stderr,
            )
            result_code = 6
            return result_code

        print(f"\n🎉 完整電容矩陣已輸出至：{matrix_path}")
        result_code = 0
        return result_code

    except Exception as exc:
        print(f"❌ Q3D 執行發生未預期例外：{exc}", file=sys.stderr)
        result_code = 3 if app is None else 9
        return result_code

    finally:
        if app is not None:
            try:
                app.modeler.fit_all()
            except Exception:
                pass

            try:
                # 僅釋放 Python 連線，不強制關閉使用者的 AEDT 視窗。
                app.release_desktop(
                    close_projects=True,
                    close_desktop=False,
                )
            except Exception as exc:
                print(f"⚠️ 釋放 AEDT 連線時發生警告：{exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Q3D 全自動佈局匯入與矩陣萃取腳本"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="layout_parameters.json",
        help="layout parameter JSON 路徑。",
    )
    arguments = parser.parse_args()
    sys.exit(run_q3d_extraction(arguments.input))