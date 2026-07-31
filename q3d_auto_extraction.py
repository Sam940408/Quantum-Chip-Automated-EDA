import os
import sys
import json
import argparse
from datetime import datetime
from pyaedt import Q3d

def run_q3d_extraction(json_file):
    json_path = os.path.abspath(json_file)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        print(f"✅ 成功讀取參數庫: {json_path}")
    except Exception as e:
        print(f"❌ 無法讀取 JSON 檔案: {e}")
        return 1

    # 1. 提取專案與模擬設定
    proj_params = params.get("project_settings", {})
    proj_name = proj_params.get("project_name", "Quantum_Chip_Auto")
    design_name = proj_params.get("design_name", "Q3D_Extraction")
    setup_name = proj_params.get("setup_name", "Setup_Q3D")
    aedt_ver = proj_params.get("aedt_version", "2021.2")
    
    freq_ghz = params.get("simulation", {}).get("adaptive_freq_ghz", 5.0)

    # 2. 提取基板與邊界條件參數
    Qubit_boundary_params = params.get("Qubit_pra", {})
    thin_cond_nm = Qubit_boundary_params.get("thin_cond_thickness_nm", 1.0)
    material_Q = Qubit_boundary_params.get("material_Q", "aluminum")

    sub_params = params.get("substrate", {})
    sub_thick_um = float(sub_params.get("thickness", 500.0))
    material_S = sub_params.get("material_S", "silicon")
    sub_name = sub_params.get("name", "Silicon_Substrate")
    
    # 從 global 取得晶片總長寬 (由接地 GND 決定)
    chip_len_um = float(params.get("global", {}).get("gnd_length", 5000))
    chip_wid_um = float(params.get("global", {}).get("gnd_width", 5000))

    # 3. 檔案路徑
    gds_file = os.path.abspath("quantum_chip_final.gds")
    matrix_path = os.path.abspath("capacitance_matrix_results.json")

    if os.path.exists(matrix_path):
        os.remove(matrix_path)
        
    if not os.path.exists(gds_file):
        print(f"❌ 錯誤：找不到 GDS 檔案 '{gds_file}'，請先執行 gds_generator.py")
        return 1

    # ==========================================
    # 啟動與清理 AEDT
    # ==========================================
    print("⏳ 正在啟動 Ansys Q3D Modeler...")
    app = Q3d(projectname=proj_name, designname=design_name, 
              specified_version=aedt_ver, non_graphical=False)
    
    # 統一使用 um 作為單位，完美對接 KLayout 的輸出！
    app.modeler.model_units = "um"

    print("🧹 正在清除前一次迭代的模型與設定...")
    try:
        # 清除所有舊的 3D 物件與 Setup
        old_objs = app.modeler.object_names if hasattr(app.modeler, 'object_names') else app.modeler.primitives.object_names
        if old_objs:
            app.modeler.delete(old_objs)
        if app.setup_names:
            for s_name in app.setup_names:
                app.delete_setup(s_name)
    except Exception:
        pass

    # ==========================================
    # 匯入 GDS 多圖層版圖
    # ==========================================
    print(f"⏳ 正在導入多圖層版圖: {gds_file}")
    
    layer_mapping_data = params.get("layer_mapping", {})
    if not layer_mapping_data:
        sys.exit("❌ JSON 內沒有找到 layer_mapping 參數！")

    layer_map_info = ["NAME:LayerMap"]
    order_entry = []
    # 🌟 核心修改：將圖層依照 GDS 數字編號 (0, 1, 2, 3...) 進行升冪排序
    sorted_layers = sorted(layer_mapping_data.items(), key=lambda item: item[1])

    for idx, (layer_name, gds_layer_num) in enumerate(sorted_layers):
        # 💡 就在這裡！我們把順序編號 (idx) 轉成字串加在名字前面
        q3d_layer_name = f"{idx}_{layer_name}" 
        
        layer_map_info.append([
            "NAME:LayerMapInfo", "LayerNum:=", gds_layer_num, "DestLayer:=", q3d_layer_name, "layer_type:=", "signal"
        ])
        # 此處的疊層順序 (order) 將嚴格對應你指定的 0~6 排序
        order_entry.extend(["entry:=", ["order:=", idx, "layer:=", q3d_layer_name]])

    import_options = [
        "NAME:options", "FileName:=", gds_file, "FlattenHierarchy:=", True,
        "ImportMethod:=", 1, layer_map_info, "OrderMap:=", order_entry
    ]
    app.modeler.oeditor.ImportGDSII(import_options)
    print("✨ 所有訊號層與 GND 已成功匯入！")

    # ==========================================
    # 建立基板 (Substrate)
    # ==========================================
    # 以原點 (0,0) 為中心，往下長厚度
    x_start = - (chip_len_um / 2.0)
    y_start = - (chip_wid_um / 2.0)

    app.modeler.create_box(
        position=[f"{x_start}um", f"{y_start}um", "0um"],
        dimensions_list=[f"{chip_len_um}um", f"{chip_wid_um}um", f"-{sub_thick_um}um"],
        name=sub_name,
        matname=material_S
    )
    print(f"✨ 尺寸為 {chip_len_um}x{chip_wid_um}x{sub_thick_um} um 的基板[{sub_name}]建立完成。")

    # ==========================================
    # 設定超導邊界與自動識別節點
    # ==========================================
    all_objs = app.modeler.object_names if hasattr(app.modeler, 'object_names') else app.modeler.primitives.object_names
    signal_objects = [obj for obj in all_objs if obj != sub_name]

    app.oboundary.AssignThinConductor([
        "NAME:Superconductor_Films",
        "Objects:=", signal_objects,
        "Material:=", material_Q,
        "Thickness:=", f"{thin_cond_nm}nm"
    ])
    print("✨ 超導薄膜邊界條件設定完成。")

    app.oboundary.AutoIdentifyNets()
    print("✨ 網路與節點識別完成 (AutoIdentifyNets)。")

    # ==========================================
    # 設定 Q3D 模擬與提取電容矩陣
    # ==========================================
    print("⏳ 正在建立並執行 Q3D 分析設置...")
    setup = app.create_setup(setupname=setup_name)
    setup.props["AdaptiveFreq"] = f"{freq_ghz}GHz"
    setup.props["Cap"]["PerError"] = 1.0  
    setup.props["DC"] = False  
    setup.props["AC"] = False  
    setup.update()

    # 執行分析
    app.analyze_setup(setup_name)
    print("✅ 模擬分析完畢！開始提取電容矩陣。")

    # 提取數據
    all_c_traces = app.get_traces_for_plot(get_self_terms=True, get_mutual_terms=True, category="C")
    if all_c_traces:
        solution = app.post.get_solution_data(expressions=all_c_traces)
        if solution:
            export_data = {
                "metadata": {
                    "project_name": app.project_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "matrix_type": "Maxwell Capacitance Matrix (fF)",
                },
                "matrix_data": {}
            }

            print("\n" + "="*50)
            print(f"{'電容元素':<25} | {'數值 (fF)':>15}")
            print("-" * 50)
            for trace in all_c_traces:
                # 取得數值，並強制轉為 fF (femto-farads) 方便觀看
                val_real = solution.data_real(trace)[0]
                unit = solution.units_data.get(trace, "")
                
                # 自動單位轉換邏輯 (AEDT 通常回傳 pF 或 fF)
                if unit == "pF":
                    val_real *= 1000
                elif unit == "F":
                    val_real *= 1e15
                
                print(f"{trace:<25} | {val_real:>12.6f} fF")
                export_data["matrix_data"][trace] = {"value": val_real, "unit": "fF"}
            print("="*50)

            # 寫入 JSON
            with open(matrix_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4, ensure_ascii=False)
            
            print(f"\n🎉 電容矩陣數據已成功打包匯出至:\n➡️ {matrix_path}")

    app.modeler.fit_all()
    # 執行完畢不強制關閉 AEDT，方便你查看網格與結果
    app.release_desktop(close_projects= True, close_desktop=False)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q3D 全自動佈局匯入與矩陣萃取腳本")
    parser.add_argument("-i", "--input", default="layout_parameters.json")
    args = parser.parse_args()
    sys.exit(run_q3d_extraction(args.input))