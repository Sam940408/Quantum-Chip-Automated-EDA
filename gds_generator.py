import sys
import os
import json
import argparse
import klayout.db as pya
from geometry import build_chip_from_params 

def read_layout_params(filename):
    if not os.path.exists(filename):
        print(f"找不到參數檔案 '{filename}'，將使用預設值。")
        return {}
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_structures(ly, top_cell, params):
    print("啟動 KLayout 幾何組裝引擎 (智慧分離多層模式)...")
    dbu = ly.dbu
    
    # 自動化圖層定義
    layer_map = params.get("layer_mapping", {})
    layers = {name: ly.layer(num, 0) for name, num in layer_map.items()}
    
    # 從工廠取得所有元件實例
    components = build_chip_from_params(params)
    
    # 建立「總金屬」與「總挖空」的收集容器
    total_metal = pya.Region()
    total_gap = pya.Region()
    
    # 定義 M90 鏡像 (以 X=0 為軸心水平翻轉)
    mirror_transform = pya.Trans(pya.Trans.M90, 0, 0)
    
    # =====================================================================
    # 1. 處理 Qubits (由單一右側實例，鏡像產生完整的左右兩顆 Qubit)
    # =====================================================================
    if "q1" in components:
        # geometry.py 中的 q1 座標在右側 (+X)，因此在版圖上它其實是 Qubit 2
        q2_m_left, q2_m_right = components["q1"].get_metal_regions(dbu)
        q2_gap = components["q1"].get_gap_region(dbu)
        
        # 透過鏡像產生左側的 Qubit 1
        # 注意：右側 Q2 的「靠內側(左半)」，鏡像過去後會變成左側 Q1 的「靠內側(右半)」！
        q1_m_right = q2_m_left.transformed(mirror_transform)
        q1_m_left = q2_m_right.transformed(mirror_transform)
        q1_gap = q2_gap.transformed(mirror_transform)

        # 寫入 Qubit 1 (左側) 圖層
        if "qubit1_right" in layers:
            top_cell.shapes(layers["qubit1_right"]).insert(q1_m_right)
        if "qubit1_left" in layers:
            top_cell.shapes(layers["qubit1_left"]).insert(q1_m_left)
            
        # 寫入 Qubit 2 (右側) 圖層
        if "qubit2_right" in layers:
            top_cell.shapes(layers["qubit2_right"]).insert(q2_m_right)
        if "qubit2_left" in layers:
            top_cell.shapes(layers["qubit2_left"]).insert(q2_m_left)
            
        # 加入全局收集箱 (兩顆都要加進去，GND 才能正確挖空)
        total_metal += (q1_m_right + q1_m_left + q2_m_right + q2_m_left)
        total_gap += (q1_gap + q2_gap)

    # =====================================================================
    # 2. 處理 Coupler (左右鏡像分離圖層)
    # =====================================================================
    if "cp" in components:
        # 正確解包 (Unpack) 回傳的金屬與挖空區域 (右側)
        cp_m_right, cp_g_right = components["cp"].get_regions(dbu)
        
        # 產生鏡像區域 (左半部)
        cp_m_left = cp_m_right.transformed(mirror_transform)
        cp_g_left = cp_g_right.transformed(mirror_transform)

        # 分別寫入專屬圖層
        if "coupler1_right" in layers:
            top_cell.shapes(layers["coupler1_right"]).insert(cp_m_right)
        if "coupler1_left" in layers:
            top_cell.shapes(layers["coupler1_left"]).insert(cp_m_left)
          
        # 加入全局收集箱以計算 GND 鋪銅
        total_metal += (cp_m_right + cp_m_left)
        total_gap += (cp_g_right + cp_g_left)


    if "t_cp" in components:
        # 解包 T-Coupler 金屬與挖空區域 (右側)
        t_cp_m_right, t_cp_g_right = components["t_cp"].get_regions(dbu)
        
        # 產生鏡像區域 (左半部)
        t_cp_m_left = t_cp_m_right.transformed(mirror_transform)
        t_cp_g_left = t_cp_g_right.transformed(mirror_transform)

        # 分別寫入專屬圖層
        if "coupler1_right" in layers:
            top_cell.shapes(layers["coupler1_right"]).insert(t_cp_m_right)
        if "coupler1_left" in layers:
            top_cell.shapes(layers["coupler1_left"]).insert(t_cp_m_left)
          
        # 加入全局收集箱以計算 GND 鋪銅，避免短路
        total_metal += (t_cp_m_right + t_cp_m_left)
        total_gap += (t_cp_g_right + t_cp_g_left)
    # =====================================================================
    # 3. 處理 Ground 鋪地與 Boolean 挖空
    # =====================================================================
    # 徹底合併總收集箱，避免邊界重疊
    total_metal.merge()
    total_gap.merge()
    
    if "GND" in layers and "gnd" in components:
        gnd_base = components["gnd"].get_metal_region(dbu)
        # Ground = 鋪銅面積 - 總挖空面積 - 總金屬面積
        final_gnd = (gnd_base - total_gap - total_metal).merge()
        top_cell.shapes(layers["GND"]).insert(final_gnd)
        
    print("結構生成與多圖層佈局完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成超導量子晶片 GDS 版圖 (獨立圖層與動態開關整合版)")
    parser.add_argument("-i", "--input", default="layout_parameters.json", help="參數 JSON 檔案路徑")
    args = parser.parse_args()

    params = read_layout_params(args.input)
    if not params:
        sys.exit("請先執行 gds_json_.py 產生參數檔")
    
    ly = pya.Layout()
    ly.dbu = 0.001 
    top_cell = ly.create_cell("Quantum_Chip_Top")
    
    build_structures(ly, top_cell, params)
    
    output_gds = "quantum_chip_final.gds"
    output_dxf = "quantum_chip_final.dxf"
    ly.write(output_gds)
    ly.write(output_dxf)
    print(f"檔案已成功生成並存檔至：\n - {os.path.abspath(output_gds)}\n - {os.path.abspath(output_dxf)}")