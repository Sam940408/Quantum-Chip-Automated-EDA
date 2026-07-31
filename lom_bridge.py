import json
import os
import numpy as np
import re
from datetime import datetime

def parse_q3d_json(json_path):
    """從 Q3D 輸出的 JSON 解析出電容矩陣與節點名稱"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matrix_data = data["matrix_data"]
    
    nodes = set()
    pattern = re.compile(r"C\(([^,]+),([^)]+)\)")
    
    for key in matrix_data.keys():
        match = pattern.match(key.replace(" ", ""))
        if match:
            nodes.add(match.group(1))
            nodes.add(match.group(2))
            
    # 將無序的 set 轉成 list，並利用前綴數字強制排序
    nodes = sorted(list(nodes))
    
    # 保險起見，再次確認 GND 是否在最後一位，若不是則移到最後
    gnd_node = next((n for n in nodes if "GND" in n), None)
    if gnd_node:
        nodes.remove(gnd_node)
        nodes.append(gnd_node)
        
    N = len(nodes)
    node_to_idx = {name: i for i, name in enumerate(nodes)}
    CM = np.zeros((N, N))
    
    for key, info in matrix_data.items():
        match = pattern.match(key.replace(" ", ""))
        if match:
            n1, n2 = match.group(1), match.group(2)
            i, j = node_to_idx[n1], node_to_idx[n2]
            val_F = info["value"] * 1e-15 
            CM[i, j] = val_F
            CM[j, i] = val_F 
            
    return CM, nodes

def calculate_lom_parameters(CM, nodes, w_vector, h_bar, e):
    """執行 LOM 模型計算 Ec 與 g，並回傳結構化字典供 JSON 輸出"""
    print(f"🔍 偵測到的節點順序: {nodes}")
    N = len(nodes)
    
    CM_reduced = np.delete(CM, -1, axis=0)
    CM_reduced = np.delete(CM_reduced, -1, axis=1)
    
    # ⚠️ 【修復 1】註解掉對角線覆寫。
    # 假設 Q3D 輸出的已經是標準 Maxwell Capacitance Matrix（對角線為正總和，非對角線為負互容）。
    # 如果 Q3D 輸出的是純粹的 SPICE 互電容，才需要把下面兩行解除註解。
    # for n in range(N-1):
    #     CM_reduced[n, n] = -np.sum(CM_reduced[n, :]) + CM_reduced[n, n]

    try:
        id_q1_p = [i for i, n in enumerate(nodes) if "qubit1" in n]
        id_q2_p = [i for i, n in enumerate(nodes) if "qubit2" in n]
        id_c_p  = [i for i, n in enumerate(nodes) if "coupler1" in n]
        
        M_trans = np.eye(N-1)
        if len(id_q1_p) == 2:
            M_trans[np.ix_(id_q1_p, id_q1_p)] = [[1, -1], [1, 1]]  
        if len(id_q2_p) == 2:
            M_trans[np.ix_(id_q2_p, id_q2_p)] = [[1, -1], [1, 1]]  
        if len(id_c_p) == 2:
            M_trans[np.ix_(id_c_p, id_c_p)] = [[1, -1], [1, 1]]
    
    except Exception as exc:
        print("⚠️ 節點匹配失敗，請確認 Q3D 的命名是否與腳本預期相符。")
        return None

    # 套用投影片標準公式：C_new = (M^-1)^T * C' * M^-1
    M_inv = np.linalg.inv(M_trans)
    C_mode = M_inv.T @ CM_reduced @ M_inv
    
    id_q = []
    if id_q1_p: id_q.append(id_q1_p[-1]) # 取轉換後矩陣中的差模 index（我們定義的第二個是共模，但通常 Python index 第一個是 0，這裡由 ix_ 排列決定，你的程式邏輯中 id_q 取出的是差模位置）
    # 📝 備註：你原先的 if id_q1_p: id_q.append(id_q1_p[-1]) 在你原本的 [[1,1], [-1,1]] 裡是抓共模。
    # 現在 M_trans 換成了 [[1,-1],[1,1]]，差模在 index 0，共模在 index 1。
    # 為了不改動你下方的排序邏輯，我稍微調整一下 id_q 抓取的位置為差模：
    id_q = []
    if id_q1_p: id_q.append(id_q1_p[0])  # 差模的 index
    if id_c_p:  id_q.append(id_c_p[0])   # 差模的 index
    if id_q2_p: id_q.append(id_q2_p[0])  # 差模的 index
    
    id_f = [i for i in range(N-1) if i not in id_q]

    id_reorder = id_q + id_f
    C_temp = C_mode[np.ix_(id_reorder, id_reorder)]
    
    M_q = len(id_q)
    Cqq = C_temp[:M_q, :M_q]
    Cqx = C_temp[:M_q, M_q:]
    Cxq = C_temp[M_q:, :M_q]
    Cxx = C_temp[M_q:, M_q:]
    
    # 完美實作舒爾補數 (Schur Complement)
    if len(id_f) > 0:
        Ceff = Cqq - Cqx @ np.linalg.inv(Cxx) @ Cxq
    else:
        Ceff = Cqq
        
    C_inv = np.linalg.inv(Ceff)

    Cq_vector = np.zeros(M_q)
    EC_vector = np.zeros(M_q)
    g_matrix = np.zeros_like(C_inv)
    
    # 準備打包成 JSON 的資料結構
    output_data = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_frequencies_GHz": {
                "w1": w_vector[0] if M_q > 0 else None,
                "wc": w_vector[1] if M_q > 1 else None,
                "w2": w_vector[2] if M_q > 2 else None
            }
        },
        "qubits_parameters": {},
        "coupling_parameters_MHz": {}
    }

    print("\n" + "="*40)
    print("📊 LOM 量子參數計算結果")
    print("="*40)
    
    names = ["Qubit_1", "Coupler", "Qubit_2"]
    for idx in range(M_q):
        Cq_vector[idx] = 1 / C_inv[idx, idx]
        ec_ghz = (0.5 * e**2 / Cq_vector[idx] / h_bar / 2 / np.pi) / 1e9 # GHz
        name = names[idx] if idx < len(names) else f"Mode_{idx}"
        
        # 基本數據寫入 JSON 結構
        output_data["qubits_parameters"][name] = {
            "C_eff_fF": round(Cq_vector[idx] / 1e-15, 4),
            "Ec_MHz": round(ec_ghz * 1000, 4)
        }
        
        # 智慧抓取各自的目標頻率 (w1 對應 Q1, wc 對應 Coupler, w2 對應 Q2)
        target_fq = w_vector[idx]
        
        if target_fq > 0 and ec_ghz > 0:
            # 核心物理公式統一反推各自所需的 Ej
            ej_ghz = ((target_fq + ec_ghz) ** 2) / (8 * ec_ghz)
            ratio = ej_ghz / ec_ghz
            
            # 將 Ej 數據完整寫入 JSON 紀錄，方便 db_manager 歸檔存入 SQLite
            output_data["qubits_parameters"][name]["Ej_GHz"] = round(ej_ghz, 4)
            output_data["qubits_parameters"][name]["Ej_Ec_ratio"] = round(ratio, 2)
            
            # 🌟 終端機全面對齊輸出 (Coupler 現在也會完整顯示 Ej 與比值)
            print(f"[{name:<9}] C_eff = {Cq_vector[idx]/1e-15:>6.2f} fF | "
                  f"Ec = {ec_ghz*1000:>6.2f} MHz | "
                  f"Ej = {ej_ghz:>6.3f} GHz (Ej/Ec = {ratio:>4.1f})")

    for nn in range(M_q):
        for mm in range(M_q):
            if nn != mm:
                g_matrix[nn, mm] = 0.5 * (C_inv[nn, mm] / np.sqrt(C_inv[nn, nn] * C_inv[mm, mm])) * np.sqrt(w_vector[nn] * w_vector[mm])

    print("-" * 40)
    if M_q == 3:
        g12 = g_matrix[0, 2] * 1000
        g1c = g_matrix[0, 1] * 1000
        g2c = g_matrix[2, 1] * 1000
        
        # 寫入耦合參數
        output_data["coupling_parameters_MHz"] = {
            "g_12": round(g12, 4),
            "g_1c": round(g1c, 4),
            "g_2c": round(g2c, 4)
        }
        
        print(f"g_12 (Q1-Q2)  = {g12:.2f} MHz")
        print(f"g_1c (Q1-C)   = {g1c:.2f} MHz")
        print(f"g_2c (Q2-C)   = {g2c:.2f} MHz")
    print("="*40)
    
    return output_data

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. 讀取主參數檔
    # ---------------------------------------------------------
    layout_json_path = "layout_parameters.json"
    try:
        with open(layout_json_path, 'r', encoding='utf-8') as f:
            layout_data = json.load(f)
            
        lom_settings = layout_data.get("lom_settings", {})
        consts = lom_settings.get("constants", {})
        freqs = lom_settings.get("frequencies", {})
        
        h_bar = consts.get("h_bar", 1.05457180013e-34)
        e = consts.get("e", 1.602176620898e-19)
        
        w1 = freqs.get("w1", 4.58)
        w2 = freqs.get("w2", 4.64)
        wc = freqs.get("wc", 6.14)
        w_vector = [w1, wc, w2]
        
    except FileNotFoundError:
        print(f"❌ 找不到 {layout_json_path}，將使用預設物理常數。")
        h_bar, e = 1.05457180013e-34, 1.602176620898e-19
        w_vector = [4.58, 6.14, 4.64]

    # ---------------------------------------------------------
    # 2. 讀取 Q3D 矩陣並計算 LOM
    # ---------------------------------------------------------
    q3d_json_path = "capacitance_matrix_results.json"
    output_json_path = "lom_results.json"
    if os.path.exists(output_json_path):
        os.remove(output_json_path)
    try:
        CM, nodes = parse_q3d_json(q3d_json_path)
        final_results = calculate_lom_parameters(CM, nodes, w_vector, h_bar, e)
        
        # ---------------------------------------------------------
        # 3. 輸出最終結果為 JSON 檔案
        # ---------------------------------------------------------
        if final_results:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=4, ensure_ascii=False)
            print(f"\n🎉 LOM 量子參數已成功輸出至: ➡️ {output_json_path}")
            
    except FileNotFoundError:
        print(f"❌ 找不到 {q3d_json_path}，請確認 Q3D 萃取腳本已成功執行。")