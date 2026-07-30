import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 直接引入你寫好的 Q3D 解析工具，確保資料源完全一致
from lom_bridge import parse_q3d_json

def generate_g_vs_wc_plot(layout_json="layout_parameters.json", q3d_json="capacitance_matrix_results.json"):
    # 1. 讀取系統基礎頻率參數與掃描參數
    if not os.path.exists(layout_json):
        print(f"❌ 找不到 {layout_json}，請確認參數檔存在。")
        return

    with open(layout_json, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    lom_settings = params.get("lom_settings", {})
    freqs = lom_settings.get("frequencies", {})
    w1 = freqs.get("w1", 4.58)
    w2 = freqs.get("w2", 4.64)

    # 2. 解析 Q3D 萃取的電容矩陣
    if not os.path.exists(q3d_json):
        print(f"❌ 找不到 {q3d_json}，請確認 Q3D 萃取腳本已成功執行。")
        return
        
    CM, nodes = parse_q3d_json(q3d_json)
    N = len(nodes)

    # 3. LOM 矩陣降階 (舒爾補數) - 與 lom_bridge 邏輯完全同步
    CM_reduced = np.delete(CM, -1, axis=0)
    CM_reduced = np.delete(CM_reduced, -1, axis=1)

    id_q1_p = [i for i, n in enumerate(nodes) if "qubit1" in n]
    id_q2_p = [i for i, n in enumerate(nodes) if "qubit2" in n]
    id_c_p  = [i for i, n in enumerate(nodes) if "coupler1" in n]

    M_trans = np.eye(N-1)
    if len(id_q1_p) == 2: M_trans[np.ix_(id_q1_p, id_q1_p)] = [[1, -1], [1, 1]]
    if len(id_q2_p) == 2: M_trans[np.ix_(id_q2_p, id_q2_p)] = [[1, -1], [1, 1]]
    if len(id_c_p) == 2:  M_trans[np.ix_(id_c_p, id_c_p)] = [[1, -1], [1, 1]]

    M_inv = np.linalg.inv(M_trans)
    C_mode = M_inv.T @ CM_reduced @ M_inv

    id_q = []
    if id_q1_p: id_q.append(id_q1_p[0])
    if id_c_p:  id_q.append(id_c_p[0])
    if id_q2_p: id_q.append(id_q2_p[0])
    
    id_f = [i for i in range(N-1) if i not in id_q]
    id_reorder = id_q + id_f
    C_temp = C_mode[np.ix_(id_reorder, id_reorder)]

    M_q = len(id_q)
    Cqq = C_temp[:M_q, :M_q]
    Cqx = C_temp[:M_q, M_q:]
    Cxq = C_temp[M_q:, :M_q]
    Cxx = C_temp[M_q:, M_q:]

    if len(id_f) > 0:
        Ceff = Cqq - Cqx @ np.linalg.inv(Cxx) @ Cxq
    else:
        Ceff = Cqq
        
    C_inv = np.linalg.inv(Ceff)

    # 4. 定義以 wc 為變數的計算閉包 (修改回傳多個參數)
    def calculate_g_sweep(wc_array):
        # 由於 id_q 的順序是 [q1, c, q2] (對應 index 0, 1, 2)
        Cinv_11 = C_inv[0, 0]
        Cinv_cc = C_inv[1, 1]
        Cinv_22 = C_inv[2, 2]
        
        Cinv_1c = C_inv[0, 1]
        Cinv_12 = C_inv[0, 2]
        Cinv_2c = C_inv[2, 1]

        # 初始化紀錄列表
        results = {'geff': [], 'g12': [], 'g1c': [], 'g2c': []}
        
        for wc in wc_array:
            # 依據靜電能公式計算各節點的耦合參數
            g12 = 0.5 * (Cinv_12 / np.sqrt(Cinv_11 * Cinv_22)) * np.sqrt(w1 * w2)
            g1c = 0.5 * (Cinv_1c / np.sqrt(Cinv_11 * Cinv_cc)) * np.sqrt(w1 * wc)
            g2c = 0.5 * (Cinv_2c / np.sqrt(Cinv_22 * Cinv_cc)) * np.sqrt(w2 * wc)

            # 計算間接耦合與總有效耦合
            # 防呆：如果 wc 正好與 qubit 頻率共振，分母不為零
            detuning_term = 0.0
            for w_q in [w1, w2]:
                if np.abs(wc - w_q) > 1e-6:
                    detuning_term += 1 / (wc - w_q) + 1 / (wc + w_q)
                else:
                    detuning_term += 1e6 # 極限近似發散值
            
            gindir = 0.5 * g1c * g2c * detuning_term
            geff = g12 - gindir
            
            # 存入紀錄
            results['geff'].append(geff)
            results['g12'].append(g12)
            results['g1c'].append(g1c)
            results['g2c'].append(g2c)
            
        # 轉為 numpy array 方便後續計算乘法
        return {k: np.array(v) for k, v in results.items()}

    # 5. 從 JSON 動態生成掃描頻段數據
    sweep_settings = lom_settings.get("sweep_settings", {})
    wc_lower_start = sweep_settings.get("wc_lower_start", 2.77)
    wc_lower_stop = sweep_settings.get("wc_lower_stop", 4.0)
    wc_upper_start = sweep_settings.get("wc_upper_start", 4.8)
    wc_upper_stop = sweep_settings.get("wc_upper_stop", 6.14)
    num_points = sweep_settings.get("num_points", 100)

    wc_list_lower = np.linspace(wc_lower_start, wc_lower_stop, num_points)
    wc_list_upper = np.linspace(wc_upper_start, wc_upper_stop, num_points)

    res_lower = calculate_g_sweep(wc_list_lower)
    res_upper = calculate_g_sweep(wc_list_upper)

    # 6. 繪製並輸出兩張獨立圖表
    
    # ------------------ 第一張圖：獨立耦合強度變化 ------------------
    plt.figure(figsize=(10, 6))
    plt.plot(wc_list_lower, res_lower['g12'] * 1e3, '-g', linewidth=2, label=r'$g_{12}/2\pi$')
    plt.plot(wc_list_lower, res_lower['g1c'] * 1e3, '-b', linewidth=2, label=r'$g_{1c}/2\pi$')
    plt.plot(wc_list_lower, res_lower['g2c'] * 1e3, '-r', linewidth=2, label=r'$g_{2c}/2\pi$')
    
    plt.plot(wc_list_upper, res_upper['g12'] * 1e3, '-g', linewidth=2)
    plt.plot(wc_list_upper, res_upper['g1c'] * 1e3, '-b', linewidth=2)
    plt.plot(wc_list_upper, res_upper['g2c'] * 1e3, '-r', linewidth=2)
    
    plt.xlabel(r'$\omega_{c}/2\pi$ (GHz)', fontsize=16)
    plt.ylabel(r'$g/2\pi$ (MHz)', fontsize=16)
    plt.title(r'Individual Coupling Strengths vs $\omega_{c}$', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gcf().set_facecolor('white')
    plt.tight_layout()
    plt.savefig("g_individual_vs_wc.png", dpi=150)
    
    # ------------------ 第二張圖：有效耦合強度變化 ------------------
    plt.figure(figsize=(10, 6))
    plt.plot(wc_list_lower, res_lower['geff'] * 1e3, '-k', linewidth=2, label='Q3D Extracted $g_{eff}$')
    plt.plot(wc_list_upper, res_upper['geff'] * 1e3, '-k', linewidth=2)
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlabel(r'$\omega_{c}/2\pi$ (GHz)', fontsize=16)
    plt.ylabel(r'$g_{eff}/2\pi$ (MHz)', fontsize=16)
    plt.title(r'Effective Coupling $g_{eff}$ vs $\omega_{c}$', fontsize=18)
    plt.ylim(-10, 10)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gcf().set_facecolor('white')
    plt.tight_layout()
    plt.savefig("geff_vs_wc.png", dpi=150)
    
    print("\n🎉 成功生成兩張獨立的圖表！已存檔為: g_individual_vs_wc.png 與 geff_vs_wc.png")
    plt.show()

if __name__ == "__main__":
    generate_g_vs_wc_plot()