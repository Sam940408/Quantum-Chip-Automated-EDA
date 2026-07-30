import json
import numpy as np
import os
import sys

try:
    from lom_bridge import parse_q3d_json
except ImportError:
    print("❌ 錯誤：找不到 lom_bridge.py，請確認腳本位置。")
    sys.exit(1)

def run_spec_check():
    layout_json = "layout_parameters.json"
    q3d_json = "capacitance_matrix_results.json"

    # ==========================================
    # 1. 讀取固定的 w1, w2 頻率
    # ==========================================
    if not os.path.exists(layout_json):
        sys.exit(1)
    with open(layout_json, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    freqs = params.get("lom_settings", {}).get("frequencies", {})
    w1 = freqs.get("w1", 4.58)
    w2 = freqs.get("w2", 4.64)

    # 制定開啟狀態 (On-state) 的 wc 上限：必須與 Qubit 保持至少 300 MHz 的失諧
    wc_max_limit = min(w1, w2) - 0.3

    # ==========================================
    # 2. 解析 Q3D 矩陣與舒爾補數降階
    # ==========================================
    if not os.path.exists(q3d_json):
        sys.exit(1)
    CM, nodes = parse_q3d_json(q3d_json)
    N = len(nodes)
    
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
    Cinv_11, Cinv_cc, Cinv_22 = C_inv[0, 0], C_inv[1, 1], C_inv[2, 2]
    Cinv_1c, Cinv_12, Cinv_2c = C_inv[0, 1], C_inv[0, 2], C_inv[2, 1]

    # ==========================================
    # 3. 建立 wc 掃描陣列並計算 geff 曲線
    # ==========================================
    # 我們讓 wc 從 3.0 GHz 掃描到接近 qubit 頻率 (避開共振奇異點)
    wc_array = np.linspace(3.0, min(w1, w2) - 0.05, 2000)
    geff_array = np.zeros_like(wc_array)

    for i, wc in enumerate(wc_array):
        g12 = 0.5 * (Cinv_12 / np.sqrt(Cinv_11 * Cinv_22)) * np.sqrt(w1 * w2)
        g1c = 0.5 * (Cinv_1c / np.sqrt(Cinv_11 * Cinv_cc)) * np.sqrt(w1 * wc)
        g2c = 0.5 * (Cinv_2c / np.sqrt(Cinv_22 * Cinv_cc)) * np.sqrt(w2 * wc)

        detuning_term = 0.0
        for w_q in [w1, w2]:
            detuning_term += 1 / (wc - w_q) + 1 / (wc + w_q)
            
        gindir = 0.5 * g1c * g2c * detuning_term
        geff_array[i] = (g12 - gindir) * 1000  # 轉成 MHz

    # ==========================================
    # 4. 進行物理規格雙重審查
    # ==========================================
    
    # 審查 A：曲線必須穿越 0 (具備關閉能力)
    # 若最大值 > 0 且 最小值 < 0，代表必然經過 0
    has_zero_crossing = np.max(geff_array) > 0 and np.min(geff_array) < 0
    
    if not has_zero_crossing:
        print("🛑 [物理淘汰] 找不到 geff = 0 的關閉點，Coupler 無法徹底關閉。")
        sys.exit(1)

    # 審查 B：在 wc <= wc_max_limit 的區間內，geff 必須能達到 5 MHz 以上
    valid_mask = wc_array <= wc_max_limit
    
    if not np.any(valid_mask):
        print(f"🛑 [物理淘汰] 合法的 wc 範圍過小，無法進行開啟測試。")
        sys.exit(1)
        
    valid_wc = wc_array[valid_mask]
    valid_geff = geff_array[valid_mask]
    
    if np.max(valid_geff) < 5.0:
        print(f"🛑 [物理淘汰] 在合法失諧範圍內 (wc <= {wc_max_limit:.3f} GHz)，最大耦合僅 {np.max(valid_geff):.2f} MHz (需 >= 5 MHz)")
        sys.exit(1)

    # ==========================================
    # 5. 抓出確切數值並放行
    # ==========================================
    # 找尋最接近 0 的點
    zero_idx = np.argmin(np.abs(geff_array))
    wc_at_zero = wc_array[zero_idx]

    # 找尋合法範圍內，大於等於 5 MHz 的點
    target_idx = np.where(valid_geff >= 5.0)[0][-1] # 取滿足條件中 wc 最大(最靠近極限)的點
    wc_at_target = valid_wc[target_idx]
    geff_at_target = valid_geff[target_idx]

    print(f"✅ [黃金參數誕生] 物理審查完美達標！")
    print(f"   🔹 關閉點 (geff ≈ 0 MHz): 位於 wc = {wc_at_zero:.3f} GHz")
    print(f"   🔹 開啟點 (geff = {geff_at_target:.2f} MHz): 位於 wc = {wc_at_target:.3f} GHz (符合失諧限制 <= {wc_max_limit:.3f} GHz)")
    
    sys.exit(0)

if __name__ == "__main__":
    run_spec_check()