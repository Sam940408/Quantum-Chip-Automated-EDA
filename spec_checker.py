import json
import numpy as np
import os
import sys
from scipy.optimize import brentq

try:
    from lom_bridge import parse_q3d_json
except ImportError:
    print("❌ 錯誤：找不到 lom_bridge.py，請確認腳本位置。")
    sys.exit(1)

def run_spec_check():
    layout_json = "layout_parameters.json"
    q3d_json = "capacitance_matrix_results.json"

    # ==========================================
    # 1. 讀取頻率與設定嚴格的物理門檻
    # ==========================================
    if not os.path.exists(layout_json):
        sys.exit(1)
    with open(layout_json, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    freqs = params.get("lom_settings", {}).get("frequencies", {})
    w1 = freqs.get("w1", 6.6)
    w2 = freqs.get("w2", 6.7)

    # 非諧性 (Anharmonicity) 絕對值設定 (單位：GHz)
    # 若 JSON 中沒有，則預設採用典型 Transmon 數值 (約 200~300 MHz)
    eta_1 = params.get("lom_settings", {}).get("eta_1", 0.22)
    eta_2 = params.get("lom_settings", {}).get("eta_2", 0.22)
    eta_c = params.get("lom_settings", {}).get("eta_c", 0.25)

    # 嚴格的色散區間門檻 (確保微擾理論與隔離度高度可靠)
    R_DISP_LIMIT = 0.12
    # 開啟狀態 (On-state) 的 wc 上限
    wc_max_limit = min(w1, w2) - 0.05

    # ==========================================
    # 2. 解析 Q3D 矩陣與動態舒爾補數降階
    # ==========================================
    if not os.path.exists(q3d_json):
        sys.exit(1)
    CM, nodes = parse_q3d_json(q3d_json)
    
    gnd_idx = next((i for i, n in enumerate(nodes) if "GND" in n.upper()), -1)
    if gnd_idx == -1:
        print("⚠️ 警告：找不到 GND 節點，假設矩陣已經降階或無接地。")
    else:
        CM_reduced = np.delete(CM, gnd_idx, axis=0)
        CM_reduced = np.delete(CM_reduced, gnd_idx, axis=1)
        nodes.pop(gnd_idx)

    N = len(nodes)
    id_q1_p = [i for i, n in enumerate(nodes) if "qubit1" in n]
    id_q2_p = [i for i, n in enumerate(nodes) if "qubit2" in n]
    id_c_p  = [i for i, n in enumerate(nodes) if "coupler1" in n]

    M_trans = np.eye(N)
    if len(id_q1_p) == 2: M_trans[np.ix_(id_q1_p, id_q1_p)] = [[1, -1], [1, 1]]
    if len(id_q2_p) == 2: M_trans[np.ix_(id_q2_p, id_q2_p)] = [[1, -1], [1, 1]]
    if len(id_c_p) == 2:  M_trans[np.ix_(id_c_p, id_c_p)] = [[1, -1], [1, 1]]

    M_inv = np.linalg.inv(M_trans)
    C_mode = M_inv.T @ CM_reduced @ M_inv

    id_q = []
    if id_q1_p: id_q.append(id_q1_p[0])
    if id_c_p:  id_q.append(id_c_p[0])
    if id_q2_p: id_q.append(id_q2_p[0])
    
    id_f = [i for i in range(N) if i not in id_q]
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
    # 3. 建立核心物理計算函數
    # ==========================================
    def calc_couplings(wc):
        g12 = 0.5 * (Cinv_12 / np.sqrt(Cinv_11 * Cinv_22)) * np.sqrt(w1 * w2)
        g1c = 0.5 * (Cinv_1c / np.sqrt(Cinv_11 * Cinv_cc)) * np.sqrt(w1 * wc)
        g2c = 0.5 * (Cinv_2c / np.sqrt(Cinv_22 * Cinv_cc)) * np.sqrt(w2 * wc)
        return g12, g1c, g2c

    def calc_gnet(wc):
        g12, g1c, g2c = calc_couplings(wc)
        detuning_term = 1/(wc - w1) + 1/(wc + w1) + 1/(wc - w2) + 1/(wc + w2)
        g_virtual = 0.5 * g1c * g2c * detuning_term
        return (g12 - g_virtual) * 1000

    def calc_rdisp(wc):
        _, g1c, g2c = calc_couplings(wc)
        r1 = np.abs(g1c / (wc - w1))
        r2 = np.abs(g2c / (wc - w2))
        return max(r1, r2)

    def calc_zz(wc):
        """根據論文 Eq. 10 & 11 計算殘餘 ZZ 耦合 (單位：MHz)"""
        g12, g1c, g2c = calc_couplings(wc)
        d12 = w1 - w2
        d1 = wc - w1
        d2 = wc - w2
        
        # Eq. 10: 領頭靜態項 (Static ZZ)
        zeta_2 = -(2 * g12**2 * (eta_1 + eta_2)) / ((d12 - eta_1) * (d12 + eta_2))
        
        # Eq. 11: 磁通依賴高階項 (Flux-dependent ZZ)
        term1 = (1/d2) * (1/d12 + 2/(-d12 + eta_1))
        term2 = (1/d1) * (2/(d12 + eta_2) - 1/d12)
        part1 = -(2 * g12 * g1c * g2c) * (term1 + term2)
        
        part2 = -(2 * g1c**2 * g2c**2) / (d1 + d2 + eta_c) * (1/d1 + 1/d2)**2
        part3 = (g1c**2 * g2c**2) / (d1**2) * (2/(d12 + eta_2) - 1/d12 + 1/d2)
        part4 = (g1c**2 * g2c**2) / (d2**2) * (2/(-d12 + eta_1) + 1/d12 + 1/d1)
        
        zeta_3_4 = part1 + part2 + part3 + part4
        
        return (zeta_2 + zeta_3_4) * 1000

    # 檢查對稱式設計的符號先決條件
    g12_static, g1c_static, g2c_static = calc_couplings(wc_max_limit)
    if g12_static * (g1c_static * g2c_static) > 0:
        print("🛑 [物理淘汰] 符號審查失敗：g12 與 g1c*g2c 符號相同，無法在下方形成零點。")
        sys.exit(1)

    # ==========================================
    # 4. 尋找 Dual-Zero 甜蜜點
    # ==========================================
    wc_array = np.linspace(1.0, wc_max_limit, 2000)
    gnet_array = np.array([calc_gnet(w) for w in wc_array])
    rdisp_array = np.array([calc_rdisp(w) for w in wc_array])

    # 尋找 g_net = 0
    signs_g = np.sign(gnet_array)
    sign_changes_g = np.where(signs_g[:-1] != signs_g[1:])[0]
    
    valid_g_zeros = []
    for idx in sign_changes_g:
        try:
            root_wc = brentq(calc_gnet, wc_array[idx], wc_array[idx+1])
            if calc_rdisp(root_wc) <= R_DISP_LIMIT:
                valid_g_zeros.append(root_wc)
        except ValueError:
            pass

    if not valid_g_zeros:
        print(f"🛑 [物理淘汰] 在色散區間內 (r_disp <= {R_DISP_LIMIT})，找不到 g_net = 0。")
        sys.exit(1)
        
    wc_at_g_zero = valid_g_zeros[-1]
    zz_at_g_zero = calc_zz(wc_at_g_zero)

    # 審查 On-state 效能
    valid_mask = rdisp_array <= R_DISP_LIMIT
    valid_wc = wc_array[valid_mask]
    valid_gnet = gnet_array[valid_mask]
    
    if len(valid_gnet) == 0 or np.max(np.abs(valid_gnet)) < 5.0:
        print(f"🛑 [物理淘汰] 色散區間內最大有效耦合小於 5 MHz。")
        sys.exit(1)

    target_idx = np.where(np.abs(valid_gnet) >= 5.0)[0][-1]
    wc_at_target = valid_wc[target_idx]
    gnet_at_target = valid_gnet[target_idx]

    # 進階 ZZ 審查：是否能找到真正的 ZZ=0？
    zz_array = np.array([calc_zz(w) for w in wc_array])
    signs_zz = np.sign(zz_array)
    sign_changes_zz = np.where(signs_zz[:-1] != signs_zz[1:])[0]
    
    wc_at_zz_zero = None
    for idx in sign_changes_zz:
        try:
            root_zz = brentq(calc_zz, wc_array[idx], wc_array[idx+1])
            if calc_rdisp(root_zz) <= R_DISP_LIMIT:
                wc_at_zz_zero = root_zz
                break
        except ValueError:
            pass

    # ==========================================
    # 5. 放行與報告
    # ==========================================
    print(f"✅ [黃金參數誕生] 物理審查完美達標！")
    print(f"   🔹 交換耦合關閉點 (g = 0): 位於 wc = {wc_at_g_zero:.3f} GHz")
    print(f"      -> 該點的殘餘串擾 (ZZ): {zz_at_g_zero:.3f} MHz")
    
    if wc_at_zz_zero:
        print(f"   🌟 完美相位關閉點 (ZZ = 0): 位於 wc = {wc_at_zz_zero:.3f} GHz")
        frequency_mismatch = abs(wc_at_g_zero - wc_at_zz_zero) * 1000
        print(f"      -> Dual-Zero 頻率錯位僅: {frequency_mismatch:.1f} MHz")
    else:
        print(f"   ⚠️ 未在色散極限內找到完全的 ZZ = 0，需依賴 g=0 點運作。")

    print(f"   🔹 強力開啟點 (g = {gnet_at_target:.2f} MHz): 位於 wc = {wc_at_target:.3f} GHz")
    
    sys.exit(0)

if __name__ == "__main__":
    run_spec_check()