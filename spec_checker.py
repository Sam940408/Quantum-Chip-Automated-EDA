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

# ==========================================
# 老師指定的硬性 QCQ 規格
# ==========================================
FREQ_MIN = 4.0       # GHz
FREQ_MAX = 8.0       # GHz
DETUNING_MIN = 0.300 # GHz
G_ENT_MIN = 5.0      # MHz

def run_spec_check():
    layout_json = "layout_parameters.json"
    q3d_json = "capacitance_matrix_results.json"
    spec_json_path = "spec_results.json"

    # 預設輸出資料結構，確保即使提早結束也有資料可以寫入
    spec_data = {
        "wc_at_g_zero_GHz": None,
        "wc_at_ent_GHz": None,
        "g_ent_max_MHz": None,
        "detuning_at_ent_MHz": None,
        "spec_pass": 0
    }
    
    def save_and_exit():
        """統一的存檔與退出出口，確保 db_manager 能接手處理"""
        try:
            with open(spec_json_path, "w", encoding="utf-8") as f:
                json.dump(spec_data, f, indent=4, ensure_ascii=False)
            print(f"💾 規格數據已匯出至: {spec_json_path}")
        except Exception as e:
            print(f"⚠️ 寫入 {spec_json_path} 失敗: {e}")
        # 無論規格是否達標，都正常退出 (0)，讓自動化工作流繼續走到歸檔步驟！
        sys.exit(0)

    # ==========================================
    # 1. 讀取並驗證 Qubit 頻率限制
    # ==========================================
    if not os.path.exists(layout_json):
        raise FileNotFoundError(f"找不到 {layout_json}")
        
    with open(layout_json, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    freqs = params.get("lom_settings", {}).get("frequencies", {})
    w1 = freqs.get("w1")
    w2 = freqs.get("w2")
    
    if w1 is None or w2 is None:
        raise ValueError("❌ layout_parameters.json 必須提供 w1 與 w2")

    spec_pass_flag = 1

    # 如果頻率不在 4~8 GHz，標記為不合格，但不中斷
    if not (FREQ_MIN <= w1 <= FREQ_MAX) or not (FREQ_MIN <= w2 <= FREQ_MAX):
        print(f"⚠️ 警告：Qubit 頻率超出 4–8 GHz 範圍 (w1={w1:.3f}, w2={w2:.3f})")
        spec_pass_flag = 0

    wc_min_limit = FREQ_MIN
    wc_max_limit = min(FREQ_MAX, min(w1, w2) - DETUNING_MIN)

    # 如果連合理的 Coupler 掃描頻帶都沒有，直接存檔紀錄失敗並結束本腳本
    if wc_max_limit <= wc_min_limit:
        print(f"⚠️ 警告：沒有合法的 coupler 頻帶 (需 4 GHz <= wc <= {wc_max_limit:.3f} GHz)")
        spec_data["spec_pass"] = 0
        save_and_exit() 

    # ==========================================
    # 2. 解析 Q3D 矩陣與拓樸防呆驗證
    # ==========================================
    if not os.path.exists(q3d_json):
        raise FileNotFoundError(f"找不到 {q3d_json}")
        
    CM_raw, nodes_raw = parse_q3d_json(q3d_json)
    CM = np.asarray(CM_raw, dtype=float)
    nodes = list(nodes_raw)
    CM_reduced = CM.copy()

    node_labels = [str(n).lower() for n in nodes]
    gnd_candidates = [i for i, n in enumerate(node_labels) if "gnd" in n]
    
    if len(gnd_candidates) > 1:
        raise ValueError(f"❌ 找到多個 GND 節點: {gnd_candidates}")
    
    if len(gnd_candidates) == 1:
        gnd_idx = gnd_candidates[0]
        CM_reduced = np.delete(CM_reduced, gnd_idx, axis=0)
        CM_reduced = np.delete(CM_reduced, gnd_idx, axis=1)
        node_labels.pop(gnd_idx)
    else:
        print("⚠️ 找不到顯式 GND，假設輸入矩陣已經以 reference conductor 降階。")

    id_q1_p = [i for i, n in enumerate(node_labels) if "qubit1" in n]
    id_q2_p = [i for i, n in enumerate(node_labels) if "qubit2" in n]
    id_c_p  = [i for i, n in enumerate(node_labels) if "coupler1" in n]

    groups = {"qubit1": id_q1_p, "qubit2": id_q2_p, "coupler1": id_c_p}
    for name, ids in groups.items():
        if len(ids) != 2:
            raise ValueError(f"❌ {name} 必須剛好找到 2 個 pad，目前找到 {len(ids)} 個: {ids}")

    if CM_reduced.shape != (len(node_labels), len(node_labels)):
        raise ValueError("❌ 電容矩陣尺寸與 nodes 數量不一致")

    # ==========================================
    # 3. 舒爾補數降階與數值穩定性防護
    # ==========================================
    N = len(node_labels)
    M_trans = np.eye(N)
    M_trans[np.ix_(id_q1_p, id_q1_p)] = [[1, -1], [1, 1]]
    M_trans[np.ix_(id_q2_p, id_q2_p)] = [[1, -1], [1, 1]]
    M_trans[np.ix_(id_c_p, id_c_p)] = [[1, -1], [1, 1]]

    M_inv = np.linalg.inv(M_trans)
    C_mode = M_inv.T @ CM_reduced @ M_inv

    id_q = [id_q1_p[0], id_c_p[0], id_q2_p[0]]
    id_f = [i for i in range(N) if i not in id_q]
    id_reorder = id_q + id_f
    C_temp = C_mode[np.ix_(id_reorder, id_reorder)]

    M_q = len(id_q)
    Cqq = C_temp[:M_q, :M_q]
    
    if len(id_f) > 0:
        Cqx = C_temp[:M_q, M_q:]
        Cxq = C_temp[M_q:, :M_q]
        Cxx = C_temp[M_q:, M_q:]
        Ceff = Cqq - Cqx @ np.linalg.solve(Cxx, Cxq)
    else:
        Ceff = Cqq
        
    Ceff = 0.5 * (Ceff + Ceff.T)

    if not np.all(np.isfinite(Ceff)):
        raise ValueError("❌ Ceff 含有 NaN 或 Inf")

    eigvals = np.linalg.eigvalsh(Ceff)
    if np.min(eigvals) <= 0:
        raise ValueError(f"❌ Ceff 不是正定矩陣，最小 eigenvalue={np.min(eigvals):.3e}")

    cond_ceff = np.linalg.cond(Ceff)
    if cond_ceff > 1e12:
        raise ValueError(f"❌ Ceff condition number 過大: {cond_ceff:.3e}")

    C_inv = np.linalg.inv(Ceff)
    Cinv_11, Cinv_cc, Cinv_22 = C_inv[0, 0], C_inv[1, 1], C_inv[2, 2]
    Cinv_1c, Cinv_12, Cinv_2c = C_inv[0, 1], C_inv[0, 2], C_inv[2, 1]

    # ==========================================
    # 4. 核心物理計算
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

    # ==========================================
    # 5. 尋找關閉點 (g=0) 與檢驗 On-state
    # ==========================================
    wc_array = np.linspace(wc_min_limit, wc_max_limit, 2000)
    gnet_array = np.array([calc_gnet(wc) for wc in wc_array])

    if not np.all(np.isfinite(gnet_array)):
        raise ValueError("❌ gnet_array 出現 NaN 或 Inf")

    zero_brackets = np.where(gnet_array[:-1] * gnet_array[1:] <= 0)[0]
    g_zero_roots = []

    for idx in zero_brackets:
        a = wc_array[idx]
        b = wc_array[idx + 1]
        try:
            root = brentq(calc_gnet, a, b)
            if abs(calc_gnet(root)) < 1e-3: 
                g_zero_roots.append(root)
        except ValueError:
            pass

    wc_at_g_zero = None
    if not g_zero_roots:
        print("⚠️ 警告：在 4–8 GHz 且 detuning >= 300 MHz 的範圍內找不到 g=0 的關閉點。")
        spec_pass_flag = 0
    else:
        wc_at_g_zero = g_zero_roots[-1]

    ent_idx = np.argmax(np.abs(gnet_array))
    wc_at_ent = wc_array[ent_idx]
    g_ent_signed = gnet_array[ent_idx]
    g_ent = abs(g_ent_signed)

    if g_ent <= G_ENT_MIN:
        print(f"⚠️ 警告：最大 g_ent 僅 {g_ent:.3f} MHz，未達 {G_ENT_MIN:.1f} MHz 門檻。")
        spec_pass_flag = 0

    # ==========================================
    # 6. 輸出最終報告與儲存 JSON
    # ==========================================
    if spec_pass_flag == 1:
        print("✅ 晶片符合老師指定的 QCQ 規格")
    else:
        print("❌ 晶片未達理想規格，但已紀錄特徵用於模型訓練。")
        
    print(f"   🔹 Qubit frequencies: w1={w1:.3f} GHz, w2={w2:.3f} GHz")
    if wc_at_g_zero:
        print(f"   🔹 關閉點: wc={wc_at_g_zero:.3f} GHz, g={calc_gnet(wc_at_g_zero):.6f} MHz")
    print(f"   🔹 開啟點: wc={wc_at_ent:.3f} GHz, g_ent={g_ent:.3f} MHz")
    print(f"   🔹 開啟點 detuning: {(min(w1, w2) - wc_at_ent) * 1000:.1f} MHz")

    spec_data["wc_at_g_zero_GHz"] = round(float(wc_at_g_zero), 4) if wc_at_g_zero else None
    spec_data["wc_at_ent_GHz"] = round(float(wc_at_ent), 4)
    spec_data["g_ent_max_MHz"] = round(float(g_ent), 4)
    spec_data["detuning_at_ent_MHz"] = round(float((min(w1, w2) - wc_at_ent) * 1000), 2)
    spec_data["spec_pass"] = spec_pass_flag
    
    save_and_exit()

if __name__ == "__main__":
    run_spec_check()