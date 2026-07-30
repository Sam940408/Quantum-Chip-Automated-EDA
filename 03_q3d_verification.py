"""
03_q3d_verification.py
==========================
Q3D 實際模擬驗證程式（配合新的三支式架構）
--------------------------------------------
與舊版 q3d_runner.py 的差異：
  - 不再自己定義 PARAM_LOW / PARAM_HIGH / OUTPUT_NAMES 等常數
  - 改成直接讀取 01_data_preprocessing.py 產生的 processed_meta.json
  - 幾何參數名稱、邊界、目標效能全部「單一來源」，
    資料改了、模型重訓了，這支程式不用跟著手動修改

流程：
  1. 讀 processed_meta.json 取得參數定義（名稱、邊界、輸出名稱）
  2. 讀 sac_quantum.pt，用 SAC policy 推論最佳幾何參數
  3. pyaedt 連接 AEDT 2025，寫入參數、執行模擬
  4. 讀回電容矩陣，萃取 EC / g
  5. 與 TARGET_PERFORMANCE 比較，輸出誤差報告 + JSON

使用方式：
  python 03_q3d_verification.py                  # 從 SAC 讀取最佳參數並跑 Q3D
  python 03_q3d_verification.py --check-only      # 只印參數，不啟動 AEDT
  python 03_q3d_verification.py --manual          # 用 MANUAL_PARAMS 手動驗證
  python 03_q3d_verification.py --headless        # 不開 AEDT GUI
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接複用 02_train_model.py 裡已經寫好的類別，
# 確保「讀資料」「建模型」的邏輯只有一份，不會兩邊改到不同步
from importlib import import_module
_HERE = Path(__file__).parent          # 本程式所在的資料夾
sys.path.insert(0, str(_HERE))         # 確保 import 找得到同目錄的 02
os.chdir(_HERE)                        # 把工作目錄切到程式所在位置
train_module = import_module("02_train_model") if (_HERE / "02_train_model.py").exists() else None

if train_module is None:
    raise RuntimeError(
        "找不到 02_train_model.py，請確認本檔案與它放在同一目錄。"
    )

ProcessedData  = train_module.ProcessedData
SurrogateModel = train_module.SurrogateModel
SACTrainer     = train_module.SACTrainer
SurrogateEnv   = train_module.SurrogateEnv
action_to_param = train_module.action_to_param
state_synthesis = train_module.state_synthesis
SAC_CONFIG      = train_module.SAC_CONFIG
TARGET_PERFORMANCE = train_module.TARGET_PERFORMANCE
REWARD_WEIGHTS      = train_module.REWARD_WEIGHTS
DATA_NPZ_PATH   = train_module.DATA_NPZ_PATH
META_JSON_PATH  = train_module.META_JSON_PATH
SURROGATE_SAVE_PATH = train_module.SURROGATE_SAVE_PATH
SAC_SAVE_PATH        = train_module.SAC_SAVE_PATH


# ══════════════════════════════════════════════════════════════
# 0. 設定區
# ══════════════════════════════════════════════════════════════

# ── AEDT 專案設定 ──
AEDT_PROJECT_PATH = r"C:\Projects\QuantumChip\qubit_design.aedt"
AEDT_DESIGN_NAME  = "Q3D_Design1"
AEDT_SETUP_NAME   = ""          # 留空 = 自動取第一個 setup
AEDT_VERSION      = "2025.1"

# ── 結果輸出 ──
RESULT_JSON = "q3d_verification_result.json"

# ── Python 參數名 → AEDT 設計變數名稱對照 ──
# 左邊會自動用 processed_meta.json 的 param_names 填入（如果名稱剛好一致就不用改）
# 若你的 AEDT 變數名稱不同（例如多了 "param_" 前綴），在這裡手動修正
AEDT_VAR_NAME_OVERRIDE = {
    # "chip_w": "param_chip_w",   # 範例：若 AEDT 變數叫 param_chip_w 才需要填
}

# ── 手動驗證用參數（--manual 模式）：key 必須對應 processed_meta.json 的 param_names ──
MANUAL_PARAMS = {}   # 留空則用 processed_meta.json 裡任一筆資料當範例

# ── 電容矩陣 Net 索引（依你的 Q3D Net 順序調整）──
NET_INDEX = {"qubit1": 0, "qubit2": 1, "coupler": 2}

# ── 物理常數 ──
E_CHARGE = 1.602176634e-19
H_PLANCK = 6.62607015e-34


# ══════════════════════════════════════════════════════════════
# 1. 讀取最佳幾何參數（來自 SAC）
# ══════════════════════════════════════════════════════════════

def load_best_params_from_sac(data) -> np.ndarray:
    """用訓練好的 SAC policy，以 deterministic 模式推論最佳幾何參數"""
    if not Path(SURROGATE_SAVE_PATH).exists() or not Path(SAC_SAVE_PATH).exists():
        raise FileNotFoundError(
            f"找不到 {SURROGATE_SAVE_PATH} 或 {SAC_SAVE_PATH}，"
            f"請先執行 python 02_train_model.py 完成訓練。"
        )

    print(f"[SAC] 載入 {SURROGATE_SAVE_PATH} / {SAC_SAVE_PATH}")
    surrogate = SurrogateModel(in_dim=data.action_dim, out_dim=data.output_dim)
    surrogate.load_state_dict(torch.load(SURROGATE_SAVE_PATH, map_location="cpu"))

    x_tar = np.array([TARGET_PERFORMANCE[n] for n in data.output_names], dtype=np.float32)
    env = SurrogateEnv(surrogate, data, x_tar, REWARD_WEIGHTS)

    sac = SACTrainer(data.state_dim, data.action_dim, SAC_CONFIG)
    sac.load(SAC_SAVE_PATH)
    sac.policy.eval()

    state = env.get_state()
    action = sac.select_action(state, deterministic=True)
    y_act = action_to_param(action, data.param_low, data.param_high)
    print("[SAC] 最佳幾何參數推論完成")
    return y_act


# ══════════════════════════════════════════════════════════════
# 2. 電容矩陣 → 量子參數萃取
# ══════════════════════════════════════════════════════════════

def extract_quantum_params(C: np.ndarray) -> dict:
    """
    EC = e² / (2 × C_self) / h        [GHz]
    g  = e² × C_mutual / (2√(C_i×C_j)) / h  [GHz]
    """
    q1, q2, cp = NET_INDEX["qubit1"], NET_INDEX["qubit2"], NET_INDEX["coupler"]

    def ec_ghz(c_self):
        return (E_CHARGE ** 2) / (2.0 * c_self) / H_PLANCK / 1e9

    def g_ghz(c_mutual, c_i, c_j):
        return (E_CHARGE ** 2) * c_mutual / (2.0 * np.sqrt(c_i * c_j + 1e-60)) / H_PLANCK / 1e9

    C_q1, C_q2, C_cp = abs(C[q1, q1]), abs(C[q2, q2]), abs(C[cp, cp])

    return {
        "EC_coupler1":  ec_ghz(C_cp),
        "EC_qubit1":    ec_ghz(C_q1),
        "EC_qubit2":    ec_ghz(C_q2),
        "g_g12_q1_q2":  g_ghz(abs(C[q1, q2]), C_q1, C_q2),
        "g_g1c_q1_cp":  g_ghz(abs(C[q1, cp]), C_q1, C_cp),
        "g_g2c_q2_cp":  g_ghz(abs(C[q2, cp]), C_q2, C_cp),
    }


# ══════════════════════════════════════════════════════════════
# 3. Q3D 模擬介面
# ══════════════════════════════════════════════════════════════

class Q3DRunner:
    def __init__(self, project_path=AEDT_PROJECT_PATH, design_name=AEDT_DESIGN_NAME,
                 setup_name=AEDT_SETUP_NAME, non_graphical=False):
        self.project_path  = project_path
        self.design_name   = design_name
        self.setup_name    = setup_name
        self.non_graphical = non_graphical
        self.app = None

    def connect(self):
        print(f"\n[Q3D] 連接 AEDT {AEDT_VERSION}...")
        import pyaedt
        self.app = pyaedt.Q3d(
            projectname=self.project_path,
            designname=self.design_name,
            non_graphical=self.non_graphical,
            new_desktop_session=True,
            specified_version=AEDT_VERSION,
        )
        print(f"[Q3D] 專案已開啟：{self.design_name}")
        if not self.setup_name:
            self.setup_name = self.app.setups[0].name
        print(f"[Q3D] 使用 Setup：{self.setup_name}")

    def write_params(self, param_names, param_values):
        """依 processed_meta.json 的 param_names 順序寫入 AEDT 變數"""
        print("\n[Q3D] 寫入幾何參數：")
        written = {}
        for name, val in zip(param_names, param_values):
            aedt_name = AEDT_VAR_NAME_OVERRIDE.get(name, name)
            aedt_val = f"{val:.6f}mm"
            self.app[aedt_name] = aedt_val
            written[aedt_name] = aedt_val
            print(f"  {aedt_name:20s} = {aedt_val}")
        return written

    def run_simulation(self) -> float:
        print(f"\n[Q3D] 開始模擬（Setup: {self.setup_name}）...")
        t0 = time.time()
        self.app.analyze_setup(self.setup_name)
        elapsed = time.time() - t0
        print(f"[Q3D] 模擬完成，耗時 {elapsed:.1f} 秒")
        return elapsed

    def read_capacitance_matrix(self) -> np.ndarray:
        print("\n[Q3D] 讀取電容矩陣...")
        sweep = f"{self.setup_name} : LastAdaptive"
        sol = self.app.post.get_solution_data(expressions=["C"], setup_sweep_name=sweep)
        c_vals = np.array(sol.data_magnitude()) * 1e-12   # pF → F
        n = int(np.sqrt(len(c_vals)))
        C = c_vals.reshape(n, n)
        print(f"[Q3D] 電容矩陣 {n}×{n}（單位：fF）：")
        for i in range(n):
            print("  " + "  ".join(f"{C[i, j]*1e15:8.3f}" for j in range(n)))
        return C

    def close(self):
        if self.app:
            self.app.release_desktop()
            print("\n[Q3D] AEDT 已釋放")


# ══════════════════════════════════════════════════════════════
# 4. 比較與報告
# ══════════════════════════════════════════════════════════════

def compare_results(simulated: dict, target: dict) -> dict:
    print("\n" + "═" * 58)
    print(f"  {'效能指標':<18} {'目標':>10} {'模擬':>10} {'誤差':>8} {'相對':>8}")
    print("─" * 58)

    report = {}
    for name in target:
        tar, sim = target[name], simulated.get(name, float("nan"))
        abs_err = abs(sim - tar)
        rel_err = abs_err / (abs(tar) + 1e-10) * 100
        status = "✓" if rel_err < 5 else ("△" if rel_err < 15 else "✗")
        print(f"  {name:<18} {tar:>9.3f} {sim:>10.3f} {abs_err:>7.3f} {rel_err:>6.1f}%  {status}")
        report[name] = {"target": tar, "simulated": sim, "abs_error": abs_err,
                         "rel_error_pct": rel_err, "pass": rel_err < 5.0}

    n_pass = sum(v["pass"] for v in report.values())
    print("─" * 58)
    print(f"  通過（誤差 <5%）：{n_pass}/{len(report)} 個指標")
    print("═" * 58)
    return report


def save_result(param_names, params, aedt_written, C, simulated, report, elapsed, out_path):
    data = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 2),
        "geometry_params": {n: round(float(v), 6) for n, v in zip(param_names, params)},
        "aedt_written_vars": aedt_written,
        "capacitance_matrix_fF": (C * 1e15).tolist(),
        "quantum_performance": {k: round(v, 4) for k, v in simulated.items()},
        "comparison_report": report,
        "all_pass": all(v["pass"] for v in report.values()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[結果] 已儲存至 {out_path}")


# ══════════════════════════════════════════════════════════════
# 5. Entry Point
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Q3D Verification (aligned with new pipeline)")
    parser.add_argument("--manual",     action="store_true", help="使用 MANUAL_PARAMS 手動驗證")
    parser.add_argument("--check-only", action="store_true", help="只印參數，不啟動 AEDT")
    parser.add_argument("--headless",   action="store_true", help="不開 AEDT GUI")
    parser.add_argument("--out", default=RESULT_JSON, help="結果 JSON 輸出路徑")
    args = parser.parse_args()

    print("=" * 58)
    print("  Q3D 驗證  (03_q3d_verification.py)  ── AEDT 2025")
    print("=" * 58)

    # 1. 載入資料定義（單一來源：processed_meta.json）
    data = ProcessedData(DATA_NPZ_PATH, META_JSON_PATH)

    # 檢查 TARGET_PERFORMANCE key 是否對得上（與 02_train_model.py 用同一份設定，理論上一定對得上）
    missing = [n for n in data.output_names if n not in TARGET_PERFORMANCE]
    if missing:
        raise RuntimeError(f"TARGET_PERFORMANCE 缺少欄位：{missing}")

    # 2. 取得要驗證的幾何參數
    if args.manual:
        if MANUAL_PARAMS:
            params = np.array([MANUAL_PARAMS[n] for n in data.param_names], dtype=np.float32)
        else:
            # 沒填手動參數就用資料集裡第一筆當範例
            params = data.params_raw[0]
            print("[手動模式] MANUAL_PARAMS 未填，改用資料集第 0 筆作為範例")
    else:
        params = load_best_params_from_sac(data)

    # 找出「補齊維度」欄位（如 chip_h_pad0）：這些是模型內部用的重複欄位，
    # AEDT 裡沒有對應變數，寫入時必須排除
    pad_cols = set()
    if data.meta.get("pad_info"):
        pad_cols = {n for n in data.param_names if "_pad" in n}

    print("\n── 幾何參數 ──")
    for i, (n, v) in enumerate(zip(data.param_names, params)):
        in_range = data.param_low[i] <= v <= data.param_high[i]
        flag = "" if in_range else "  ⚠ 超出訓練範圍"
        pad_tag = "  （補齊欄位，不寫入 AEDT）" if n in pad_cols else ""
        print(f"  {n:20s} = {v:.5f}{flag}{pad_tag}")

    if args.check_only:
        print("\n[check-only] 參數檢查完成，不執行模擬")
        return

    # 3. 連接 Q3D 並模擬（排除補齊欄位，只寫入 AEDT 真實存在的變數）
    write_names, write_values = [], []
    for n, v in zip(data.param_names, params):
        if n not in pad_cols:
            write_names.append(n)
            write_values.append(v)

    runner = Q3DRunner(non_graphical=args.headless)
    try:
        runner.connect()
        written = runner.write_params(write_names, write_values)
        elapsed = runner.run_simulation()
        C = runner.read_capacitance_matrix()
    finally:
        runner.close()

    # 4. 萃取
    print("\n[萃取] 從電容矩陣計算 EC 和 g...")
    simulated = extract_quantum_params(C)
    for name, val in simulated.items():
        print(f"  {name:<18} = {val:.4f} GHz")

    # 5. 比對
    report = compare_results(simulated, {n: TARGET_PERFORMANCE[n] for n in data.output_names})

    # 6. 儲存
    save_result(data.param_names, params, written, C, simulated, report, elapsed, args.out)

    n_pass = sum(v["pass"] for v in report.values())
    print(f"\n{'✓ 全部通過！' if n_pass == len(report) else f'△ {len(report) - n_pass} 個指標超出 5% 誤差範圍'}")
    if n_pass < len(report):
        print("建議：把這筆真實 Q3D 結果加回 simulation_logs，重新跑 01/02 兩支程式")


if __name__ == "__main__":
    main()
