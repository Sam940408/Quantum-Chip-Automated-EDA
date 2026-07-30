"""
01_data_preprocessing.py
==========================
資料前處理與對齊程式
--------------------
職責（只做這件事，不碰模型）：
  1. 從 quantum_dataset.db 讀取原始模擬記錄
  2. 資料品質檢查（缺值、重複列、離群值）
  3. 自動偵測固定參數 vs 可變參數（並印出報告）
  4. 對齊欄位順序（確保幾何參數與效能輸出的順序固定不變）
  5. 正規化（mean / std），計算統計量
  6. 切分訓練 / 驗證集索引
  7. 輸出「乾淨資料包」：
       processed_data.npz   ← 數值陣列（給模型讀取）
       processed_meta.json  ← 中繼資料（欄位名稱、正規化統計、邊界）

下一步：
  02_train_model.py 會直接讀取這兩個檔案，不重新碰觸資料庫，
  確保「資料處理」與「模型訓練」完全解耦。

使用方式：
  python 01_data_preprocessing.py
"""

import sqlite3
import json
import hashlib
from datetime import datetime
import os
from pathlib import Path
os.chdir(Path(__file__).parent)
import numpy as np


# ══════════════════════════════════════════════════════════════
# 0. 設定區
# ══════════════════════════════════════════════════════════════

DB_PATH = r"C:\Users\j6149\OneDrive\桌面\量子電腦\quantum_dataset.db"   # ← 改成你的實際路徑
TABLE_NAME = "simulation_logs"

OUTPUT_NPZ_PATH  = "processed_data.npz"
OUTPUT_META_PATH = "processed_meta.json"

# 資料表裡完整的幾何參數欄位（含可能固定的）
RAW_PARAM_COLS = [
    "param_chip_w", "param_chip_h", "param_box_w", "param_box_l",
    "param_finger_l", "param_num_pairs", "param_gap_cq", "param_gap_cross",
    "param_finger_w", "param_spacing", "param_bus_w", "param_gap_tip",
    "param_leg_w", "param_leg_l", "param_tube_w", "param_gap_u_res",
    "param_gap_tube", "param_gap_resonator", "param_u_insert_idx",
    "param_arm_w", "param_arm_l", "param_gap_qr",
]

# 量子效能輸出欄位（順序固定，模型輸出永遠依此順序）
RAW_OUTPUT_COLS = [
    "EC_coupler1", "EC_qubit1", "EC_qubit2",
    "g_g12_q1_q2", "g_g1c_q1_cp", "g_g2c_q2_cp",
]

# 判定「固定參數」的標準差門檻（std < 此值 → 視為固定，不納入模型輸入）
FIXED_STD_THRESHOLD = 1e-5

# 離群值偵測：用 z-score，超過此倍數標準差視為離群樣本
OUTLIER_Z_THRESHOLD = 5.0

# 訓練 / 驗證切分比例
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# ACTION_DIM 需要固定為 13（歷史模型設計），若可變參數不足 13 個，
# 複製 chip_h 補齊第 13 維（維持與既有 SAC / Surrogate 架構相容）
TARGET_ACTION_DIM = 13
PAD_COLUMN_SOURCE = "param_chip_h"   # 補齊時複製這一欄


# ══════════════════════════════════════════════════════════════
# 1. 讀取原始資料
# ══════════════════════════════════════════════════════════════

def load_raw_data(db_path: str, table: str):
    """
    從 sqlite DB 讀取原始資料。
    回傳：
      raw_params  (N, len(RAW_PARAM_COLS))
      raw_outputs (N, len(RAW_OUTPUT_COLS))
      row_ids     (N,)  ─ 用來追蹤資料列來源（若表中有 id 欄則用它，否則用列號）
    """
    print(f"[1/6] 連接資料庫：{db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 確認表格存在
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cur.fetchall()]
    if table not in tables:
        conn.close()
        raise RuntimeError(
            f"找不到資料表 '{table}'，資料庫內現有表格：{tables}\n"
            f"請確認 DB_PATH 指向正確的檔案。"
        )

    # 確認所有欄位都存在，缺欄位要明確報錯（而不是靜默出錯）
    cur.execute(f"PRAGMA table_info({table})")
    existing_cols = {row[1] for row in cur.fetchall()}
    all_needed = RAW_PARAM_COLS + RAW_OUTPUT_COLS
    missing = [c for c in all_needed if c not in existing_cols]
    if missing:
        conn.close()
        raise RuntimeError(f"資料表缺少欄位：{missing}")

    # 讀取
    select_cols = ", ".join(all_needed)
    cur.execute(f"SELECT rowid, {select_cols} FROM {table}")
    rows = cur.fetchall()
    conn.close()

    if len(rows) == 0:
        raise RuntimeError(f"資料表 '{table}' 是空的，沒有任何記錄。")

    arr = np.array(rows, dtype=np.float64)
    row_ids     = arr[:, 0].astype(np.int64)
    raw_params  = arr[:, 1 : 1 + len(RAW_PARAM_COLS)]
    raw_outputs = arr[:, 1 + len(RAW_PARAM_COLS):]

    print(f"      讀到 {len(rows)} 筆原始記錄，"
          f"{raw_params.shape[1]} 個幾何欄位，{raw_outputs.shape[1]} 個效能欄位")
    return raw_params, raw_outputs, row_ids


# ══════════════════════════════════════════════════════════════
# 2. 資料品質檢查與清理
# ══════════════════════════════════════════════════════════════

def clean_and_align(raw_params, raw_outputs, row_ids):
    """
    資料對齊 / 清理，依序執行：
      a) 移除含 NaN 或 Inf 的列
      b) 移除完全重複的列
      c) 用 z-score 偵測並移除離群值列
    每一步都印出移除筆數，方便追蹤資料品質。
    """
    n0 = len(raw_params)
    print(f"\n[2/6] 資料清理（起始 {n0} 筆）")

    # ── a) NaN / Inf 檢查 ──
    finite_mask = np.isfinite(raw_params).all(axis=1) & np.isfinite(raw_outputs).all(axis=1)
    n_bad = (~finite_mask).sum()
    if n_bad > 0:
        print(f"      移除含 NaN/Inf 的列：{n_bad} 筆")
    raw_params, raw_outputs, row_ids = (
        raw_params[finite_mask], raw_outputs[finite_mask], row_ids[finite_mask]
    )

    # ── b) 完全重複列 ──
    combined = np.concatenate([raw_params, raw_outputs], axis=1)
    _, unique_idx = np.unique(combined, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    n_dup = len(combined) - len(unique_idx)
    if n_dup > 0:
        print(f"      移除完全重複的列：{n_dup} 筆")
    raw_params, raw_outputs, row_ids = (
        raw_params[unique_idx], raw_outputs[unique_idx], row_ids[unique_idx]
    )

    # ── c) 離群值（僅針對效能輸出欄位，幾何參數本身邊界由 DB 記錄決定）──
    z = np.abs((raw_outputs - raw_outputs.mean(0)) / (raw_outputs.std(0) + 1e-8))
    outlier_mask = (z > OUTLIER_Z_THRESHOLD).any(axis=1)
    n_out = outlier_mask.sum()
    if n_out > 0:
        print(f"      移除效能離群值列（|z|>{OUTLIER_Z_THRESHOLD}）：{n_out} 筆")
    keep = ~outlier_mask
    raw_params, raw_outputs, row_ids = (
        raw_params[keep], raw_outputs[keep], row_ids[keep]
    )

    n1 = len(raw_params)
    print(f"      清理完成：{n0} → {n1} 筆（移除 {n0 - n1} 筆，"
          f"保留率 {n1/n0*100:.1f}%）")

    return raw_params, raw_outputs, row_ids


# ══════════════════════════════════════════════════════════════
# 3. 偵測固定參數 / 可變參數，並對齊到固定 13 維
# ══════════════════════════════════════════════════════════════

def split_fixed_variable(raw_params):
    """
    依標準差自動判斷每個幾何欄位是「固定」還是「可變」。
    回傳：
      variable_cols   (list[str])  可變欄位名稱（依原始順序）
      variable_data   (N, n_var)   可變欄位數值
      fixed_params    (dict)       固定欄位名稱 → 固定值
    """
    print(f"\n[3/6] 偵測固定 / 可變參數（threshold std < {FIXED_STD_THRESHOLD}）")

    stds  = raw_params.std(axis=0)
    means = raw_params.mean(axis=0)

    variable_cols, variable_idx = [], []
    fixed_params = {}

    for i, name in enumerate(RAW_PARAM_COLS):
        clean_name = name.replace("param_", "")
        if stds[i] < FIXED_STD_THRESHOLD:
            fixed_params[clean_name] = float(means[i])
        else:
            variable_cols.append(clean_name)
            variable_idx.append(i)

    variable_data = raw_params[:, variable_idx]

    print(f"      可變參數（{len(variable_cols)} 個）：{variable_cols}")
    print(f"      固定參數（{len(fixed_params)} 個）：{list(fixed_params.keys())}")

    return variable_cols, variable_data, fixed_params


def pad_to_action_dim(variable_cols, variable_data, target_dim=TARGET_ACTION_DIM):
    """
    模型架構（Surrogate / SAC）固定吃 ACTION_DIM=13 維輸入。
    若實際可變參數不足 13 個，複製指定欄位補齊，
    並在 meta 中明確記錄「哪一維是補齊出來的」，避免後續誤用。
    """
    n_var = len(variable_cols)
    if n_var == target_dim:
        print(f"[4/6] 可變參數剛好 {target_dim} 維，不需補齊")
        return variable_cols, variable_data, None

    if n_var > target_dim:
        raise RuntimeError(
            f"可變參數有 {n_var} 個，超過模型設計的 {target_dim} 維。"
            f"請確認 TARGET_ACTION_DIM 設定，或檢查資料是否有誤。"
        )

    print(f"[4/6] 可變參數只有 {n_var} 維，補齊到 {target_dim} 維")
    if PAD_COLUMN_SOURCE.replace("param_", "") not in variable_cols:
        raise RuntimeError(
            f"補齊來源欄位 '{PAD_COLUMN_SOURCE}' 不在可變參數清單中，"
            f"請檢查 PAD_COLUMN_SOURCE 設定。"
        )

    src_idx = variable_cols.index(PAD_COLUMN_SOURCE.replace("param_", ""))
    n_pad = target_dim - n_var
    pad_data = np.repeat(variable_data[:, src_idx:src_idx+1], n_pad, axis=1)

    padded_cols = variable_cols + [f"{variable_cols[src_idx]}_pad{i}" for i in range(n_pad)]
    padded_data = np.concatenate([variable_data, pad_data], axis=1)

    print(f"      補齊欄位：{padded_cols[n_var:]}（複製自 '{variable_cols[src_idx]}'）")
    return padded_cols, padded_data, {"padded_from": variable_cols[src_idx], "n_pad": n_pad}


# ══════════════════════════════════════════════════════════════
# 4. 正規化
# ══════════════════════════════════════════════════════════════

def normalize(data: np.ndarray):
    """回傳正規化後資料、以及 (mean, std) 供之後反正規化使用"""
    mean = data.mean(axis=0)
    std  = data.std(axis=0) + 1e-8
    return (data - mean) / std, mean, std


# ══════════════════════════════════════════════════════════════
# 5. 訓練 / 驗證切分
# ══════════════════════════════════════════════════════════════

def make_split(n_samples: int, val_split: float, seed: int):
    """回傳 train_idx, val_idx（固定 seed 確保可重現）"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_val = int(n_samples * val_split)
    return idx[n_val:], idx[:n_val]


# ══════════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  資料前處理與對齊  (01_data_preprocessing.py)")
    print("=" * 60)

    # 1. 讀取
    raw_params, raw_outputs, row_ids = load_raw_data(DB_PATH, TABLE_NAME)

    # 2. 清理
    raw_params, raw_outputs, row_ids = clean_and_align(raw_params, raw_outputs, row_ids)

    # 3. 固定 / 可變參數拆分
    var_cols, var_data, fixed_params = split_fixed_variable(raw_params)

    # 4. 補齊到固定維度
    var_cols, var_data, pad_info = pad_to_action_dim(var_cols, var_data)

    # 5. 訓練 / 驗證切分（先切分，正規化統計只用訓練集算，避免資料洩漏）
    n = len(var_data)
    train_idx, val_idx = make_split(n, VAL_SPLIT, RANDOM_SEED)
    print(f"\n[5/6] 切分資料：訓練 {len(train_idx)} 筆 / 驗證 {len(val_idx)} 筆")

    # 6. 正規化 ── mean/std 只從「訓練集」計算，再套用到全部資料
    #    （若用全資料算統計量，驗證集的資訊會洩漏進訓練過程，
    #      使驗證 loss 略微低估、模型評估過度樂觀）
    print(f"[6/6] 正規化（統計量僅取自訓練集）")
    param_mean  = var_data[train_idx].mean(axis=0)
    param_std   = var_data[train_idx].std(axis=0) + 1e-8
    output_mean = raw_outputs[train_idx].mean(axis=0)
    output_std  = raw_outputs[train_idx].std(axis=0) + 1e-8
    params_norm  = (var_data    - param_mean)  / param_std
    outputs_norm = (raw_outputs - output_mean) / output_std
    print(f"      幾何參數 mean 範圍：[{param_mean.min():.4f}, {param_mean.max():.4f}]")
    print(f"      效能輸出 mean 範圍：[{output_mean.min():.4f}, {output_mean.max():.4f}]")

    # 幾何參數邊界（給 SAC 動作空間使用）
    param_low  = var_data.min(axis=0)
    param_high = var_data.max(axis=0)

    # ── 輸出 npz（數值陣列） ──
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        params_raw     = var_data.astype(np.float32),
        outputs_raw    = raw_outputs.astype(np.float32),
        params_norm    = params_norm.astype(np.float32),
        outputs_norm   = outputs_norm.astype(np.float32),
        train_idx      = train_idx,
        val_idx        = val_idx,
        row_ids        = row_ids,
    )

    # 用資料內容算一個 hash，讓下游程式可以驗證資料是否被中途替換
    # 注意：必須用「實際存進 npz 的 float32 版本」計算，
    # 否則 02 讀出來重算時位元組不同、hash 會對不上
    params_f32  = var_data.astype(np.float32)
    outputs_f32 = raw_outputs.astype(np.float32)
    data_hash = hashlib.sha256(params_f32.tobytes() + outputs_f32.tobytes()).hexdigest()[:16]
    meta = {
        "created_at":      datetime.now().isoformat(),
        "source_db":       DB_PATH,
        "n_samples":       int(n),
        "n_train":         int(len(train_idx)),
        "n_val":           int(len(val_idx)),
        "data_hash":       data_hash,

        "param_names":     var_cols,
        "output_names":    RAW_OUTPUT_COLS,

        "param_mean":      param_mean.tolist(),
        "param_std":       param_std.tolist(),
        "output_mean":     output_mean.tolist(),
        "output_std":      output_std.tolist(),

        "param_low":       param_low.tolist(),
        "param_high":      param_high.tolist(),

        "fixed_params":    fixed_params,
        "pad_info":        pad_info,

        "cleaning_config": {
            "fixed_std_threshold": FIXED_STD_THRESHOLD,
            "outlier_z_threshold": OUTLIER_Z_THRESHOLD,
            "val_split":            VAL_SPLIT,
            "random_seed":          RANDOM_SEED,
        },
    }

    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 已輸出：")
    print(f"    {OUTPUT_NPZ_PATH}   （數值資料）")
    print(f"    {OUTPUT_META_PATH}  （中繼資料，data_hash={data_hash}）")
    print(f"\n下一步：執行 python 02_train_model.py 開始訓練")


if __name__ == "__main__":
    main()
