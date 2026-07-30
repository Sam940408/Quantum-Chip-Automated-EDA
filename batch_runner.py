import json
import random
import subprocess
import time
import os
import sys
import copy
import numpy as np
from scipy.stats import qmc

# 載入你原本的腳本作為模組 (用來生成基礎範本)
try:
    import gds_json_
except ImportError:
    print("❌ 錯誤：找不到 gds_json_.py，請確認它與 batch_runner.py 在同一資料夾。")
    sys.exit(1)

TARGET_JSON_FILE = "layout_parameters.json"

# =========================================================================
# 🎛️ 參數範圍限制設定 (支援無限層級巢狀結構)
# =========================================================================
PARAM_LIMITS = {
    "qubit": {
        "gap_size": {"min": 30.0, "max": 50.0},
        "slit_width": {"min": 50.0, "max": 70.0},  # 極板間距 (Gap)
        "rect_length": {"min": 50.0, "max": 1000.0}, # 配合極板總面積 (Pad Area)
        "rect_width": {"min": 50.0, "max": 1000.0},  # 配合極板總面積 (Pad Area)
        "q_c_dis": {"min": 70.0, "max": 120.0},
    },
    "h_coupler": {
        "arm_length": {"min": 50.0, "max": 500.0},
        "head1_width": {"min": 10.0, "max": 200.0},
        "head2_width": {"min": 10.0, "max": 200.0},
        "head1_length": {"min": 10.0, "max": 200.0},
        "head2_length": {"min": 10.0, "max": 200.0},
        "gap_size": {"min": 30.0, "max": 50.0},
        "center_dis": {"min": 25.0, "max": 35.0},
    }
}

def get_pruned_limits(base_json_data):
    """根據當前的形狀開關，修剪 PARAM_LIMITS，並過濾掉 min >= max 的無效區間"""
    current_limits = copy.deepcopy(PARAM_LIMITS)
    toggles = base_json_data.get("toggles", {})
    q_type = toggles.get("qubit_type", "rect")
    c_type = toggles.get("coupler_type", "arc")

    # 針對 Qubit 形狀修剪
    if "qubit" in current_limits:
        if q_type == "rect":
            current_limits["qubit"].pop("radius", None)
        elif q_type == "circle":
            current_limits["qubit"].pop("rect_length", None)
            current_limits["qubit"].pop("rect_width", None)

    # 針對 Coupler 形狀修剪
    if c_type == "arc":
        current_limits.pop("t_coupler", None)
        current_limits.pop("h_coupler", None)
    elif c_type == "t_shape":
        current_limits.pop("coupler", None)
        current_limits.pop("h_coupler", None)
    elif c_type == "h_shape":
        current_limits.pop("coupler", None)
        current_limits.pop("t_coupler", None)

    # 將巢狀字典平坦化，並加入 min < max 的防呆檢查
    flat_limits = {}
    def flatten(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict) and "min" in v and "max" in v:
                # 🌟 新增防呆：只有當 min 真的小於 max 時，才加入 LHS 抽樣
                if float(v["min"]) < float(v["max"]):
                    flat_limits[prefix + k] = v
                else:
                    print(f"⚠️ 提示：[{prefix + k}] 的 min ({v['min']}) >= max ({v['max']})，已改為固定值，不參與 LHS 抽樣。")
            elif isinstance(v, dict):
                flatten(v, prefix + k + ".")
                
    flatten(current_limits)
    return flat_limits

def generate_lhs_samples(num_samples, flat_limits):
    """生成 LHS 樣本矩陣並縮放至參數上下限"""
    d = len(flat_limits)
    if d == 0:
        return [], []
        
    # 建立拉丁超立方採樣器
    sampler = qmc.LatinHypercube(d=d)
    # 生成 [0, 1] 之間的分佈矩陣
    sample_matrix = sampler.random(n=num_samples) 

    keys = list(flat_limits.keys())
    l_bounds = [flat_limits[k]["min"] for k in keys]
    u_bounds = [flat_limits[k]["max"] for k in keys]

    # 將 [0,1] 縮放到你定義的 [min, max]
    scaled_samples = qmc.scale(sample_matrix, l_bounds, u_bounds)
    return keys, scaled_samples

def apply_single_lhs_sample(target_dict, keys, sample_row, flat_limits):
    """將單列 LHS 數據精準注入到 JSON 巢狀結構中"""
    updated_records = {}
    for i, key_path in enumerate(keys):
        val = sample_row[i]
        
        # 處理資料型態 (整數或浮點數)
        if flat_limits[key_path].get("type") == "int":
            val = int(round(val))
        else:
            val = round(val, 3)

        # 沿著路徑 (如 qubit -> gap_size) 將數值寫入 target_dict
        keys_split = key_path.split('.')
        current = target_dict
        for k in keys_split[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys_split[-1]] = val
        updated_records[key_path] = val

    return updated_records

def run_batch(max_iterations=10000):
    """執行批次自動化迴圈 (LHS 版本)"""
    print(f"🔄 啟動 LHS 拉丁超立方參數空間探索，預計執行 {max_iterations} 次迴圈...")
    print(f"⚠️ 請確認主程式 (main_pipeline.py) 的 'gds_json' 開關已設為 False！\n")
    
    # 1. 產生基礎 JSON 以便獲取當前形狀結構
    gds_json_.generate_layout_json(TARGET_JSON_FILE, print_content=False)
    with open(TARGET_JSON_FILE, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    # 2. 獲取修剪後的有效參數與邊界
    flat_limits = get_pruned_limits(base_config)
    
    # 3. 🎯 核心：一次性生成所有 LHS 樣本
    keys, lhs_matrix = generate_lhs_samples(max_iterations, flat_limits)
    if len(keys) == 0:
        print("❌ 錯誤：找不到任何有效的參數範圍限制。請檢查 PARAM_LIMITS。")
        return
        
    print(f"📐 成功生成 LHS 樣本矩陣！維度: {lhs_matrix.shape} (變數數量: {len(keys)})")
    
    # 4. 開始執行迴圈
    for i in range(max_iterations):
        print(f"\n{'='*80}")
        print(f"▶️ 開始執行第 {i+1}/{max_iterations} 次迭代")
        print(f"{'='*80}")
        
        # 【重要防呆】每次迴圈都重新生成乾淨的預設 JSON，避免上一次的殘留參數干擾
        gds_json_.generate_layout_json(TARGET_JSON_FILE, print_content=False)
        with open(TARGET_JSON_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        # 注入第 i 列的 LHS 數值
        updated_params = apply_single_lhs_sample(config_data, keys, lhs_matrix[i], flat_limits)
        
        print(f"🎲 本次注入 LHS 參數:")
        for k, v in updated_params.items():
            print(f"   - {k}: {v}")

        # 將更新後的內容寫回 JSON 檔案
        with open(TARGET_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        
        # 呼叫主流程 main_pipeline.py
        start_time = time.time()
        try:
            subprocess.run(
                [sys.executable, "main_pipeline.py"], 
                check=True,
                capture_output=False 
            )
            elapsed = time.time() - start_time
            print(f"\n✅ 第 {i+1} 次迭代成功完成！耗時: {elapsed:.2f} 秒")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 第 {i+1} 次迭代被 DRC 或模擬錯誤攔截 (返回碼 {e.returncode})。")
            print("⚠️ 系統將暫停 2 秒後產生下一組全新參數...")
            time.sleep(2)
            continue 

if __name__ == "__main__":
    # 這裡已經幫你設定為 10000 筆了，隨時可以起跑！
    run_batch(max_iterations=2)