import json
import os
import subprocess
import sys
import time

def update_nested_json(file_path, keys_path, new_value):
    """讀取 JSON，更新指定的巢狀鍵值，然後存檔"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 遞迴找到最後一層並更新數值
    current = data
    for key in keys_path[:-1]:
        current = current[key]
    current[keys_path[-1]] = new_value
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def read_g12_from_lom(lom_file="lom_results.json"):
    """從 LOM 結果檔中讀取 g12 的數值"""
    if not os.path.exists(lom_file):
        return None
    with open(lom_file, 'r', encoding='utf-8') as f:
        lom_data = json.load(f)
    try:
        return lom_data["coupling_parameters_MHz"]["g_12"]
    except KeyError:
        return None

def run_sweep():
    print("🌟 啟動 Q3D 自動參數掃描引擎 (尋找負 g12) 🌟\n")
    
    layout_json = "layout_parameters.json"
    lom_json = "lom_results.json"
    
    # =========================================================
    # 🎯 掃描設定區 (你可以在這裡修改想探索的變數)
    # =========================================================
    # 目標變數的路徑 (對應 layout_parameters.json)
    # 建議先測試改變 Coupler 的中心距離，或是改變 Qubit 的間隙
    target_keys = ["qubit", "radius"] 

    # 你想測試的數值列表 (例如：將距離從 18 縮小到 10，看 g12 的變化)
    test_values = [40, 30, 20, 10, ]
    
    # 要執行的腳本順序 (只跑產生版圖、Q3D、LOM，先不畫圖)
    scripts_to_run = [
        "gds_generator.py",
        "q3d_auto_extraction.py",
        "lom_bridge.py"
    ]
    # =========================================================
    
    results = []

    for val in test_values:
        print(f"\n{'='*50}")
        print(f"🔄 測試參數: {' -> '.join(target_keys)} = {val}")
        print(f"{'='*50}")
        
        # 1. 更新 JSON
        update_nested_json(layout_json, target_keys, val)
        
        # 2. 如果前一次的 LOM 檔案還在，先刪除以免讀到舊資料
        if os.path.exists(lom_json):
            os.remove(lom_json)
            
        # 3. 依序執行工作流
        success = True
        for script in scripts_to_run:
            try:
                subprocess.run([sys.executable, script], check=True)
            except subprocess.CalledProcessError:
                print(f"❌ 執行 {script} 時發生錯誤，跳過此參數。")
                success = False
                break
                
        # 4. 讀取並記錄結果
        if success:
            g12_val = read_g12_from_lom(lom_json)
            if g12_val is not None:
                results.append((val, g12_val))
                print(f"👉 獲得結果: 參數 = {val} | g12 = {g12_val:.4f} MHz")
            else:
                print("❌ 無法從 LOM 結果中讀取 g12。")
        
        # 稍微暫停，讓作業系統釋放檔案資源
        time.sleep(2)

    # 5. 印出最終總結報告
    print("\n" + "🎉"*20)
    print("📊 參數掃描總結報告")
    print("🎉"*20)
    print(f"{'變數值':<15} | {'g_12 (MHz)':<15}")
    print("-" * 35)
    for val, g12 in results:
        trend = "📉 (往負向走!)" if g12 < 0.5 else "📈"
        print(f"{val:<15} | {g12:<15.4f} {trend}")
        
    print("\n💡 觀察提示：尋找讓 g_12 下降的趨勢，順著那個方向繼續掃描！")

if __name__ == "__main__":
    run_sweep()