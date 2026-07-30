import subprocess
import time
import os
import sys

# =========================================================================
# 🎛️ 全自動化工作流執行開關 (True: 執行, False: 跳過)
# =========================================================================
PIPELINE_SWITCHES = {
    "gds_json": False,          # 1. 生成版圖與物理參數 (產出 layout_parameters.json)
    "drc_check": True,         # 1.5 新增：執行 DRC 與參數限制檢查 (防呆攔截)
    "gds_generator": True,     # 2. 繪製多圖層 GDS 版圖 (產出 quantum_chip_final.gds)
    "q3d_extraction": True,    # 3. Ansys Q3D 靜電模擬與矩陣萃取 (產出 capacitance_matrix_results.json)
    "lom_bridge": True,        # 4. LOM 模型量子參數計算 (產出 lom_results.json)
    "spec_check" :True,
    "db_manager": True,        # 5. 模擬軌跡與結果歸檔 (寫入 quantum_simulation.db)
    "plot_geff": False          # 6. 繪製有效耦合強度曲線 (產出 geff_vs_wc_pipeline.png)

}

# 定義所有可用步驟、對應腳本與功能說明
PIPELINE_CONFIGS = [
    ("gds_json",       "gds_json_.py",           "1. 生成版圖與物理參數"),
    ("drc_check",      "drc_checker.py",         "1.5 執行 DRC 與參數限制檢查"),
    ("gds_generator",  "gds_generator.py",       "2. 繪製多圖層 GDS 版圖"),
    ("q3d_extraction", "q3d_auto_extraction.py", "3. Ansys Q3D 靜電場模擬與矩陣萃取"),
    ("lom_bridge",     "lom_bridge.py",          "4. LOM 模型量子參數計算"),
    ("spec_check",     "spec_checker.py",        "4.5 嚴格物理指標過濾 (Detuning & Geff)"),
    ("db_manager",     "db_manager.py",          "5. 模擬軌跡與結果歸檔"),
    ("plot_geff",      "plot_geff_sweep.py",     "6. 繪製有效耦合強度曲線")
]

def run_step(script_name, description):
    """
    呼叫獨立的 Python 腳本並測量執行時間
    """
    print(f"\n{'='*60}")
    print(f"🚀 啟動階段: {description}")
    print(f"📄 執行腳本: {script_name}")
    print(f"{'='*60}")

    start_time = time.time()
    
    try:
        # 使用當前虛擬環境的 Python 直譯器執行子程序
        subprocess.run([sys.executable, script_name], check=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ 階段完成！耗時: {elapsed_time:.2f} 秒")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 致命錯誤：執行 {script_name} 時發生異常中斷！")
        print(f"錯誤碼 (Return Code): {e.returncode}")
        return False
        
    except FileNotFoundError:
        print(f"\n❌ 錯誤：找不到腳本 {script_name}。")
        return False

def check_dependencies():
    """
    檢查必要的第三方 Python 套件是否已安裝
    """
    required_packages = ["pyaedt", "numpy", "scipy", "matplotlib"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            # 如果關閉了 Q3D 模擬，就不強求安裝 pyaedt
            if package == "pyaedt" and not PIPELINE_SWITCHES.get("q3d_extraction", False):
                continue
            missing_packages.append(package)

    if missing_packages:
        print("❌ 錯誤：缺少必要的 Python 套件。")
        print("請先安裝以下套件後再執行主程式：")
        for pkg in missing_packages:
            print(f" - pip install {pkg}")
        sys.exit(1)

def check_py():
    """
    智慧環境檢查：僅針對「已啟用」的步驟檢查其腳本與配置檔是否存在
    """
    # 基礎靜態配置檔（不論執行哪些步驟，通常都需要作為基礎環境存在）
    required_configs = ["db_config.json", "constraints.json"] # 將 constraints.json 加入必要檢查
    
    # 根據開關動態收集啟用的腳本
    required_scripts = []
    for key, script, desc in PIPELINE_CONFIGS:
        if PIPELINE_SWITCHES.get(key, False):
            required_scripts.append(script)
            
    # 進行檔案存在性檢查
    missing_files = []
    for f in required_configs + required_scripts:
        if not os.path.exists(f):
            missing_files.append(f)
            
    if missing_files:
        print("❌ 錯誤：缺少執行選定流程所需的必要檔案。")
        print("請確認以下檔案是否存在於當前資料夾，或到頂部 PIPELINE_SWITCHES 關閉未使用的步驟：")
        for f in missing_files:
            print(f" - {f}")
        sys.exit(1)

def main():
    print("🌟 啟動超導量子晶片全自動化 EDA 工作流 (自訂步驟版) 🌟")
    print(f"當前工作目錄: {os.getcwd()}\n")
    
    total_start_time = time.time()
    executed_count = 0

    # 依序走訪並篩選要執行的步驟
    for key, script, description in PIPELINE_CONFIGS:
        if PIPELINE_SWITCHES.get(key, False):
            success = run_step(script, description)
            if not success:
                print("\n🛑 流程已中斷！")
                print("請檢查上述錯誤訊息，確認參數設定或軟體授權後再重新啟動主程式。")
                sys.exit(1) # 若任一步驟失敗 (例如 DRC 不通過)，直接終止主程式
            executed_count += 1
        else:
            print(f"⏭️  [跳過] {description} (開關已關閉)")

    total_time = time.time() - total_start_time
    print("\n" + "🎉"*20)
    print(f"🏆 工作流執行結束！共執行了 {executed_count}/{len(PIPELINE_CONFIGS)} 個步驟")
    print(f"⏱️  總耗時: {total_time:.2f} 秒")
    print("🎉"*20 + "\n")

if __name__ == "__main__":   
    check_dependencies()
    check_py()
    main()