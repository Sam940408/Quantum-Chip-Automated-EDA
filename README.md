# Superconducting Quantum Chip EDA & Surrogate Modeling
**浮接式 QCQ (Qubit-Coupler-Qubit) 超導量子晶片全自動化設計與機器學習代理模型環境**

這是一個專為超導量子晶片設計的全自動化 EDA (Electronic Design Automation) 工作流。本專案將幾何版圖生成、靜電場模擬、量子參數萃取與物理規格審查完美串聯，並具備拉丁超立方採樣 (LHS) 能力，為後續強化學習 (如 SAC) 與機器學習代理模型建立強大的數據收集平台。

## ✨ 核心功能 (Features)
* **參數化版圖生成 (KLayout):** 支援圓形/矩形 Qubit 與 T型/H型/圓弧 Coupler 的動態生成。
* **幾何防呆與 DRC 檢查:** 在模擬前自動攔截畸形參數與製程衝突。
* **自動化矩陣萃取 (Ansys Q3D):** 透過 PyAEDT 自動驅動 Q3D 進行 Maxwell 電容矩陣萃取。
* **量子參數橋接 (LOM Bridge):** 自動執行 Schur Complement 降階，精準反推 $E_C, E_J$ 與耦合強度 $g_{12}, g_{1c}, g_{2c}$。
* **物理規格過濾:** 自動計算有效耦合強度 ($g_{\text{eff}}$) 並驗證關閉點 (Zero-crossing) 與失諧限制。
* **智慧歸檔系統:** SQLite 資料庫自動記錄幾何參數、完整 7x7 電容矩陣元素與 LOM 特徵，打造完美 Data Flywheel。

## 🏗️ 系統工作流架構 (System Architecture)
* 參考附件pdf檔案


## 🚀 工作流架構 (Pipeline)
1. `batch_runner.py` / `main_pipeline.py` (中央調度與 LHS 採樣)
2. `drc_checker.py` (幾何防呆)
3. `gds_generator.py` (KLayout GDS 繪製)
4. `q3d_auto_extraction.py` (Ansys Q3D 萃取)
5. `lom_bridge.py` (LOM 模型轉換)
6. `spec_checker.py` (物理指標審查)
7. `db_manager.py` (SQLite 特徵歸檔)

## 🛠️ 依賴套件 (Dependencies)複製貼上在您的終端機

```bash
必須使用Python 3.10
py -m pip install pyaedt==0.5.0
py -m pip install Klayout
& C:/Users/LAB_PC/AppData/Local/Programs/Python/Python310/python.exe -m pip install --upgrade --force-reinstall clr_loader pythonnet
py -m pip install --upgrade pythonnet clr-loader
py -m pip install scipy
winget install --id GitHub.cli --source winget

---
首次使用這台電腦的基礎設定
# 設定您的使用者名稱 (英文)
git config --global user.name "Your Name"

# 設定您的電子信箱 (與 GitHub 帳號相同)
git config --global user.email "your.email@example.com"

---
更新上傳步驟

git status
git add .
git commit -m "Update: 更新模擬檔案"
git push origin main

複製貼上就可上傳

---
刪除檔案
# 1. 從 Git 追蹤名單中移除檔案 (保留本機檔案)
git rm --cached 檔案名稱
# 若要移除整個資料夾，請加 -r，例如：git rm -r --cached 資料夾名稱

# 2. 提交這個「移除」的動作
git commit -m "Fix: 從追蹤清單移除不該上傳的檔案"

# 3. 推送到 GitHub
git push origin main
---
把 GitHub 上的最新進度抓回本機
git pull origin main


