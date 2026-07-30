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


---

### 第二步：在 GitHub 建立新專案
1. 登入您的 [GitHub](https://github.com/) 帳號。
2. 點擊右上角的 **「+」**，選擇 **「New repository」**。
3. 填寫專案名稱（例如：`Quantum-Chip-Automated-EDA`）。
4. 設定為 Public 或 Private。
5. **注意：** 不要勾選 "Add a README" 或 "Add .gitignore"（因為我們剛剛已經在本地建好了）。
6. 點擊 **「Create repository」**。

---


---
更新上傳步驟

git status
git add .
git commit -m "Update: 更新模擬檔案"
git push origin main