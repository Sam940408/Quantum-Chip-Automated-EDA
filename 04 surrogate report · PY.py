"""
04_surrogate_report.py
==========================
Surrogate 預先驗證報告產生器（配合三支式架構）
------------------------------------------------
用途：
  在送進 Q3D 真實模擬（03）之前，先用 Surrogate 快速評估
  SAC 建議的幾何參數是否合理，並產生 HTML 圖表報告供瀏覽器查看。
 
流程：
  1. 讀取 processed_data.npz / processed_meta.json（單一資料來源）
  2. 讀取 surrogate.pt / sac_quantum.pt
  3. 用 SAC policy 推論最佳幾何參數
  4. 用 Surrogate 預測對應的量子效能，與目標值比較
  5. 同時抽樣驗證集評估 Surrogate 整體可信度
  6. 產生 surrogate_report.html（含圖表）
 
使用方式：
  python 04_surrogate_report.py
  → 用瀏覽器開啟 surrogate_report.html
"""
 
import sys
import os
import json
import datetime
from pathlib import Path
 
import numpy as np
import torch
 
# ── 路徑處理：以本程式所在位置為基準，避免工作目錄問題 ──
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
os.chdir(_HERE)
 
from importlib import import_module
train_module = import_module("02_train_model") if (_HERE / "02_train_model.py").exists() else None
if train_module is None:
    raise RuntimeError("找不到 02_train_model.py，請確認本檔案與它放在同一目錄。")
 
ProcessedData      = train_module.ProcessedData
SurrogateModel     = train_module.SurrogateModel
SACTrainer         = train_module.SACTrainer
SurrogateEnv       = train_module.SurrogateEnv
action_to_param    = train_module.action_to_param
SAC_CONFIG         = train_module.SAC_CONFIG
TARGET_PERFORMANCE = train_module.TARGET_PERFORMANCE
REWARD_WEIGHTS     = train_module.REWARD_WEIGHTS
DATA_NPZ_PATH      = train_module.DATA_NPZ_PATH
META_JSON_PATH     = train_module.META_JSON_PATH
SURROGATE_SAVE_PATH = train_module.SURROGATE_SAVE_PATH
SAC_SAVE_PATH        = train_module.SAC_SAVE_PATH
 
OUTPUT_REPORT_PATH = "surrogate_report.html"
N_VAL_SAMPLE = 500   # 從驗證集抽樣評估 Surrogate 可信度的筆數
 
 
# ══════════════════════════════════════════════════════════════
# 1. 載入資料與模型
# ══════════════════════════════════════════════════════════════
 
print("=" * 55)
print("  Surrogate 預先驗證報告  (04_surrogate_report.py)")
print("=" * 55)
 
print("[1/5] 載入資料包...")
data = ProcessedData(DATA_NPZ_PATH, META_JSON_PATH)
 
print("[2/5] 載入 Surrogate 與 SAC...")
for p in (SURROGATE_SAVE_PATH, SAC_SAVE_PATH):
    if not Path(p).exists():
        raise FileNotFoundError(f"找不到 {p}，請先執行 python 02_train_model.py 完成訓練。")
 
surrogate = SurrogateModel(in_dim=data.action_dim, out_dim=data.output_dim)
surrogate.load_state_dict(torch.load(SURROGATE_SAVE_PATH, map_location="cpu"))
surrogate.eval()
 
x_tar = np.array([TARGET_PERFORMANCE[n] for n in data.output_names], dtype=np.float32)
env = SurrogateEnv(surrogate, data, x_tar, REWARD_WEIGHTS)
 
sac = SACTrainer(data.state_dim, data.action_dim, SAC_CONFIG)
sac.load(SAC_SAVE_PATH)
sac.policy.eval()
 
# ══════════════════════════════════════════════════════════════
# 2. SAC 推論最佳參數 → Surrogate 預測效能
# ══════════════════════════════════════════════════════════════
 
print("[3/5] SAC 推論最佳幾何參數並預測效能...")
state  = env.get_state()
action = sac.select_action(state, deterministic=True)
best_params = action_to_param(action, data.param_low, data.param_high)
 
with torch.no_grad():
    y_norm = torch.tensor(data.norm_param(best_params), dtype=torch.float32).unsqueeze(0)
    pred = data.denorm_output(surrogate(y_norm).numpy()[0])
 
abs_errs = np.abs(pred - x_tar)
rel_errs = abs_errs / (np.abs(x_tar) + 1e-8) * 100
 
# ══════════════════════════════════════════════════════════════
# 3. 驗證集整體可信度評估（只用驗證集，訓練集沒意義）
# ══════════════════════════════════════════════════════════════
 
print("[4/5] 評估 Surrogate 在驗證集上的可信度...")
val_idx = data.val_idx
n_sample = min(N_VAL_SAMPLE, len(val_idx))
rng = np.random.default_rng(0)
pick = rng.choice(val_idx, n_sample, replace=False)
 
X_val = torch.tensor(data.params_norm[pick], dtype=torch.float32)
Y_val = data.outputs_raw[pick]
with torch.no_grad():
    Y_pred = surrogate(X_val).numpy() * data.output_std + data.output_mean
 
val_rel = np.abs(Y_pred - Y_val) / (np.abs(Y_val) + 1e-8) * 100
mean_rel = val_rel.mean(axis=0)
p90_rel  = np.percentile(val_rel, 90, axis=0)
 
# 幾何參數在邊界間的位置
param_positions = (best_params - data.param_low) / (data.param_high - data.param_low + 1e-8)
 
# ══════════════════════════════════════════════════════════════
# 4. 產生 HTML 報告
# ══════════════════════════════════════════════════════════════
 
print("[5/5] 產生 HTML 報告...")
 
report_data = {
    "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "output_names":    data.output_names,
    "param_names":     data.param_names,
    "targets":         x_tar.tolist(),
    "predicted":       pred.tolist(),
    "abs_errors":      abs_errs.tolist(),
    "rel_errors":      rel_errs.tolist(),
    "mean_rel_errors": mean_rel.tolist(),
    "p90_rel_errors":  p90_rel.tolist(),
    "best_params":     best_params.tolist(),
    "param_positions": param_positions.tolist(),
    "param_low":       data.param_low.tolist(),
    "param_high":      data.param_high.tolist(),
    "n_sample":        int(n_sample),
    "data_hash":       data.meta["data_hash"],
}
 
HTML = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Surrogate 預先驗證報告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft JhengHei",sans-serif;background:#f5f4f0;color:#1a1a18;font-size:14px;line-height:1.6}}
  .header{{background:#fff;border-bottom:1px solid #e5e4e0;padding:24px 40px}}
  .header h1{{font-size:20px;font-weight:500;margin-bottom:4px}}
  .header .sub{{color:#73726c;font-size:13px}}
  .main{{max-width:1100px;margin:0 auto;padding:28px 40px;display:flex;flex-direction:column;gap:24px}}
  .section-title{{font-size:13px;font-weight:500;color:#73726c;text-transform:uppercase;letter-spacing:.5px;margin-bottom:12px}}
  .card{{background:#fff;border:0.5px solid #d3d1c7;border-radius:12px;padding:20px 24px}}
  .summary-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
  .metric-card{{background:#f5f4f0;border-radius:8px;padding:14px 16px}}
  .metric-label{{font-size:12px;color:#73726c;margin-bottom:4px}}
  .metric-val{{font-size:22px;font-weight:500}}
  .pass{{color:#0f6e56}}.warn{{color:#854f0b}}.fail{{color:#993c1d}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;padding:8px 12px;border-bottom:1px solid #e5e4e0;font-weight:500;color:#73726c;font-size:12px}}
  td{{padding:9px 12px;border-bottom:0.5px solid #e5e4e0}}
  tr:last-child td{{border-bottom:none}}
  .badge{{display:inline-block;font-size:11px;padding:2px 8px;border-radius:99px;font-weight:500}}
  .b-ok{{background:#e1f5ee;color:#085041}}
  .b-warn{{background:#faeeda;color:#633806}}
  .b-fail{{background:#faece7;color:#4a1b0c}}
  .bar-wrap{{background:#e5e4e0;border-radius:4px;height:6px;width:100%;overflow:hidden}}
  .bar-fill{{height:100%;border-radius:4px}}
  .chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
  .chart-box{{position:relative;height:260px}}
  .param-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px 24px}}
  .param-row{{display:flex;flex-direction:column;gap:4px;padding:8px 0;border-bottom:0.5px solid #e5e4e0}}
  .param-name{{font-size:12px;color:#73726c}}
  .param-val{{font-size:13px;font-weight:500}}
  .prog-wrap{{display:flex;align-items:center;gap:8px}}
  .prog-bar{{flex:1;background:#e5e4e0;border-radius:4px;height:5px;overflow:hidden}}
  .prog-fill{{height:100%;background:#534AB7;border-radius:4px}}
  .prog-pct{{font-size:11px;color:#888780;min-width:36px;text-align:right}}
  .footer-note{{font-size:12px;color:#888780;padding:0 4px}}
  @media(max-width:700px){{.chart-grid,.param-grid,.summary-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="header">
  <h1>Surrogate 預先驗證報告</h1>
  <div class="sub">產生時間：{report_data["timestamp"]} ・ 驗證集抽樣：{n_sample} 筆 ・ 資料指紋：{report_data["data_hash"]}</div>
</div>
<div class="main">
 
<div>
  <div class="section-title">最佳參數預測摘要</div>
  <div class="card"><div class="summary-grid" id="summary-grid"></div></div>
</div>
 
<div>
  <div class="section-title">效能指標詳細對照（SAC 最佳參數 → Surrogate 預測）</div>
  <div class="card">
    <table>
      <thead><tr>
        <th>效能指標</th><th>目標值 (GHz)</th><th>預測值 (GHz)</th>
        <th>絕對誤差</th><th>相對誤差</th><th>狀態</th><th>誤差條</th>
      </tr></thead>
      <tbody id="perf-table"></tbody>
    </table>
  </div>
</div>
 
<div class="chart-grid">
  <div>
    <div class="section-title">預測值 vs 目標值</div>
    <div class="card"><div class="chart-box"><canvas id="chart-bar"></canvas></div></div>
  </div>
  <div>
    <div class="section-title">Surrogate 可信度（驗證集 {n_sample} 筆）</div>
    <div class="card"><div class="chart-box"><canvas id="chart-err"></canvas></div></div>
  </div>
</div>
 
<div>
  <div class="section-title">幾何參數位置（左端 = 訓練資料下界，右端 = 上界）</div>
  <div class="card"><div class="param-grid" id="param-grid"></div></div>
</div>
 
<div class="footer-note">
  判讀提示：右上圖是 Surrogate 本身的預測可信度——如果某個指標的驗證集誤差就很大（例如 &gt;10%），
  代表左表中該指標的預測值僅供參考，需以 Q3D 真實模擬（03_q3d_verification.py）為準。
</div>
 
</div>
 
<script>
const D = {json.dumps(report_data, ensure_ascii=False)};
 
// ── 摘要卡片 ──
const nPass = D.rel_errors.filter(e=>e<5).length;
const avgErr = D.rel_errors.reduce((a,b)=>a+b,0)/D.rel_errors.length;
const worstIdx = D.rel_errors.indexOf(Math.max(...D.rel_errors));
const summaries = [
  {{label:"通過指標 (<5%)", val:nPass+"/"+D.output_names.length, cls:nPass===D.output_names.length?"pass":"warn"}},
  {{label:"平均相對誤差", val:avgErr.toFixed(1)+"%", cls:avgErr<5?"pass":avgErr<15?"warn":"fail"}},
  {{label:"最大誤差指標", val:D.output_names[worstIdx], cls:""}},
];
const sg = document.getElementById("summary-grid");
summaries.forEach(s=>{{
  sg.innerHTML += `<div class="metric-card"><div class="metric-label">${{s.label}}</div><div class="metric-val ${{s.cls}}">${{s.val}}</div></div>`;
}});
 
// ── 效能表格 ──
const tb = document.getElementById("perf-table");
D.output_names.forEach((name,i)=>{{
  const re = D.rel_errors[i];
  const badge = re<5?'<span class="badge b-ok">✓ 通過</span>':re<15?'<span class="badge b-warn">△ 偏差</span>':'<span class="badge b-fail">✗ 超標</span>';
  const barColor = re<5?"#1D9E75":re<15?"#EF9F27":"#D85A30";
  tb.innerHTML += `<tr>
    <td style="font-weight:500">${{name}}</td>
    <td>${{D.targets[i].toFixed(3)}}</td>
    <td>${{D.predicted[i].toFixed(3)}}</td>
    <td>${{D.abs_errors[i].toFixed(3)}}</td>
    <td>${{re.toFixed(1)}}%</td>
    <td>${{badge}}</td>
    <td style="min-width:80px"><div class="bar-wrap"><div class="bar-fill" style="width:${{Math.min(re,50)*2}}%;background:${{barColor}}"></div></div></td>
  </tr>`;
}});
 
// ── 圖表 ──
const shortNames = D.output_names.map(n=>n.replace("EC_","EC ").replace("g_g","g ").replace(/_q(\\d)/g,"·q$1").replace("_cp","·cp"));
 
new Chart(document.getElementById("chart-bar"),{{
  type:"bar",
  data:{{labels:shortNames,datasets:[
    {{label:"目標值",data:D.targets,backgroundColor:"rgba(83,74,183,0.25)",borderColor:"#534AB7",borderWidth:1.5}},
    {{label:"預測值",data:D.predicted,backgroundColor:"rgba(29,158,117,0.25)",borderColor:"#1D9E75",borderWidth:1.5}},
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{position:"top",labels:{{font:{{size:12}},boxWidth:14}}}},tooltip:{{callbacks:{{label:c=>c.dataset.label+": "+c.raw.toFixed(3)+" GHz"}}}}}},
    scales:{{x:{{ticks:{{font:{{size:11}}}}}},y:{{ticks:{{font:{{size:11}}}},title:{{display:true,text:"GHz",font:{{size:11}}}}}}}}}}
}});
 
new Chart(document.getElementById("chart-err"),{{
  type:"bar",
  data:{{labels:shortNames,datasets:[
    {{label:"平均相對誤差 %",data:D.mean_rel_errors,backgroundColor:"rgba(83,74,183,0.3)",borderColor:"#534AB7",borderWidth:1.5}},
    {{label:"P90 誤差 %",data:D.p90_rel_errors,backgroundColor:"rgba(216,90,48,0.25)",borderColor:"#D85A30",borderWidth:1.5}},
  ]}},
  options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{position:"top",labels:{{font:{{size:12}},boxWidth:14}}}},tooltip:{{callbacks:{{label:c=>c.dataset.label+": "+c.raw.toFixed(2)+"%"}}}}}},
    scales:{{x:{{ticks:{{font:{{size:11}}}}}},y:{{ticks:{{font:{{size:11}}}},title:{{display:true,text:"%",font:{{size:11}}}}}}}}}}
}});
 
// ── 幾何參數位置 ──
const pg = document.getElementById("param-grid");
D.param_names.forEach((name,i)=>{{
  const pct = (D.param_positions[i]*100).toFixed(1);
  const isPad = name.includes("_pad");
  pg.innerHTML += `<div class="param-row">
    <div style="display:flex;justify-content:space-between">
      <span class="param-name">${{name}}${{isPad?"（補齊欄位）":""}}</span>
      <span class="param-val">${{D.best_params[i].toFixed(5)}}</span>
    </div>
    <div class="prog-wrap">
      <span style="font-size:11px;color:#888780;min-width:48px">${{D.param_low[i].toFixed(4)}}</span>
      <div class="prog-bar"><div class="prog-fill" style="width:${{pct}}%"></div></div>
      <span style="font-size:11px;color:#888780;min-width:48px;text-align:right">${{D.param_high[i].toFixed(4)}}</span>
      <span class="prog-pct">${{pct}}%</span>
    </div>
  </div>`;
}});
</script>
</body>
</html>"""
 
with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(HTML)
 
# ══════════════════════════════════════════════════════════════
# 5. 終端機摘要
# ══════════════════════════════════════════════════════════════
 
print(f"\n✓ 報告已產生：{Path(OUTPUT_REPORT_PATH).resolve()}")
print("  用瀏覽器開啟即可查看完整圖表\n")
 
print("─" * 60)
print(f"  {'效能指標':<18} {'目標':>8} {'預測':>10} {'相對誤差':>10}  狀態")
print("─" * 60)
for i, name in enumerate(data.output_names):
    re = rel_errs[i]
    st = "✓" if re < 5 else ("△" if re < 15 else "✗")
    print(f"  {name:<18} {x_tar[i]:>8.3f} {pred[i]:>10.3f} {re:>8.1f}%   {st}")
print("─" * 60)
n_pass = sum(1 for e in rel_errs if e < 5)
print(f"  通過：{n_pass}/{len(data.output_names)}  |  平均誤差：{rel_errs.mean():.1f}%")
if n_pass == len(data.output_names):
    print("\n✓ 預評估全部通過，可以執行 python 03_q3d_verification.py 做真實模擬")
else:
    print("\n△ 部分指標偏差較大，建議：")
    print("  1. 檢查報告右上圖：若 Surrogate 本身在該指標誤差就大，預測僅供參考")
    print("  2. 增加 02 的 total_steps 重新訓練，或直接用 03 以 Q3D 確認真實誤差")