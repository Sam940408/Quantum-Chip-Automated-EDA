"""
02_train_model.py
==========================
模型訓練程式（讀取已清理好的資料包，不碰資料庫）
--------------------------------------------------
職責：
  1. 讀取 01_data_preprocessing.py 產生的 processed_data.npz / processed_meta.json
  2. 驗證資料完整性（data_hash 對得上才繼續，避免資料被中途置換）
  3. 監督式預訓練 Surrogate Model（幾何參數 → 量子效能）
  4. 用 Surrogate 當作快速環境，訓練 SAC 強化學習 Agent
  5. 輸出可推論的模型檔（surrogate.pt / sac_quantum.pt）與訓練歷史

前置需求：
  先執行 01_data_preprocessing.py，確認同目錄下已有
    processed_data.npz
    processed_meta.json

使用方式：
  python 02_train_model.py
"""

import json
import time
import os
from pathlib import Path
os.chdir(Path(__file__).parent)
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


# ══════════════════════════════════════════════════════════════
# 0. 設定區 ── 所有可調權重與超參數都在這裡
# ══════════════════════════════════════════════════════════════

# ── 輸入檔案（由 01_data_preprocessing.py 產生）──
DATA_NPZ_PATH  = "processed_data.npz"
META_JSON_PATH = "processed_meta.json"

# ── 0-A. 目標量子效能（GHz）── key 必須對應 meta 裡的 output_names
TARGET_PERFORMANCE = {
    "EC_coupler1":  100.0,
    "EC_qubit1":    190.0,
    "EC_qubit2":    190.0,
    "g_g12_q1_q2":   5.0,
    "g_g1c_q1_cp":  110.0,
    "g_g2c_q2_cp":  110.0,
}

# ── 0-B. Reward 各效能指標權重（數值越大越重要，自動正規化）──
REWARD_WEIGHTS = {
    "EC_coupler1":  1.0,
    "EC_qubit1":    1.0,
    "EC_qubit2":    1.0,
    "g_g12_q1_q2":  2.0,   # 耦合強度較難控制，加權
    "g_g1c_q1_cp":  1.0,
    "g_g2c_q2_cp":  1.0,
}

# ── 0-C. Surrogate 監督式預訓練超參數 ──
PRETRAIN_CONFIG = {
    "epochs":     50,
    "batch_size": 256,
    "lr":         1e-3,
    "weight_decay": 1e-4,
    "early_stop": 8,
}

# ── 0-D. SAC 超參數 ──
SAC_CONFIG = {
    "lr_actor":        3e-4,
    "lr_critic":       3e-4,
    "lr_alpha":        3e-4,
    "gamma":           0.99,
    "tau":             0.005,
    "buffer_size":     100_000,
    "batch_size":      256,
    "warmup_steps":    1000,
    "update_per_step": 1,
    "init_alpha":      0.2,
    "hidden_dims":     (256, 256, 256),
}

# ── 0-E. 訓練規模 ──
TRAIN_CONFIG = {
    "total_steps":     10_000,
    "bc_init_samples": 5000,     # 從資料集抽樣填入 Replay Buffer 的筆數
    "log_every":       500,
}

SURROGATE_SAVE_PATH = "surrogate.pt"
SAC_SAVE_PATH        = "sac_quantum.pt"
TRAIN_HISTORY_PATH   = "train_history.json"


# ══════════════════════════════════════════════════════════════
# 1. 載入資料包並驗證完整性
# ══════════════════════════════════════════════════════════════

class ProcessedData:
    """
    包裝 01_data_preprocessing.py 輸出的資料，
    提供模型訓練所需的所有陣列與中繼資訊。
    """

    def __init__(self, npz_path: str, meta_path: str):
        if not Path(npz_path).exists() or not Path(meta_path).exists():
            raise FileNotFoundError(
                f"找不到 {npz_path} 或 {meta_path}。\n"
                f"請先執行 python 01_data_preprocessing.py 產生資料包。"
            )

        print(f"[Data] 載入資料包：{npz_path} / {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        npz = np.load(npz_path)
        self.params_raw   = npz["params_raw"]     # (N, action_dim)
        self.outputs_raw  = npz["outputs_raw"]    # (N, output_dim)
        self.params_norm  = npz["params_norm"]
        self.outputs_norm = npz["outputs_norm"]
        self.train_idx    = npz["train_idx"]
        self.val_idx      = npz["val_idx"]

        # ── 完整性驗證：重新計算 hash，確認資料沒被中途竄改 ──
        import hashlib
        recomputed = hashlib.sha256(
            self.params_raw.tobytes() + self.outputs_raw.tobytes()
        ).hexdigest()[:16]
        if recomputed != self.meta["data_hash"]:
            raise RuntimeError(
                f"資料完整性檢查失敗！\n"
                f"  meta 記錄的 hash：{self.meta['data_hash']}\n"
                f"  實際資料的 hash：{recomputed}\n"
                f"資料包可能已損毀或被替換，請重新執行 01_data_preprocessing.py"
            )
        print(f"[Data] 完整性驗證通過（hash={recomputed}）")

        # ── 從 meta 取出關鍵維度資訊 ──
        self.param_names  = self.meta["param_names"]
        self.output_names = self.meta["output_names"]
        self.action_dim   = len(self.param_names)
        self.output_dim   = len(self.output_names)
        self.state_dim    = self.output_dim * 2

        self.param_mean  = np.array(self.meta["param_mean"],  dtype=np.float32)
        self.param_std   = np.array(self.meta["param_std"],   dtype=np.float32)
        self.output_mean = np.array(self.meta["output_mean"], dtype=np.float32)
        self.output_std  = np.array(self.meta["output_std"],  dtype=np.float32)
        self.param_low   = np.array(self.meta["param_low"],   dtype=np.float32)
        self.param_high  = np.array(self.meta["param_high"],  dtype=np.float32)

        print(f"[Data] {self.meta['n_samples']} 筆樣本 | "
              f"action_dim={self.action_dim} | output_dim={self.output_dim} | "
              f"train={self.meta['n_train']} / val={self.meta['n_val']}")
        if self.meta.get("fixed_params"):
            print(f"[Data] 已排除固定參數：{list(self.meta['fixed_params'].keys())}")
        if self.meta.get("pad_info"):
            print(f"[Data] 補齊維度資訊：{self.meta['pad_info']}")

    def denorm_output(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.output_std + self.output_mean

    def norm_param(self, p: np.ndarray) -> np.ndarray:
        return (p - self.param_mean) / self.param_std

    def get_loaders(self, batch_size: int):
        """回傳 (train X,Y tensors), (val X,Y tensors)，供監督式訓練使用"""
        X = torch.tensor(self.params_norm,  dtype=torch.float32)
        Y = torch.tensor(self.outputs_norm, dtype=torch.float32)
        train_ds = torch.utils.data.TensorDataset(X[self.train_idx], Y[self.train_idx])
        val_ds   = torch.utils.data.TensorDataset(X[self.val_idx],   Y[self.val_idx])
        return (
            torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        )


# ══════════════════════════════════════════════════════════════
# 2. Surrogate Model（幾何參數 → 量子效能，取代 Q3D 做快速評估）
# ══════════════════════════════════════════════════════════════

class SurrogateModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(256, 256, 256)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def pretrain_surrogate(model: SurrogateModel, data: ProcessedData, cfg: Dict) -> List[Dict]:
    train_loader, val_loader = data.get_loaders(cfg["batch_size"])
    opt       = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    best_val, no_improve, best_state = np.inf, 0, None
    history = []

    print("\n" + "─" * 55)
    print("  監督式預訓練 Surrogate Model")
    print("─" * 55)

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += F.mse_loss(model(xb), yb).item()
        val_loss /= len(val_loader)
        scheduler.step()
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val
        if improved:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if ep % 5 == 0 or ep == 1:
            tag = " ← best" if improved else ""
            print(f"  Epoch {ep:3d}/{cfg['epochs']} | train={train_loss:.5f} | val={val_loss:.5f}{tag}")

        if no_improve >= cfg["early_stop"]:
            print(f"  Early stop at epoch {ep}（val loss {cfg['early_stop']} epoch 未改善）")
            break

    model.load_state_dict(best_state)
    print(f"  預訓練完成，最佳 val loss = {best_val:.5f}")

    # ── 補充：以「物理單位（GHz）」報告每個輸出欄位在驗證集上的誤差 ──
    #    正規化後的 MSE 難以直觀判斷好壞，
    #    這裡反正規化回真實數值，印出每個 EC/g 的平均絕對誤差與相對誤差
    model.eval()
    with torch.no_grad():
        X_val = torch.tensor(data.params_norm[data.val_idx], dtype=torch.float32)
        pred_norm = model(X_val).numpy()
    pred_real = pred_norm * data.output_std + data.output_mean
    true_real = data.outputs_raw[data.val_idx]
    mae = np.abs(pred_real - true_real).mean(axis=0)
    rel = (np.abs(pred_real - true_real) / (np.abs(true_real) + 1e-8)).mean(axis=0) * 100
    print("  驗證集各輸出誤差（物理單位）：")
    for name, m, r_ in zip(data.output_names, mae, rel):
        print(f"    {name:<16} MAE={m:8.4f} GHz | 平均相對誤差={r_:5.2f}%")
    print()
    return history


# ══════════════════════════════════════════════════════════════
# 3. Reward 函式
# ══════════════════════════════════════════════════════════════

def build_reward_fn(output_names: List[str], x_tar: np.ndarray, weights: Dict[str, float]):
    """r = -Σ wᵢ · |x^sim_i - x^tar_i| / |x^tar_i|，wᵢ 依 output_names 順序取出並正規化"""
    w = np.array([weights[n] for n in output_names], dtype=np.float32)
    w = w / w.sum()

    def reward_fn(x_sim: np.ndarray, penalty: float = 0.0) -> float:
        rel_err = np.abs(x_sim - x_tar) / (np.abs(x_tar) + 1e-8)
        return -float((w * rel_err).sum()) + penalty

    return reward_fn


# ══════════════════════════════════════════════════════════════
# 4. 環境（用 Surrogate 快速模擬）
# ══════════════════════════════════════════════════════════════

class SurrogateEnv:
    def __init__(self, surrogate: SurrogateModel, data: ProcessedData,
                 x_tar: np.ndarray, reward_weights: Dict[str, float]):
        self.surrogate = surrogate
        self.data      = data
        self.x_tar     = x_tar
        self.reward_fn = build_reward_fn(data.output_names, x_tar, reward_weights)
        self.x_sim     = x_tar.copy()

    def _simulate(self, y_act: np.ndarray) -> np.ndarray:
        self.surrogate.eval()
        with torch.no_grad():
            y_norm = self.data.norm_param(y_act)
            pred = self.surrogate(
                torch.tensor(y_norm, dtype=torch.float32).unsqueeze(0)
            ).numpy()[0]
        return self.data.denorm_output(pred)

    def step(self, y_act: np.ndarray, penalty: float = 0.0) -> Tuple[np.ndarray, float]:
        self.x_sim = self._simulate(y_act)
        r = self.reward_fn(self.x_sim, penalty)
        return self.x_sim, r

    def get_state(self) -> np.ndarray:
        return np.concatenate([self.x_sim, self.x_tar])


def state_synthesis(x_sim: np.ndarray, x_tar: np.ndarray) -> np.ndarray:
    return np.concatenate([x_sim, x_tar])


# ══════════════════════════════════════════════════════════════
# 5. SAC 網路
# ══════════════════════════════════════════════════════════════

LOG_STD_MIN, LOG_STD_MAX = -5, 2

class SACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256, 256, 256)):
        super().__init__()
        layers, d = [], state_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.net      = nn.Sequential(*layers)
        self.mean_out = nn.Linear(d, action_dim)
        self.std_out  = nn.Linear(d, action_dim)

    def forward(self, state):
        h = self.net(state)
        mean = self.mean_out(h)
        log_std = self.std_out(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x = dist.rsample()
        y = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return y, log_prob, torch.tanh(mean)


class SACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(256, 256, 256)):
        super().__init__()
        def _mlp():
            layers, d = [], state_dim + action_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            layers.append(nn.Linear(d, 1))
            return nn.Sequential(*layers)
        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        to = lambda x: torch.tensor(np.stack(x), dtype=torch.float32)
        return to(s), to(a), to(r), to(s2), to(d)

    def __len__(self):
        return len(self.buf)


class SACTrainer:
    def __init__(self, state_dim: int, action_dim: int, cfg: Dict):
        self.cfg = cfg
        self.policy     = SACPolicy(state_dim, action_dim, cfg["hidden_dims"])
        self.critic     = SACCritic(state_dim, action_dim, cfg["hidden_dims"])
        self.critic_tgt = SACCritic(state_dim, action_dim, cfg["hidden_dims"])
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.log_alpha  = torch.tensor(np.log(cfg["init_alpha"]), requires_grad=True)
        self.target_ent = -action_dim

        self.opt_actor  = optim.Adam(self.policy.parameters(), lr=cfg["lr_actor"])
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg["lr_critic"])
        self.opt_alpha  = optim.Adam([self.log_alpha], lr=cfg["lr_alpha"])

        self.buffer = ReplayBuffer(cfg["buffer_size"])

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self) -> Dict[str, float]:
        cfg = self.cfg
        s, a, r, s2, d = self.buffer.sample(cfg["batch_size"])

        with torch.no_grad():
            a2, log_pi2, _ = self.policy.sample(s2)
            q1_t, q2_t = self.critic_tgt(s2, a2)
            q_t = torch.min(q1_t, q2_t) - self.alpha * log_pi2
            y = r + cfg["gamma"] * (1 - d) * q_t

        q1, q2 = self.critic(s, a)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_critic.zero_grad(); c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_critic.step()

        a_new, log_pi, _ = self.policy.sample(s)
        q1_n, q2_n = self.critic(s, a_new)
        q_n = torch.min(q1_n, q2_n)
        a_loss = (self.alpha.detach() * log_pi - q_n).mean()
        self.opt_actor.zero_grad(); a_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt_actor.step()

        ent_loss = -(self.log_alpha * (log_pi + self.target_ent).detach()).mean()
        self.opt_alpha.zero_grad(); ent_loss.backward(); self.opt_alpha.step()

        tau = cfg["tau"]
        for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        return {"critic_loss": c_loss.item(), "actor_loss": a_loss.item(), "alpha": self.alpha.item()}

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, _, mean = self.policy.sample(s)
        chosen = mean if deterministic else action
        return chosen.detach().numpy()[0]

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "log_alpha": self.log_alpha.detach(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ckpt["policy"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_tgt.load_state_dict(ckpt["critic"])   # target 也同步，避免載入後 target 是隨機權重
        self.log_alpha = ckpt["log_alpha"].requires_grad_(True)
        # 重要：opt_alpha 原本綁定的是「舊的」log_alpha tensor，
        # 換新 tensor 後必須重建 optimizer，否則之後繼續訓練時 alpha 不會更新
        self.opt_alpha = optim.Adam([self.log_alpha], lr=self.cfg["lr_alpha"])


# ══════════════════════════════════════════════════════════════
# 6. 動作縮放
# ══════════════════════════════════════════════════════════════

def action_to_param(action: np.ndarray, param_low: np.ndarray, param_high: np.ndarray) -> np.ndarray:
    a = np.clip(action, -1, 1)
    return param_low + (a + 1) / 2.0 * (param_high - param_low)


def param_to_action(param: np.ndarray, param_low: np.ndarray, param_high: np.ndarray) -> np.ndarray:
    ratio = (param - param_low) / (param_high - param_low + 1e-8)
    return ratio * 2.0 - 1.0


def constraint_check(y_act: np.ndarray, param_low: np.ndarray, param_high: np.ndarray,
                      penalty_val: float = -0.5) -> Tuple[bool, float]:
    ok = bool(np.all(y_act >= param_low) and np.all(y_act <= param_high))
    return ok, (0.0 if ok else penalty_val)


# ══════════════════════════════════════════════════════════════
# 7. BC 初始化：把資料集樣本填入 Replay Buffer
# ══════════════════════════════════════════════════════════════

def bc_init_buffer(sac: SACTrainer, env: SurrogateEnv, data: ProcessedData,
                    n_samples: int, x_tar: np.ndarray):
    """
    讓 SAC 一開始不是從隨機動作探索，而是先看過歷史上「真實出現過」的設計，
    加速收斂並降低訓練初期的無效探索。
    """
    print(f"[BC init] 從資料集抽樣 {n_samples} 筆填入 Replay Buffer...")
    n_samples = min(n_samples, len(data.params_raw))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(data.params_raw), size=n_samples, replace=False)

    for i in idx:
        p = data.params_raw[i]
        o = data.outputs_raw[i]
        a = param_to_action(p, data.param_low, data.param_high)
        s = state_synthesis(o, x_tar)
        _, r = env.step(p)
        s2 = env.get_state()
        sac.buffer.push(s, a, np.array([r]), s2, np.array([0.0]))

    print(f"[BC init] 完成，Buffer 現有 {len(sac.buffer)} 筆")


# ══════════════════════════════════════════════════════════════
# 8. SAC 訓練主流程
# ══════════════════════════════════════════════════════════════

def train_sac(sac: SACTrainer, env: SurrogateEnv, data: ProcessedData,
              cfg: Dict, save_path: str) -> Dict:
    history = {"step": [], "reward": [], "error_norm": []}
    best_r = -np.inf

    print("\n" + "═" * 60)
    print("  SAC 訓練  |  Surrogate 快速環境")
    print("═" * 60)

    for step in range(1, cfg["total_steps"] + 1):
        state = env.get_state()

        if len(sac.buffer) < SAC_CONFIG["warmup_steps"]:
            action = np.random.uniform(-1, 1, data.action_dim)
        else:
            action = sac.select_action(state)

        # 註：SAC 動作經 tanh 壓縮 + action_to_param 的 clip 後，
        # 幾何參數「必然」落在 [param_low, param_high] 內，
        # 邊界約束由動作空間設計本身保證，不需要額外的 penalty 機制。
        # （舊版的 constraint_check 在此永遠回傳 True，屬於無效程式碼，已移除）
        y_act = action_to_param(action, data.param_low, data.param_high)
        x_sim, r = env.step(y_act)
        next_state = env.get_state()

        sac.buffer.push(state, action, np.array([r]), next_state, np.array([0.0]))

        if len(sac.buffer) >= SAC_CONFIG["warmup_steps"]:
            for _ in range(SAC_CONFIG["update_per_step"]):
                sac.update()

        err = float(np.linalg.norm(x_sim - env.x_tar))
        history["step"].append(step)
        history["reward"].append(r)
        history["error_norm"].append(err)

        if step % cfg["log_every"] == 0:
            avg_r = float(np.mean(history["reward"][-200:]))
            print(f"  Step {step:6d} | avg_r={avg_r:7.4f} | ‖err‖={err:.3f}")
            if avg_r > best_r:
                best_r = avg_r
                sac.save(save_path)
                print(f"    → 新最佳 avg_r={best_r:.4f}，模型已儲存")

    return history


# ══════════════════════════════════════════════════════════════
# 9. 推論
# ══════════════════════════════════════════════════════════════

def infer(sac: SACTrainer, env: SurrogateEnv, data: ProcessedData) -> np.ndarray:
    sac.policy.eval()
    state = env.get_state()
    action = sac.select_action(state, deterministic=True)
    y_act = action_to_param(action, data.param_low, data.param_high)
    print("\n最終幾何參數建議：")
    for name, val in zip(data.param_names, y_act):
        print(f"  {name:20s} = {val:.5f}")
    return y_act


# ══════════════════════════════════════════════════════════════
# 10. Entry Point
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("  模型訓練  (02_train_model.py)")
    print("=" * 60)

    # 1. 載入已清理資料包
    data = ProcessedData(DATA_NPZ_PATH, META_JSON_PATH)

    # 檢查 TARGET_PERFORMANCE / REWARD_WEIGHTS 的 key 是否對得上 meta 的輸出欄位
    missing_tar = [n for n in data.output_names if n not in TARGET_PERFORMANCE]
    missing_w   = [n for n in data.output_names if n not in REWARD_WEIGHTS]
    if missing_tar or missing_w:
        raise RuntimeError(
            f"設定不完整：TARGET_PERFORMANCE 缺少 {missing_tar}，"
            f"REWARD_WEIGHTS 缺少 {missing_w}。請對照 processed_meta.json 的 output_names 補齊。"
        )

    # 2. 預訓練 Surrogate
    surrogate = SurrogateModel(in_dim=data.action_dim, out_dim=data.output_dim)
    pretrain_history = pretrain_surrogate(surrogate, data, PRETRAIN_CONFIG)
    torch.save(surrogate.state_dict(), SURROGATE_SAVE_PATH)
    print(f"[Save] Surrogate 已儲存至 {SURROGATE_SAVE_PATH}")

    # 3. 建立環境
    x_tar = np.array([TARGET_PERFORMANCE[n] for n in data.output_names], dtype=np.float32)
    env = SurrogateEnv(surrogate, data, x_tar, REWARD_WEIGHTS)

    # 4. 建立 SAC
    sac = SACTrainer(data.state_dim, data.action_dim, SAC_CONFIG)

    # 5. BC 初始化
    bc_init_buffer(sac, env, data, TRAIN_CONFIG["bc_init_samples"], x_tar)

    # 6. 訓練
    t0 = time.time()
    sac_history = train_sac(sac, env, data, TRAIN_CONFIG, SAC_SAVE_PATH)
    elapsed = time.time() - t0
    print(f"\n[完成] SAC 訓練耗時 {elapsed/60:.1f} 分鐘")

    # 7. 儲存訓練歷史
    with open(TRAIN_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "pretrain": pretrain_history,
            "sac": {k: v[::10] for k, v in sac_history.items()},  # 每 10 步存一次，避免檔案過大
            "elapsed_sec": elapsed,
        }, f, indent=2)
    print(f"[Save] 訓練歷史已儲存至 {TRAIN_HISTORY_PATH}")

    # 8. 推論最佳結果
    sac.load(SAC_SAVE_PATH)
    infer(sac, env, data)


if __name__ == "__main__":
    main()
