"""
02_train_model.py
=================
動態 Surrogate Model + SAC 訓練程式。

所有資料路徑、輸入／輸出名稱、單位、最佳化 target、reward 權重與
訓練超參數均由 ai_config.json 讀取。

使用方式：
  python 02_train_model.py --config ai_config.json
  python 02_train_model.py --config ai_config.json --surrogate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "ai_config.json"

DEFAULT_PRETRAIN_CONFIG = {
    "epochs": 50,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "early_stop": 8,
    "hidden_dims": [256, 256, 256],
}
DEFAULT_SAC_CONFIG = {
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_size": 100_000,
    "batch_size": 256,
    "warmup_steps": 1000,
    "update_per_step": 1,
    "init_alpha": 0.2,
    "hidden_dims": [256, 256, 256],
}
DEFAULT_TRAIN_CONFIG = {
    "total_steps": 10_000,
    "bc_init_samples": 5000,
    "log_every": 500,
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到設定檔：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"設定檔最外層必須是 JSON 物件：{path}")
    return data


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _merged(defaults: Mapping[str, Any], overrides: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result = dict(defaults)
    if overrides:
        result.update(dict(overrides))
    return result


def load_runtime_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = _load_json(config_path)
    base_dir = config_path.parent

    artifacts = dict(config.get("artifacts", {}))
    artifact_dir = _resolve_path(str(artifacts.get("directory", "ai_artifacts/default")), base_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pretrain = _merged(DEFAULT_PRETRAIN_CONFIG, config.get("training", {}).get("surrogate"))
    sac = _merged(DEFAULT_SAC_CONFIG, config.get("training", {}).get("sac"))
    run = _merged(DEFAULT_TRAIN_CONFIG, config.get("training", {}).get("run"))
    pretrain["hidden_dims"] = tuple(int(v) for v in pretrain.get("hidden_dims", [256, 256, 256]))
    sac["hidden_dims"] = tuple(int(v) for v in sac.get("hidden_dims", [256, 256, 256]))

    objectives = dict(config.get("optimization", {}).get("objectives", {}))
    target_performance = {
        name: float(spec["target"])
        for name, spec in objectives.items()
        if isinstance(spec, dict) and spec.get("target") is not None
    }
    reward_weights = {
        name: float(spec.get("weight", 0.0))
        for name, spec in objectives.items()
        if isinstance(spec, dict)
    }

    return {
        "config": config,
        "config_path": config_path,
        "artifact_dir": artifact_dir,
        "data_npz": artifact_dir / str(artifacts.get("processed_data", "processed_data.npz")),
        "meta_json": artifact_dir / str(artifacts.get("metadata", "processed_meta.json")),
        "surrogate_path": artifact_dir / str(artifacts.get("surrogate_model", "surrogate.pt")),
        "sac_path": artifact_dir / str(artifacts.get("sac_model", "sac_quantum.pt")),
        "history_path": artifact_dir / str(artifacts.get("train_history", "train_history.json")),
        "candidate_path": artifact_dir / str(artifacts.get("candidate", "sac_candidate.json")),
        "pretrain": pretrain,
        "sac": sac,
        "run": run,
        "objectives": objectives,
        "target_performance": target_performance,
        "reward_weights": reward_weights,
    }


def _default_exports() -> Dict[str, Any]:
    if DEFAULT_CONFIG_PATH.is_file():
        return load_runtime_config(DEFAULT_CONFIG_PATH)
    artifact_dir = HERE
    return {
        "config": {},
        "config_path": DEFAULT_CONFIG_PATH,
        "artifact_dir": artifact_dir,
        "data_npz": artifact_dir / "processed_data.npz",
        "meta_json": artifact_dir / "processed_meta.json",
        "surrogate_path": artifact_dir / "surrogate.pt",
        "sac_path": artifact_dir / "sac_quantum.pt",
        "history_path": artifact_dir / "train_history.json",
        "candidate_path": artifact_dir / "sac_candidate.json",
        "pretrain": dict(DEFAULT_PRETRAIN_CONFIG, hidden_dims=(256, 256, 256)),
        "sac": dict(DEFAULT_SAC_CONFIG, hidden_dims=(256, 256, 256)),
        "run": dict(DEFAULT_TRAIN_CONFIG),
        "objectives": {},
        "target_performance": {},
        "reward_weights": {},
    }


# 保留這些名稱，讓既有的 04 報告程式仍可 import。
_DEFAULT = _default_exports()
DATA_NPZ_PATH = str(_DEFAULT["data_npz"])
META_JSON_PATH = str(_DEFAULT["meta_json"])
SURROGATE_SAVE_PATH = str(_DEFAULT["surrogate_path"])
SAC_SAVE_PATH = str(_DEFAULT["sac_path"])
TRAIN_HISTORY_PATH = str(_DEFAULT["history_path"])
TARGET_PERFORMANCE = dict(_DEFAULT["target_performance"])
REWARD_WEIGHTS = dict(_DEFAULT["reward_weights"])
PRETRAIN_CONFIG = dict(_DEFAULT["pretrain"])
SAC_CONFIG = dict(_DEFAULT["sac"])
TRAIN_CONFIG = dict(_DEFAULT["run"])

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
        self.param_units = dict(self.meta.get("param_units", {}))
        self.output_units = dict(self.meta.get("output_units", {}))
        self.feature_specs = list(self.meta.get("feature_specs", []))
        self.target_specs = list(self.meta.get("target_specs", []))

        print(f"[Data] {self.meta['n_samples']} 筆樣本 | "
              f"action_dim={self.action_dim} | output_dim={self.output_dim} | "
              f"train={self.meta['n_train']} / val={self.meta['n_val']}")
        if self.meta.get("fixed_params"):
            print(f"[Data] 已排除固定參數：{list(self.meta['fixed_params'].keys())}")

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

    # ── 依 processed_meta.json 的 output_units 顯示物理單位 ──
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
        unit = data.output_units.get(name, "")
        unit_text = f" {unit}" if unit else ""
        print(f"    {name:<24} MAE={m:10.5f}{unit_text} | 平均相對誤差={r_:6.2f}%")
    print()
    return history


# ══════════════════════════════════════════════════════════════
# 3. Reward 函式
# ══════════════════════════════════════════════════════════════

def build_reward_fn(output_names: List[str], x_tar: np.ndarray, weights: Dict[str, float]):
    """r = -Σ wᵢ · |x^sim_i - x^tar_i| / |x^tar_i|，wᵢ 依 output_names 順序取出並正規化"""
    w = np.array([float(weights.get(n, 0.0)) for n in output_names], dtype=np.float32)
    if np.any(w < 0):
        raise ValueError("Reward weight 不可為負值。")
    if float(w.sum()) <= 0:
        raise ValueError("至少一個輸出欄位的 Reward weight 必須大於 0。")
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

        if len(sac.buffer) < sac.cfg["warmup_steps"]:
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

        if len(sac.buffer) >= sac.cfg["warmup_steps"]:
            for _ in range(sac.cfg["update_per_step"]):
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

    if not Path(save_path).exists():
        sac.save(save_path)
        print(f"    → 訓練結束，已儲存 SAC 模型：{save_path}")
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
# 10. 動態設定、候選輸出與 Entry Point
# ══════════════════════════════════════════════════════════════


def build_optimization_vectors(
    runtime: Mapping[str, Any],
    data: ProcessedData,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, str]]:
    objectives = dict(runtime.get("objectives", {}))
    targets: List[float] = []
    weights: Dict[str, float] = {}
    units: Dict[str, str] = {}

    for index, name in enumerate(data.output_names):
        spec = objectives.get(name, {})
        if not isinstance(spec, dict):
            raise TypeError(f"optimization.objectives.{name} 必須是 JSON 物件。")

        # 未設定 target 的輸出以訓練集平均值作為 state 參考，但 reward 權重預設為 0。
        target = float(spec.get("target", data.output_mean[index]))
        weight = float(spec.get("weight", 0.0))
        config_unit = str(spec.get("unit", ""))
        data_unit = str(data.output_units.get(name, ""))
        if config_unit and data_unit and config_unit != data_unit:
            raise ValueError(
                f"輸出 {name} 單位不一致：資料為 {data_unit!r}，objective 為 {config_unit!r}"
            )
        targets.append(target)
        weights[name] = weight
        units[name] = data_unit or config_unit

    if sum(weights.values()) <= 0:
        raise ValueError("optimization.objectives 至少要有一個 weight > 0。")
    return np.asarray(targets, dtype=np.float32), weights, units


def predict_surrogate(
    surrogate: SurrogateModel,
    data: ProcessedData,
    parameters: np.ndarray,
) -> np.ndarray:
    surrogate.eval()
    with torch.no_grad():
        normalized = torch.tensor(
            data.norm_param(parameters), dtype=torch.float32
        ).unsqueeze(0)
        prediction_norm = surrogate(normalized).cpu().numpy()[0]
    return data.denorm_output(prediction_norm)


def save_candidate(
    path: Path,
    data: ProcessedData,
    parameters: np.ndarray,
    predicted: np.ndarray,
    targets: np.ndarray,
    weights: Mapping[str, float],
    units: Mapping[str, str],
    runtime: Mapping[str, Any],
) -> None:
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(runtime["config_path"]),
        "data_hash": data.meta["data_hash"],
        "parameters": {
            name: float(value) for name, value in zip(data.param_names, parameters)
        },
        "parameter_units": {
            name: data.param_units.get(name, "") for name in data.param_names
        },
        "predicted_outputs": {
            name: float(value) for name, value in zip(data.output_names, predicted)
        },
        "targets": {
            name: float(value) for name, value in zip(data.output_names, targets)
        },
        "reward_weights": {name: float(weights.get(name, 0.0)) for name in data.output_names},
        "output_units": {name: units.get(name, "") for name in data.output_names},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"[Save] SAC 候選參數已儲存至 {path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="動態 Surrogate + SAC 訓練")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="AI JSON 設定檔。")
    parser.add_argument(
        "--surrogate-only",
        action="store_true",
        help="只訓練 Surrogate，不訓練 SAC。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    runtime = load_runtime_config(args.config)

    seed = int(runtime["config"].get("training", {}).get("random_seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 72)
    print("  動態模型訓練  (02_train_model.py)")
    print("=" * 72)
    print(f"設定檔：{runtime['config_path']}")

    data = ProcessedData(str(runtime["data_npz"]), str(runtime["meta_json"]))
    target_vector, reward_weights, output_units = build_optimization_vectors(runtime, data)

    surrogate = SurrogateModel(
        in_dim=data.action_dim,
        out_dim=data.output_dim,
        hidden=runtime["pretrain"]["hidden_dims"],
    )
    pretrain_history = pretrain_surrogate(surrogate, data, runtime["pretrain"])
    torch.save(surrogate.state_dict(), runtime["surrogate_path"])
    print(f"[Save] Surrogate 已儲存至 {runtime['surrogate_path']}")

    if args.surrogate_only:
        with Path(runtime["history_path"]).open("w", encoding="utf-8") as file:
            json.dump({"pretrain": pretrain_history, "sac": None}, file, indent=2)
        print("[完成] 已依 --surrogate-only 跳過 SAC。")
        return

    env = SurrogateEnv(surrogate, data, target_vector, reward_weights)
    sac = SACTrainer(data.state_dim, data.action_dim, runtime["sac"])
    bc_init_buffer(
        sac,
        env,
        data,
        int(runtime["run"]["bc_init_samples"]),
        target_vector,
    )

    start_time = time.time()
    sac_history = train_sac(
        sac,
        env,
        data,
        runtime["run"],
        str(runtime["sac_path"]),
    )
    elapsed = time.time() - start_time
    print(f"\n[完成] SAC 訓練耗時 {elapsed / 60:.1f} 分鐘")

    with Path(runtime["history_path"]).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "pretrain": pretrain_history,
                "sac": {key: values[::10] for key, values in sac_history.items()},
                "elapsed_sec": elapsed,
                "config_path": str(runtime["config_path"]),
                "data_hash": data.meta["data_hash"],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Save] 訓練歷史已儲存至 {runtime['history_path']}")

    sac.load(str(runtime["sac_path"]))
    best_parameters = infer(sac, env, data)
    predicted_outputs = predict_surrogate(surrogate, data, best_parameters)
    save_candidate(
        Path(runtime["candidate_path"]),
        data,
        best_parameters,
        predicted_outputs,
        target_vector,
        reward_weights,
        output_units,
        runtime,
    )


if __name__ == "__main__":
    main()
