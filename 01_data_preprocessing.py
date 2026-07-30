"""
01_data_preprocessing.py
========================
動態資料前處理程式。

支援：
  * SQLite、CSV、Parquet 資料來源
  * 由 ai_config.json 動態指定輸入欄位、輸出欄位、名稱與單位
  * 動態輸入／輸出維度，不再補齊固定 13 維
  * NaN/Inf、重複資料、輸出離群值清理
  * 可選擇移除固定輸入參數
  * 僅使用訓練集統計量進行正規化

使用方式：
  python 01_data_preprocessing.py --config ai_config.json

可臨時覆寫資料來源：
  python 01_data_preprocessing.py --config ai_config.json \
      --source other.db --table simulation_records \
      --where "q3d_success = 1 AND lom_success = 1"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "ai_config.json"
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到設定檔：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 最外層必須是物件：{path}")
    return data


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def validate_identifier(name: str, label: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(name):
        raise ValueError(f"{label} 不是安全的識別名稱：{name!r}")
    return name


def normalize_field_specs(raw_specs: Sequence[Any], kind: str) -> List[Dict[str, Any]]:
    """
    欄位可用兩種方式設定：
      "column_name"
    或
      {
        "name": "模型使用名稱",
        "column": "資料來源欄位",
        "unit": "um",
        "json_path": "qubit.rect_length"
      }
    """
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"設定中的 {kind} 必須是非空陣列。")

    normalized: List[Dict[str, Any]] = []
    seen_names = set()
    seen_columns = set()

    for index, item in enumerate(raw_specs):
        if isinstance(item, str):
            spec = {"name": item, "column": item, "unit": ""}
        elif isinstance(item, dict):
            spec = dict(item)
            column = spec.get("column") or spec.get("name")
            name = spec.get("name") or column
            if not isinstance(column, str) or not column:
                raise ValueError(f"{kind}[{index}] 缺少 column/name。")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{kind}[{index}] 的 name 無效。")
            spec["column"] = column
            spec["name"] = name
            spec.setdefault("unit", "")
        else:
            raise TypeError(f"{kind}[{index}] 必須是字串或 JSON 物件。")

        validate_identifier(str(spec["column"]), f"{kind} column")
        if spec["name"] in seen_names:
            raise ValueError(f"{kind} 出現重複模型名稱：{spec['name']}")
        if spec["column"] in seen_columns:
            raise ValueError(f"{kind} 出現重複來源欄位：{spec['column']}")
        seen_names.add(spec["name"])
        seen_columns.add(spec["column"])
        normalized.append(spec)

    return normalized


def _to_float(value: Any, row_label: Any, column: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"資料列 {row_label!r} 的欄位 {column!r} 無法轉成數值：{value!r}"
        ) from exc


def _rows_to_arrays(
    rows: Sequence[Mapping[str, Any]],
    feature_specs: Sequence[Mapping[str, Any]],
    target_specs: Sequence[Mapping[str, Any]],
    id_column: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: List[List[float]] = []
    targets: List[List[float]] = []
    row_ids: List[str] = []

    for index, row in enumerate(rows):
        row_id = row.get(id_column) if id_column else row.get("__row_id__", index)
        row_ids.append(str(row_id))
        features.append(
            [_to_float(row.get(spec["column"]), row_id, spec["column"]) for spec in feature_specs]
        )
        targets.append(
            [_to_float(row.get(spec["column"]), row_id, spec["column"]) for spec in target_specs]
        )

    if not rows:
        raise RuntimeError("資料來源沒有任何資料列。")

    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(row_ids, dtype=np.str_),
    )


def load_sqlite(
    path: Path,
    source_cfg: Mapping[str, Any],
    feature_specs: Sequence[Mapping[str, Any]],
    target_specs: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = validate_identifier(str(source_cfg.get("table", "")), "SQLite table")
    id_column = source_cfg.get("id_column")
    if id_column:
        validate_identifier(str(id_column), "SQLite id_column")

    columns = [spec["column"] for spec in feature_specs] + [
        spec["column"] for spec in target_specs
    ]
    where = str(source_cfg.get("where", "")).strip()

    if not path.is_file():
        raise FileNotFoundError(f"找不到 SQLite 資料庫：{path}")

    with sqlite3.connect(str(path)) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        if table not in tables:
            raise RuntimeError(f"資料庫找不到資料表 {table!r}；現有資料表：{sorted(tables)}")

        cursor.execute(f'PRAGMA table_info("{table}")')
        existing = {row[1] for row in cursor.fetchall()}
        required = set(columns)
        if id_column:
            required.add(str(id_column))
        missing = sorted(required - existing)
        if missing:
            raise RuntimeError(f"資料表 {table!r} 缺少欄位：{missing}")

        select_items = []
        if id_column:
            select_items.append(f'"{id_column}" AS "__row_id__"')
        else:
            select_items.append('rowid AS "__row_id__"')
        select_items.extend(f'"{column}"' for column in columns)
        sql = f'SELECT {", ".join(select_items)} FROM "{table}"'
        if where:
            sql += f" WHERE {where}"

        cursor.execute(sql)
        column_names = [description[0] for description in cursor.description]
        rows = [dict(zip(column_names, values)) for values in cursor.fetchall()]

    return _rows_to_arrays(rows, feature_specs, target_specs, "__row_id__")


def load_csv_file(
    path: Path,
    source_cfg: Mapping[str, Any],
    feature_specs: Sequence[Mapping[str, Any]],
    target_specs: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 CSV：{path}")
    delimiter = str(source_cfg.get("delimiter", ","))
    encoding = str(source_cfg.get("encoding", "utf-8-sig"))
    id_column = source_cfg.get("id_column")

    with path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        existing = set(reader.fieldnames or [])
        required = {
            spec["column"] for spec in list(feature_specs) + list(target_specs)
        }
        if id_column:
            required.add(str(id_column))
        missing = sorted(required - existing)
        if missing:
            raise RuntimeError(f"CSV 缺少欄位：{missing}")
        rows = list(reader)

    return _rows_to_arrays(rows, feature_specs, target_specs, str(id_column) if id_column else None)


def load_parquet_file(
    path: Path,
    source_cfg: Mapping[str, Any],
    feature_specs: Sequence[Mapping[str, Any]],
    target_specs: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "讀取 Parquet 需要 pandas 與 pyarrow：pip install pandas pyarrow"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(f"找不到 Parquet：{path}")
    frame = pd.read_parquet(path)
    id_column = source_cfg.get("id_column")
    required = {spec["column"] for spec in list(feature_specs) + list(target_specs)}
    if id_column:
        required.add(str(id_column))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Parquet 缺少欄位：{missing}")

    rows = frame.to_dict(orient="records")
    return _rows_to_arrays(rows, feature_specs, target_specs, str(id_column) if id_column else None)


def load_source(
    source_cfg: Mapping[str, Any],
    config_dir: Path,
    feature_specs: Sequence[Mapping[str, Any]],
    target_specs: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    source_type = str(source_cfg.get("type") or source_cfg.get("source_type") or "sqlite").lower()
    raw_path = source_cfg.get("path")
    if not raw_path:
        raise ValueError("dataset.path 不可為空。")
    path = resolve_path(str(raw_path), config_dir)

    print(f"[1/6] 讀取資料：type={source_type}, path={path}")
    if source_type in {"sqlite", "sqlite3", "db"}:
        arrays = load_sqlite(path, source_cfg, feature_specs, target_specs)
    elif source_type == "csv":
        arrays = load_csv_file(path, source_cfg, feature_specs, target_specs)
    elif source_type in {"parquet", "pq"}:
        arrays = load_parquet_file(path, source_cfg, feature_specs, target_specs)
    else:
        raise ValueError(f"不支援的 dataset.type：{source_type}")

    return arrays[0], arrays[1], arrays[2], path


def clean_data(
    features: np.ndarray,
    targets: np.ndarray,
    row_ids: np.ndarray,
    cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    start_count = len(features)
    stats = {"initial": start_count, "non_finite": 0, "duplicates": 0, "outliers": 0}
    print(f"\n[2/6] 資料清理（起始 {start_count} 筆）")

    finite = np.isfinite(features).all(axis=1) & np.isfinite(targets).all(axis=1)
    stats["non_finite"] = int((~finite).sum())
    features, targets, row_ids = features[finite], targets[finite], row_ids[finite]

    if bool(cfg.get("drop_duplicate_rows", True)):
        combined = np.concatenate([features, targets], axis=1)
        _, indices = np.unique(combined, axis=0, return_index=True)
        indices = np.sort(indices)
        stats["duplicates"] = int(len(combined) - len(indices))
        features, targets, row_ids = features[indices], targets[indices], row_ids[indices]

    threshold = cfg.get("outlier_z_threshold", 5.0)
    if threshold is not None and float(threshold) > 0 and len(targets) >= 3:
        target_std = targets.std(axis=0)
        safe_std = np.where(target_std < 1e-12, 1.0, target_std)
        z_score = np.abs((targets - targets.mean(axis=0)) / safe_std)
        outliers = (z_score > float(threshold)).any(axis=1)
        stats["outliers"] = int(outliers.sum())
        keep = ~outliers
        features, targets, row_ids = features[keep], targets[keep], row_ids[keep]

    if len(features) < 3:
        raise RuntimeError(f"清理後只剩 {len(features)} 筆，至少需要 3 筆資料。")

    print(
        "      非有限值移除={non_finite}, 重複移除={duplicates}, "
        "離群移除={outliers}, 最終={final}".format(final=len(features), **stats)
    )
    stats["final"] = int(len(features))
    return features, targets, row_ids, stats


def select_variable_features(
    features: np.ndarray,
    feature_specs: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
    threshold = float(cfg.get("fixed_std_threshold", 1e-5))
    drop_fixed = bool(cfg.get("drop_fixed_features", True))
    stds = features.std(axis=0)
    means = features.mean(axis=0)

    keep_indices: List[int] = []
    kept_specs: List[Dict[str, Any]] = []
    fixed: Dict[str, float] = {}
    for index, spec in enumerate(feature_specs):
        is_fixed = stds[index] < threshold
        if is_fixed:
            fixed[str(spec["name"])] = float(means[index])
        if not (drop_fixed and is_fixed):
            keep_indices.append(index)
            kept_specs.append(dict(spec))

    if not keep_indices:
        raise RuntimeError("所有輸入欄位皆被判定為固定參數，無法建立模型輸入。")

    selected = features[:, keep_indices]
    print(f"\n[3/6] 輸入欄位：指定 {len(feature_specs)} 個，模型實際使用 {len(kept_specs)} 個")
    print(f"      使用：{[spec['name'] for spec in kept_specs]}")
    if fixed:
        action = "移除" if drop_fixed else "保留"
        print(f"      固定欄位（{action}）：{fixed}")
    return selected, kept_specs, fixed


def make_split(n_samples: int, val_split: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("validation_split 必須介於 0 與 1 之間。")
    generator = np.random.default_rng(seed)
    indices = generator.permutation(n_samples)
    n_val = max(1, int(round(n_samples * val_split)))
    n_val = min(n_val, n_samples - 1)
    return indices[n_val:], indices[:n_val]


def build_action_bounds(
    features: np.ndarray,
    feature_specs: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed_low = features.min(axis=0)
    observed_high = features.max(axis=0)
    action_low = observed_low.copy()
    action_high = observed_high.copy()

    for index, spec in enumerate(feature_specs):
        if spec.get("min") is not None:
            action_low[index] = float(spec["min"])
        if spec.get("max") is not None:
            action_high[index] = float(spec["max"])
        if action_low[index] > action_high[index]:
            raise ValueError(
                f"輸入 {spec['name']} 的 min 大於 max："
                f"{action_low[index]} > {action_high[index]}"
            )

    return action_low, action_high, observed_low, observed_high


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="動態 QCQ AI 資料前處理")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="AI JSON 設定檔。")
    parser.add_argument("--source", default=None, help="臨時覆寫 dataset.path。")
    parser.add_argument("--source-type", default=None, help="臨時覆寫 dataset.type。")
    parser.add_argument("--table", default=None, help="臨時覆寫 SQLite table。")
    parser.add_argument("--where", default=None, help="臨時覆寫 SQLite WHERE 條件。")
    parser.add_argument("--output-dir", default=None, help="臨時覆寫 artifacts.directory。")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config_path = Path(args.config).expanduser().resolve()
    config = load_json(config_path)
    config_dir = config_path.parent

    source_cfg = dict(config.get("dataset", {}))
    if args.source is not None:
        source_cfg["path"] = args.source
    if args.source_type is not None:
        source_cfg["type"] = args.source_type
    if args.table is not None:
        source_cfg["table"] = args.table
    if args.where is not None:
        source_cfg["where"] = args.where

    feature_specs = normalize_field_specs(config.get("features", []), "features")
    target_specs = normalize_field_specs(config.get("targets", []), "targets")
    preprocessing_cfg = dict(config.get("preprocessing", {}))

    print("=" * 72)
    print("  動態資料前處理  (01_data_preprocessing.py)")
    print("=" * 72)

    raw_features, raw_targets, row_ids, source_path = load_source(
        source_cfg,
        config_dir,
        feature_specs,
        target_specs,
    )
    print(
        f"      讀到 {len(raw_features)} 筆，"
        f"輸入維度={raw_features.shape[1]}，輸出維度={raw_targets.shape[1]}"
    )

    features, targets, row_ids, cleaning_stats = clean_data(
        raw_features,
        raw_targets,
        row_ids,
        preprocessing_cfg,
    )
    features, active_feature_specs, fixed_params = select_variable_features(
        features,
        feature_specs,
        preprocessing_cfg,
    )

    validation_split = float(preprocessing_cfg.get("validation_split", 0.1))
    random_seed = int(preprocessing_cfg.get("random_seed", 42))
    train_idx, val_idx = make_split(len(features), validation_split, random_seed)
    print(f"\n[4/6] 資料切分：train={len(train_idx)}, validation={len(val_idx)}")

    feature_mean = features[train_idx].mean(axis=0)
    feature_std = features[train_idx].std(axis=0)
    feature_std = np.where(feature_std < 1e-12, 1.0, feature_std)
    target_mean = targets[train_idx].mean(axis=0)
    target_std = targets[train_idx].std(axis=0)
    target_std = np.where(target_std < 1e-12, 1.0, target_std)

    features_norm = (features - feature_mean) / feature_std
    targets_norm = (targets - target_mean) / target_std
    action_low, action_high, observed_low, observed_high = build_action_bounds(
        features,
        active_feature_specs,
    )
    print("[5/6] 正規化完成；統計量僅由訓練集計算。")

    artifacts_cfg = dict(config.get("artifacts", {}))
    artifact_dir_value = args.output_dir or artifacts_cfg.get("directory", "ai_artifacts/default")
    artifact_dir = resolve_path(str(artifact_dir_value), config_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    npz_path = artifact_dir / str(artifacts_cfg.get("processed_data", "processed_data.npz"))
    meta_path = artifact_dir / str(artifacts_cfg.get("metadata", "processed_meta.json"))

    feature_f32 = features.astype(np.float32)
    target_f32 = targets.astype(np.float32)
    np.savez_compressed(
        npz_path,
        params_raw=feature_f32,
        outputs_raw=target_f32,
        params_norm=features_norm.astype(np.float32),
        outputs_norm=targets_norm.astype(np.float32),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        row_ids=row_ids,
    )

    data_hash = hashlib.sha256(feature_f32.tobytes() + target_f32.tobytes()).hexdigest()[:16]
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]

    param_names = [str(spec["name"]) for spec in active_feature_specs]
    output_names = [str(spec["name"]) for spec in target_specs]
    meta = {
        "schema_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "source": {
            "type": str(source_cfg.get("type") or source_cfg.get("source_type") or "sqlite"),
            "path": str(source_path),
            "table": source_cfg.get("table"),
            "where": source_cfg.get("where"),
            "id_column": source_cfg.get("id_column"),
        },
        "n_samples": int(len(features)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "input_dim": int(len(param_names)),
        "output_dim": int(len(output_names)),
        "data_hash": data_hash,
        "param_names": param_names,
        "output_names": output_names,
        "param_source_columns": {
            str(spec["name"]): str(spec["column"]) for spec in active_feature_specs
        },
        "output_source_columns": {
            str(spec["name"]): str(spec["column"]) for spec in target_specs
        },
        "param_json_paths": {
            str(spec["name"]): spec.get("json_path") for spec in active_feature_specs
        },
        "param_units": {
            str(spec["name"]): str(spec.get("unit", "")) for spec in active_feature_specs
        },
        "output_units": {
            str(spec["name"]): str(spec.get("unit", "")) for spec in target_specs
        },
        "feature_specs": active_feature_specs,
        "target_specs": target_specs,
        "param_mean": feature_mean.tolist(),
        "param_std": feature_std.tolist(),
        "output_mean": target_mean.tolist(),
        "output_std": target_std.tolist(),
        "param_low": action_low.tolist(),
        "param_high": action_high.tolist(),
        "param_observed_low": observed_low.tolist(),
        "param_observed_high": observed_high.tolist(),
        "fixed_params": fixed_params,
        "pad_info": None,
        "cleaning": cleaning_stats,
        "preprocessing_config": {
            "fixed_std_threshold": float(preprocessing_cfg.get("fixed_std_threshold", 1e-5)),
            "drop_fixed_features": bool(preprocessing_cfg.get("drop_fixed_features", True)),
            "outlier_z_threshold": preprocessing_cfg.get("outlier_z_threshold", 5.0),
            "drop_duplicate_rows": bool(preprocessing_cfg.get("drop_duplicate_rows", True)),
            "validation_split": validation_split,
            "random_seed": random_seed,
        },
    }

    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)

    print("[6/6] 輸出完成")
    print(f"      NPZ：{npz_path}")
    print(f"      META：{meta_path}")
    print(f"      動態輸入維度={len(param_names)}，動態輸出維度={len(output_names)}")
    print(f"      data_hash={data_hash}")


if __name__ == "__main__":
    main()