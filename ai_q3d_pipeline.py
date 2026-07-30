"""
ai_q3d_pipeline.py
==================
AI 與既有 EDA/Q3D 流水線之間的唯一橋接層。

本檔案不直接操作 PyAEDT。它只負責：
  1. 讀取 SAC/手動候選參數
  2. 依 processed_meta.json 的 param_json_paths 回寫 layout JSON
  3. 呼叫既有 main_pipeline.py
  4. 讀取 Q3D/LOM 結果並與 Surrogate 預測比較

這可確保「資料收集」與「AI 回驗」使用完全相同的 DRC、GDS、Q3D、LOM 程式。
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "ai_config.json"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 最外層必須是物件：{path}")
    return data


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def load_ai_context(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = load_json(config_path)
    config_dir = config_path.parent
    artifacts = dict(config.get("artifacts", {}))
    pipeline = dict(config.get("pipeline", {}))

    artifact_dir = resolve_path(
        str(artifacts.get("directory", "ai_artifacts/default")), config_dir
    )
    meta_path = artifact_dir / str(artifacts.get("metadata", "processed_meta.json"))
    main_pipeline_path = resolve_path(
        str(pipeline.get("main_pipeline_path", "main_pipeline.py")), config_dir
    )
    base_layout_path = resolve_path(
        str(pipeline.get("base_layout_path", "layout_parameters.json")), config_dir
    )

    return {
        "config": config,
        "config_path": config_path,
        "config_dir": config_dir,
        "artifact_dir": artifact_dir,
        "meta_path": meta_path,
        "main_pipeline_path": main_pipeline_path,
        "base_layout_path": base_layout_path,
        "pipeline": pipeline,
        "artifacts": artifacts,
    }


def load_candidate(candidate_path: str | Path) -> Dict[str, Any]:
    payload = load_json(Path(candidate_path).expanduser().resolve())
    parameters = payload.get("parameters", payload)
    if not isinstance(parameters, dict):
        raise ValueError("候選 JSON 必須包含 parameters 物件，或本身就是參數物件。")
    payload["parameters"] = parameters
    return payload


def _path_tokens(path: str) -> list[str]:
    tokens = [token.strip() for token in path.split(".") if token.strip()]
    if not tokens:
        raise ValueError(f"無效 JSON path：{path!r}")
    return tokens


def set_json_path(
    root: Dict[str, Any],
    path: str,
    value: Any,
    create_missing: bool = False,
) -> None:
    tokens = _path_tokens(path)
    current: Dict[str, Any] = root
    for token in tokens[:-1]:
        if token not in current:
            if not create_missing:
                raise KeyError(f"base layout 不存在 JSON path 節點：{path!r}（缺少 {token!r}）")
            current[token] = {}
        child = current[token]
        if not isinstance(child, dict):
            raise TypeError(f"JSON path {path!r} 的節點 {token!r} 不是物件。")
        current = child

    last = tokens[-1]
    if last not in current and not create_missing:
        raise KeyError(f"base layout 不存在 JSON path：{path!r}")
    current[last] = value


def get_json_path(root: Mapping[str, Any], path: str) -> Any:
    current: Any = root
    for token in _path_tokens(path):
        if not isinstance(current, Mapping) or token not in current:
            raise KeyError(f"結果 JSON 找不到 path：{path!r}")
        current = current[token]
    return current


def prepare_candidate_layout(
    config_path: str | Path,
    candidate: Mapping[str, Any],
    sample_id: str,
    base_layout_path: Optional[str | Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    context = load_ai_context(config_path)
    meta = load_json(context["meta_path"])
    parameters = candidate.get("parameters", candidate)
    if not isinstance(parameters, Mapping):
        raise TypeError("candidate.parameters 必須是物件。")

    base_path = (
        Path(base_layout_path).expanduser().resolve()
        if base_layout_path is not None
        else context["base_layout_path"]
    )
    layout = load_json(base_path)
    param_names = list(meta.get("param_names", []))
    json_paths = dict(meta.get("param_json_paths", {}))
    create_missing = bool(context["pipeline"].get("create_missing_json_paths", False))

    missing_parameters = [name for name in param_names if name not in parameters]
    if missing_parameters:
        raise ValueError(f"候選參數缺少模型輸入欄位：{missing_parameters}")

    applied: Dict[str, Any] = {}
    for name in param_names:
        json_path = json_paths.get(name)
        if not json_path:
            raise ValueError(
                f"輸入欄位 {name!r} 沒有 json_path。"
                "請在 ai_config.json 的 features 中設定 json_path，並重新執行 01。"
            )
        value = float(parameters[name])
        set_json_path(layout, str(json_path), value, create_missing=create_missing)
        applied[name] = {"value": value, "json_path": json_path}

    candidate_dir = context["artifact_dir"] / str(
        context["artifacts"].get("candidate_layout_directory", "candidate_layouts")
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    output_path = candidate_dir / f"{sample_id}_layout_parameters.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(layout, file, indent=2, ensure_ascii=False)

    preparation = {
        "base_layout_path": str(base_path),
        "candidate_layout_path": str(output_path),
        "applied_parameters": applied,
    }
    return output_path, preparation


def _load_module_from_path(module_path: Path):
    if not module_path.is_file():
        raise FileNotFoundError(f"找不到 main_pipeline.py：{module_path}")
    spec = importlib.util.spec_from_file_location("qcq_main_pipeline_for_ai", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入 main_pipeline.py：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_verification_outputs(
    context: Mapping[str, Any],
    sample_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    targets = list(context["config"].get("targets", []))
    actual: Dict[str, float] = {}
    units: Dict[str, str] = {}
    file_cache: Dict[Path, Dict[str, Any]] = {}

    for raw_spec in targets:
        if isinstance(raw_spec, str):
            continue
        if not isinstance(raw_spec, dict):
            continue
        name = str(raw_spec.get("name") or raw_spec.get("column") or "")
        verification = raw_spec.get("verification")
        if not name or not isinstance(verification, dict):
            continue
        filename = verification.get("file")
        json_path = verification.get("json_path")
        if not filename or not json_path:
            continue

        result_path = sample_dir / str(filename)
        if not result_path.is_file():
            continue
        if result_path not in file_cache:
            file_cache[result_path] = load_json(result_path)
        try:
            actual[name] = float(get_json_path(file_cache[result_path], str(json_path)))
            units[name] = str(raw_spec.get("unit", ""))
        except (KeyError, TypeError, ValueError):
            continue

    return actual, units


def _compare_values(
    actual: Mapping[str, float],
    reference: Mapping[str, Any],
) -> Dict[str, Dict[str, float]]:
    comparison: Dict[str, Dict[str, float]] = {}
    for name, actual_value in actual.items():
        if name not in reference:
            continue
        reference_value = float(reference[name])
        absolute_error = abs(actual_value - reference_value)
        relative_error = absolute_error / (abs(reference_value) + 1e-12) * 100.0
        comparison[name] = {
            "reference": reference_value,
            "actual": float(actual_value),
            "absolute_error": float(absolute_error),
            "relative_error_percent": float(relative_error),
        }
    return comparison


def run_ai_candidate(
    config_path: str | Path,
    candidate: Mapping[str, Any],
    sample_id: str,
    run_name: Optional[str] = None,
    base_layout_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    context = load_ai_context(config_path)
    candidate_layout, preparation = prepare_candidate_layout(
        config_path=config_path,
        candidate=candidate,
        sample_id=sample_id,
        base_layout_path=base_layout_path,
    )

    pipeline_module = _load_module_from_path(context["main_pipeline_path"])
    if not hasattr(pipeline_module, "main_pipeline_entry"):
        raise RuntimeError("main_pipeline.py 缺少 main_pipeline_entry。")

    selected_run_name = run_name or str(
        context["pipeline"].get("ai_run_name", "AI_Q3D_Verification")
    )
    trace = pipeline_module.main_pipeline_entry(
        sample_id=sample_id,
        run_name=selected_run_name,
        input_json=str(candidate_layout),
    )
    if trace is None:
        raise RuntimeError("main_pipeline_entry 沒有回傳 trace_record。")

    sample_dir = Path(trace["sample_dir"]).resolve()
    actual_outputs, output_units = extract_verification_outputs(context, sample_dir)
    predicted_outputs = candidate.get("predicted_outputs", {})
    target_outputs = candidate.get("targets", {})

    result = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(context["config_path"]),
        "sample_id": sample_id,
        "run_name": selected_run_name,
        "trace": trace,
        "preparation": preparation,
        "parameters": dict(candidate.get("parameters", candidate)),
        "predicted_outputs": dict(predicted_outputs) if isinstance(predicted_outputs, Mapping) else {},
        "target_outputs": dict(target_outputs) if isinstance(target_outputs, Mapping) else {},
        "actual_outputs": actual_outputs,
        "output_units": output_units,
        "surrogate_vs_q3d": _compare_values(
            actual_outputs,
            predicted_outputs if isinstance(predicted_outputs, Mapping) else {},
        ),
        "target_vs_q3d": _compare_values(
            actual_outputs,
            target_outputs if isinstance(target_outputs, Mapping) else {},
        ),
        "result_files": {
            "sample_dir": str(sample_dir),
            "q3d": str(sample_dir / "capacitance_matrix_results.json"),
            "lom": str(sample_dir / "lom_results.json"),
        },
    }

    result_dir = context["artifact_dir"] / str(
        context["artifacts"].get("verification_directory", "q3d_verifications")
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{sample_id}_verification.json"
    result["verification_result_path"] = str(result_path)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    return result
