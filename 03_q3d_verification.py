"""
03_q3d_verification.py
======================
AI 候選設計的真實 Q3D 回驗入口。

本程式不直接操作 PyAEDT，而是呼叫 ai_q3d_pipeline.py，最後再由
main_pipeline.py 使用既有 DRC → GDS → Q3D → LOM 流程。

使用方式：
  python 03_q3d_verification.py --config ai_config.json

指定候選檔：
  python 03_q3d_verification.py --config ai_config.json \
      --candidate-json ai_artifacts/rect_h_v1/sac_candidate.json

只產生候選 layout JSON，不啟動 Q3D：
  python 03_q3d_verification.py --config ai_config.json --check-only

臨時覆寫某個模型輸入：
  python 03_q3d_verification.py --config ai_config.json \
      --set in_qubit_q_c_dis=105 --set in_qubit_gap_size=35
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ai_q3d_pipeline import (
    DEFAULT_CONFIG_PATH,
    load_ai_context,
    load_candidate,
    prepare_candidate_layout,
    run_ai_candidate,
)


def parse_override(text: str) -> tuple[str, float]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--set 格式必須是 NAME=VALUE。")
    name, raw_value = text.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("--set 的 NAME 不可為空。")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--set 的 VALUE 必須是數值：{text}") from exc
    return name, value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用既有 QCQ EDA 流程回驗 AI 候選設計")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="AI JSON 設定檔。")
    parser.add_argument(
        "--candidate-json",
        default=None,
        help="候選參數 JSON；省略時使用 artifacts.candidate。",
    )
    parser.add_argument("--base-layout", default=None, help="臨時覆寫基準 layout JSON。")
    parser.add_argument("--sample-id", default=None, help="指定唯一 sample ID。")
    parser.add_argument("--run-name", default=None, help="指定資料庫 run_name。")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只回寫並輸出候選 layout JSON，不執行 DRC/Q3D/LOM。",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        type=parse_override,
        metavar="NAME=VALUE",
        help="臨時覆寫候選參數，可重複指定。",
    )
    return parser.parse_args()


def resolve_candidate_path(config_path: str, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    context = load_ai_context(config_path)
    candidate_name = str(context["artifacts"].get("candidate", "sac_candidate.json"))
    return (context["artifact_dir"] / candidate_name).resolve()


def print_comparison(result: Dict[str, Any]) -> None:
    print("\n" + "=" * 76)
    print("  Q3D 真實回驗結果")
    print("=" * 76)
    trace = result.get("trace", {})
    print(f"sample_id      : {result.get('sample_id')}")
    print(f"failure_stage  : {trace.get('failure_stage')}")
    print(f"sample_dir     : {trace.get('sample_dir')}")

    actual = result.get("actual_outputs", {})
    predicted = result.get("predicted_outputs", {})
    units = result.get("output_units", {})
    if actual:
        print("\n輸出比較：")
        for name, actual_value in actual.items():
            unit = units.get(name, "")
            predicted_value = predicted.get(name)
            if predicted_value is None:
                print(f"  {name:<28} Q3D/LOM={actual_value:12.6f} {unit}")
                continue
            rel = abs(actual_value - float(predicted_value)) / (abs(float(predicted_value)) + 1e-12) * 100
            print(
                f"  {name:<28} surrogate={float(predicted_value):12.6f} {unit} | "
                f"Q3D/LOM={actual_value:12.6f} {unit} | error={rel:8.3f}%"
            )
    else:
        print("\n尚未取得可映射的真實輸出；請查看 trace 與各階段 log。")

    print(f"\n驗證報告：{result.get('verification_result_path')}")


def main() -> int:
    args = parse_arguments()
    candidate_path = resolve_candidate_path(args.config, args.candidate_json)
    candidate = load_candidate(candidate_path)

    for name, value in args.set:
        candidate.setdefault("parameters", {})[name] = value

    sample_id = args.sample_id or datetime.now().strftime("ai_verify_%Y%m%d_%H%M%S_%f")

    if args.check_only:
        layout_path, preparation = prepare_candidate_layout(
            config_path=args.config,
            candidate=candidate,
            sample_id=sample_id,
            base_layout_path=args.base_layout,
        )
        print("候選 layout JSON 已建立，未啟動 Q3D：")
        print(f"  {layout_path}")
        print(json.dumps(preparation, indent=2, ensure_ascii=False))
        return 0

    result = run_ai_candidate(
        config_path=args.config,
        candidate=candidate,
        sample_id=sample_id,
        run_name=args.run_name,
        base_layout_path=args.base_layout,
    )
    print_comparison(result)
    trace = result.get("trace", {})
    return 0 if trace.get("failure_stage") == "None" else 1


if __name__ == "__main__":
    sys.exit(main())
