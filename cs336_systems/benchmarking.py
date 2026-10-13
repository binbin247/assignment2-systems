"""批量运行模型基准测试。

在项目根目录执行：python -m cs336_systems.benchmarking
修改 size_list、mode_list 和 common_args 可以选择不同的测试配置。
每组使用独立进程，结果保存为 JSON、日志及 summary.csv。
所有配置都会运行；存在失败配置时，脚本最终返回退出码 1。
"""

import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

size_list = ["small", "medium", "large", "xl", "10B"]
mode_list = ["forward", "forward-backward", "train"]

# 这些参数会传给 benchmarking_script.py，所有配置共用。
common_args = [
    "--device", "cuda:0",
    "--dtype", "float32",
    "--batch-size", "4",
    "--context-length", "512",
    "--warmup-steps", "5",
    "--num-steps", "10",
    "--timing", "breakdown",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
output_root = PROJECT_ROOT / "benchmark_results"
PHASES = ("zero_grad", "forward", "loss", "backward", "optimizer", "total")


def save_summary(rows: list[dict], path: Path) -> None:
    fields = [
        "model_size", "mode", "dtype", "context_length", "timing",
        "status", "returncode", "num_parameters", "result_file", "log_file",
    ]
    fields += [f"{phase}_{stat}_ms" for phase in PHASES for stat in ("mean", "std")]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True)
    rows = []
    total = len(size_list) * len(mode_list)
    successes = 0

    print(f"共 {total} 组配置，结果目录：{output_dir}", flush=True)

    for model_size in size_list:
        for mode in mode_list:
            index = len(rows) + 1
            case_name = f"{index:02d}_{model_size}_{mode}"
            result_path = output_dir / f"{case_name}.json"
            log_path = output_dir / f"{case_name}.log"
            command = [
                sys.executable, "-m", "cs336_systems.benchmarking_script",
                "--model-size", model_size,
                "--mode", mode,
                *common_args,
                "--output", str(result_path),
            ]

            print(f"[{index}/{total}] {model_size}, {mode}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                log.write(shlex.join(command) + "\n\n")
                log.flush()
                # check=False：某组失败后，仍然继续运行后面的配置。
                process = subprocess.run(
                    command, cwd=PROJECT_ROOT, stdout=log,
                    stderr=subprocess.STDOUT, check=False,
                )

            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(result, dict):
                    raise TypeError("结果 JSON 必须是一个对象")
            except (OSError, TypeError, ValueError) as error:
                result = {"status": "failed", "error": str(error)}
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n批量脚本读取结果失败：{error}\n")

            status = result.get("status", "failed")
            succeeded = process.returncode == 0 and status == "ok"
            if not succeeded and status == "ok":
                status = "failed"
            successes += int(succeeded)
            config = result.get("config", {})
            row = {
                "model_size": model_size,
                "mode": mode,
                "dtype": config.get("dtype", ""),
                "context_length": config.get("context_length", ""),
                "timing": config.get("timing", ""),
                "status": status,
                "returncode": process.returncode,
                "num_parameters": result.get("num_parameters", ""),
                "result_file": result_path.name if result_path.exists() else "",
                "log_file": log_path.name,
            }
            for phase, statistics in result.get("summary", {}).items():
                if phase in PHASES:
                    row[f"{phase}_mean_ms"] = statistics["mean_ms"]
                    row[f"{phase}_std_ms"] = statistics["std_ms"]
            rows.append(row)
            # 每组结束就更新汇总，保留已经完成的结果。
            save_summary(rows, output_dir / "summary.csv")
            print(f"  {status}，日志：{log_path.name}", flush=True)

    print(f"完成：成功 {successes}/{total}，失败 {total - successes}/{total}", flush=True)
    print(f"汇总：{output_dir / 'summary.csv'}", flush=True)
    return 0 if successes == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
