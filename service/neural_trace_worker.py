"""独立串行 GPU worker；只有领取任务后才启动会加载模型的子进程。"""

import argparse
import csv
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from config import Settings
from db.database import Database
from db.neural_trace import NeuralTraceRepository
from service.neural_trace import MODEL_ROOT, STAGES, TRACE_ROOT, checked_result, collect_result, read_json
from utils.locking import exclusive_lock


def check_model(model_root=MODEL_ROOT):
    required = [model_root / name for name in ("config.json", "tokenizer_config.json", "tokenizer.json")]
    index = model_root / "model.safetensors.index.json"
    if index.is_file():
        names = set(read_json(index)["weight_map"].values())
        if not names or any(Path(name).name != name for name in names):
            raise RuntimeError("模型权重索引无效，请重新下载 Qwen3-4B。")
        required.extend(model_root / name for name in names)
    else:
        required.append(model_root / "model.safetensors")
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise RuntimeError("模型文件不完整，请按《神经因果溯源接口与部署.md》下载到 neurotrace-run/models/Qwen3-4B。")


def run_command(command, cwd, log_path):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            return_code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if return_code:
        raise RuntimeError(f"{log_path.stem} 执行失败（退出码 {return_code}）；请管理员检查该任务的阶段日志。")


class NeuralTraceWorker:
    def __init__(self, repository, run_root=None):
        self.repository = repository
        self.run_root = Path(run_root) if run_root else repository.database.path.parent / "neural-trace-runs"

    def run_once(self):
        task = self.repository.claim()
        if task is None:
            return False
        try:
            check_model()
            missing = [name for name in ("torch", "transformers", "nnsight", "accelerate", "matplotlib")
                       if importlib.util.find_spec(name) is None]
            if missing:
                raise RuntimeError("缺少溯源依赖：" + ", ".join(missing))
            # 原始脚本的文档阶段也会加载模型，须在它开始前检查共享 GPU。
            gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, check=True)
            if gpu.stdout.strip():
                raise RuntimeError("GPU 已有计算进程，请等待空闲后重新提交。")
            result = self.execute(task)
            self.repository.finish(task["id"], result=result)
        except BaseException as error:
            self.repository.finish(task["id"], error_message=(
                (str(error) or type(error).__name__) if isinstance(error, Exception)
                else "worker 停止，任务已中断，请重新提交。"))
            if not isinstance(error, Exception):
                raise
        return True

    def execute(self, task):
        directory = self.run_root / task["id"]
        directory.mkdir(parents=True, exist_ok=False)
        case_id = task["case_id"]
        upstream = directory / "upstream" / case_id
        upstream.mkdir(parents=True)
        for name, content in task["input"]["sources"].items():
            (upstream / f"{name}.json").write_text(content, encoding="utf-8", newline="\n")
        review = directory / "manual_review.csv"
        with review.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(task["input"]["review"]))
            writer.writeheader()
            writer.writerow(task["input"]["review"])
        documents, inputs, layers, activation, components = (
            directory / name for name in ("documents", "inputs", "layers", "activation", "components")
        )
        common = ["--case-id", case_id]
        commands = (
            ("document_trace.py", [*common, "--review-csv", review, "--results-dir", upstream.parent,
                                   "--output-dir", documents]),
            ("prepare_neural_trace_inputs.py", [*common, "--review-csv", review, "--results-dir", upstream.parent,
                                               "--document-trace-dir", documents, "--output-dir", inputs]),
            ("layer_scan.py", [*common, "--input-root", inputs, "--output-dir", layers,
                               "--layers", "all", "--patch-scope", "candidate_prediction_span"]),
            ("activation_difference_scan.py", [*common, "--input-root", inputs,
                                               "--layer-scan-dir", layers, "--output-dir", activation]),
            ("component_scan.py", [*common, "--input-root", inputs, "--layer-scan-dir", layers,
                                   "--output-dir", components, "--top-k", "3", "--generate-patched-answers"]),
        )
        outputs = dict(zip(STAGES, (
            documents / f"{case_id}.json", inputs / case_id / "token_alignment.json",
            layers / f"{case_id}.layer_scan.candidate_prediction_span.json",
            activation / f"{case_id}.activation_difference.json", components / f"{case_id}.component_scan.json",
        )))
        for stage, (script, arguments) in zip(STAGES, commands):
            self.repository.stage(task["id"], stage)
            command = [sys.executable, str(TRACE_ROOT / "forensics" / script), *map(str, arguments)]
            run_command(command, TRACE_ROOT, directory / f"{stage}.log")
            checked_result(outputs[stage], case_id, stage)
        self.repository.stage(task["id"], "collect_result")
        return collect_result(task, outputs)


def main():
    parser = argparse.ArgumentParser(description="串行执行神经因果溯源任务；不会在启动时加载模型")
    parser.add_argument("--once", action="store_true", help="最多执行一个任务后退出")
    parser.add_argument("--check", action="store_true", help="只检查模型文件和依赖存在性，不加载模型、不领取任务")
    args = parser.parse_args()
    if args.check:
        check_model()
        missing = [name for name in ("torch", "transformers", "nnsight", "accelerate", "matplotlib")
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError("缺少溯源依赖：" + ", ".join(missing))
        print("模型文件和依赖存在；尚未验证 CUDA、权重加载或实际 GPU 实验。")
        return
    settings = Settings.load()
    database = Database(settings.database_path)
    database.initialize()
    repository = NeuralTraceRepository(database)
    worker = NeuralTraceWorker(repository)
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    with exclusive_lock(database.path.parent / "neural-trace-worker" / "worker.lock", "神经溯源 worker 已运行。"):
        repository.recover_running()
        while True:
            worked = worker.run_once()
            if args.once:
                return
            if not worked:
                time.sleep(1)


if __name__ == "__main__":
    main()
