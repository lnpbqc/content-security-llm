"""训练子进程：stdout 专用于 JSON 事件，上传代码的打印重定向到 stderr。"""

import argparse
import contextlib
import json
import os
import sys
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    event_stream = sys.stdout

    if os.name == "nt":
        # 监控父进程句柄，不阻塞 stdin；NumPy 的 Windows DLL 初始化会访问 stdin。
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        handle = kernel.OpenProcess(0x00100000, False, args.parent_pid)
        if not handle:
            raise OSError(ctypes.get_last_error(), "无法监控父训练 worker")

        def parent_watchdog():
            kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
            os._exit(1)
    else:
        def parent_watchdog():
            sys.stdin.buffer.read()
            os._exit(1)

    threading.Thread(target=parent_watchdog, daemon=True).start()

    def emit(event):
        event_stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        event_stream.flush()

    with contextlib.redirect_stdout(sys.stderr):
        try:
            from service.training_engine import run_training
            config = json.loads((args.directory / "config.json").read_text(encoding="utf-8"))
            records = json.loads((args.directory / "data.json").read_text(encoding="utf-8"))
            run_training(config, records, args.directory, emit,
                         lambda: (args.directory / "cancel.request").exists())
            emit({"type": "finished", "status": "succeeded"})
        except Exception as exc:
            if type(exc).__name__ == "TrainingCanceled":
                emit({"type": "finished", "status": "canceled"})
            else:
                emit({"type": "finished", "status": "failed",
                      "error": "{}: {}".format(type(exc).__name__, exc)})
                raise


if __name__ == "__main__":
    main()
