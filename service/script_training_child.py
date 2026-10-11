"""执行入口快照；Linux 父 worker 异常退出时清理整个任务进程组。"""

import argparse
import os
import runpy
import signal
import sys
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        def watch_parent():
            sys.stdin.buffer.read()
            os.killpg(os.getpgrp(), signal.SIGKILL)
        threading.Thread(target=watch_parent, daemon=True).start()
    source = args.directory / "source.py"
    sys.argv = [str(source), "--config", str(args.directory / "config.json")]
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
