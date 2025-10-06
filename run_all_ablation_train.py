#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
from datetime import datetime


EXPERIMENTS = [
    "baseline",
    "film_ctx",
    "film_kdec",
    "film_kdec_key",
    # "ht",
]


def launch_training(exp: str, conf_dir: str, data_name: str) -> subprocess.Popen:
    conf_path = os.path.join(conf_dir, f"{exp}.yaml")
    # For logging path, we do not know data_root; use provided data_name to prefix if available
    prefix = f"{data_name}_" if data_name else ""
    save_dir = os.path.join("checkpoints", f"{prefix}{exp}")
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    with open(log_path, "a") as f:
        f.write(f"\n--- Launch {exp} at {datetime.now().isoformat()} ---\n")
    # Launch in background with stdout/stderr redirected to log
    log_file = open(log_path, "a")
    cmd = [sys.executable, "train.py", conf_path]

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        close_fds=True,
    )
    return proc


def main():
    parser = argparse.ArgumentParser(description="Run all ablation trainings.")
    parser.add_argument("conf_dir", nargs="?", default="config", help="Directory of config YAMLs")
    parser.add_argument("--sequential", action="store_true", help="Run experiments sequentially (default: parallel)")
    parser.add_argument("--data-name", default=None, help="Dataset name prefix for checkpoints (defaults per config data_root)")
    args = parser.parse_args()

    procs = []

    def handle_sigint(signum, frame):
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    if args.sequential:
        for exp in EXPERIMENTS:
            p = launch_training(exp, args.conf_dir, args.data_name)
            procs.append(p)
            ret = p.wait()
            if ret != 0:
                print(f"Experiment {exp} exited with code {ret}")
    else:
        for exp in EXPERIMENTS:
            procs.append(launch_training(exp, args.conf_dir, args.data_name))
        # Wait and report statuses
        for exp, p in zip(EXPERIMENTS, procs):
            ret = p.wait()
            if ret != 0:
                print(f"Experiment {exp} exited with code {ret}")
            else:
                print(f"Experiment {exp} completed successfully")


if __name__ == "__main__":
    main()


