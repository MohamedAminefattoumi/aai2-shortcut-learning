"""Regenerate every result and figure of the project with one command.

It runs the main scripts one after the other, in this order:
    1. data_checks        checks the data and saves results/figures/data_examples.png
    2. run_experiments    mitigation grid -> results/results.csv (+ checkpoints for XAI)
    3. run_architectures  MLP vs CNN -> results/architectures*.csv, confusion_*.csv
    4. report             tables and figures -> results/summary.md, results/figures/*.png
    5. run_xai            Grad-CAM, Integrated Gradients and LIME -> results/xai/*.png

Usage, from the repository root:
    python -m src.run_all            # everything, full size (slow on a CPU)
    python -m src.run_all --quick    # small version of steps 1-4, only to check that the code runs
"""

import argparse
import subprocess
import sys
import time


def run(module, quick=False):
    """Run 'python -m <module>' and stop everything if it fails."""
    command = [sys.executable, "-m", module] + (["--quick"] if quick else [])
    print(f"\n===== {' '.join(command[1:])}", flush=True)
    start = time.time()
    subprocess.run(command, check=True)          # check=True: raise an error if the step fails
    print(f"===== {module} finished in {time.time() - start:.0f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="small and fast run, writes the *_quick files (only to check the code)")
    args = parser.parse_args()

    run("src.data_checks")
    run("src.run_experiments", args.quick)
    run("src.run_architectures", args.quick)
    run("src.report", args.quick)
    # XAI needs the p = 0.99 checkpoints that only the full run of run_experiments saves.
    if not args.quick:
        run("src.run_xai")

    print("\nall done: see results/ and results/figures/")


if __name__ == "__main__":
    main()
