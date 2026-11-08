"""Run every experiment in the order of the paper, then the independent check."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ["sec2_forward_model.py", "sec4_degree_weight.py", "sec5_1_retained_rank.py", "sec5_2_benchmarks.py",
           "sec5_3_prior_study.py", "sec5_4_endpoint_energy.py", "sec5_5_synthesis.py", "sec5_6_data_driven.py",
           "make_figures.py"]

for name in SCRIPTS:
    print(f"\n===== experiments/{name}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "experiments" / name)], check=True)
print("\n===== verify_paper.py", flush=True)
subprocess.run([sys.executable, str(ROOT / "verify_paper.py")], check=True)
