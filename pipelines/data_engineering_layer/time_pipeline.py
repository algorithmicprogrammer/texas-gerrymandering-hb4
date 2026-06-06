# time_pipeline.py  — put it next to pipeline.py, run it the way you run run_table3.py
import time
from pipeline import run_pipeline   # adjust if your entry point is named differently

t0 = time.perf_counter()
run_pipeline()
elapsed = time.perf_counter() - t0
print(f"\nData engineering layer: {elapsed:.1f} s ({elapsed/60:.2f} min)")