import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.eval.quantile.multi.charts import run_post_compute

eval_root = "result/eval"

for tag in ["xgb_quantile_val", "xgb_quantile_test",
            "xgb_quantile_market_state_val", "xgb_quantile_market_state_test"]:
    for pool in ["union", "intersection", "all"]:
        for ng in [10, 20]:
            out_dirs = {
                ret_h: os.path.join(eval_root, tag, pool, f"g{ng}", ret_h)
                for ret_h in ["ret100", "ret200", "ret300"]
                if os.path.isdir(os.path.join(eval_root, tag, pool, f"g{ng}", ret_h))
            }
            if out_dirs:
                print(f"[{tag}/{pool}/g{ng}] 重新生成图表...")
                run_post_compute(out_dirs)
