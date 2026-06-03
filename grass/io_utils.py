import os
import pandas as pd


def save_metric_table(save_dir, metric_rows, filename="metrics_summary.csv"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(metric_rows)
    out_path = os.path.join(save_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path