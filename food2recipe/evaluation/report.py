# File: food2recipe/evaluation/report.py
import pandas as pd
from typing import Dict, List
from food2recipe.core.settings import load_settings

def save_report(metrics: Dict, confusion_data: List[Dict], report_name="eval_report"):
    settings = load_settings()
    report_dir = settings.REPORTS_DIR
    
    # Save Summary
    summary_path = report_dir / f"{report_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Evaluation Report\n")
        f.write("=================\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
            
    # Save Confusion/Detailed CSV
    # confusion_data: list of dicts {true, predicted, correct}
    df = pd.DataFrame(confusion_data)
    csv_path = report_dir / f"{report_name}_details.csv"
    df.to_csv(csv_path, index=False)
    
    return summary_path, csv_path
