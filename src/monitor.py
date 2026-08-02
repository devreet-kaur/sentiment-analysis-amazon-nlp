"""
src/monitor.py - MAI203 Amazon Fine Food Reviews
EvidentlyAI drift detection comparing reference data to inference logs.
Called by: dvc repro monitor  OR  python src/monitor.py

Generates: reports/drift/drift_report.html
"""

from pathlib import Path

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
from evidently.report import Report

PARAMS  = yaml.safe_load(open('params.yaml'))
MON_CFG = PARAMS['monitor']


def load_reference() -> pd.DataFrame:
    ref_path = Path(MON_CFG['reference_data_path'])
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference data not found at {ref_path}. Run: dvc repro prepare"
        )
    df = pd.read_csv(ref_path)
    df = df[['cleaned_text', 'sentiment']].dropna().rename(
        columns={'cleaned_text': 'text', 'sentiment': 'label'})
    df['text_length'] = df['text'].str.split().str.len()
    return df.sample(min(2000, len(df)), random_state=42).reset_index(drop=True)


def load_current() -> pd.DataFrame:
    cur_path = Path(MON_CFG['current_data_path'])
    if not cur_path.exists():
        print(f"No inference log at {cur_path}. Run the API and make some predictions first.")
        return pd.DataFrame(columns=['text_length', 'label', 'confidence'])
    df = pd.read_csv(cur_path)
    df = df[['text_length', 'label', 'confidence']].dropna()
    return df


def run_monitor():
    output_dir = Path(MON_CFG['report_output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reference data...")
    reference = load_reference()

    print("Loading current (inference) data...")
    current = load_current()

    if len(current) < 10:
        print(f"Only {len(current)} inference records found.")
        print("Make at least 10 API calls before running drift detection.")
        return

    print(f"Reference rows: {len(reference):,}")
    print(f"Current rows:   {len(current):,}")

    # Column mapping for Evidently
    column_mapping = ColumnMapping(
        target='label',
        numerical_features=['text_length'],
        categorical_features=['label'],
    )

    # Run drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[['text_length', 'label']],
        current_data=current[['text_length', 'label']],
        column_mapping=column_mapping
    )

    report_path = output_dir / 'drift_report.html'
    report.save_html(str(report_path))
    print(f"\nDrift report saved: {report_path}")
    print("Open in browser to review feature drift scores.")

    # Check drift threshold
    drift_result = report.as_dict()
    try:
        share_drifted = drift_result['metrics'][0]['result']['share_of_drifted_columns']
        threshold     = MON_CFG['drift_threshold']
        status        = "ALERT" if share_drifted > threshold else "OK"
        print(f"\nDrift share: {share_drifted:.2%}  (threshold: {threshold:.0%})  [{status}]")
    except (KeyError, IndexError):
        print("Could not extract drift share from report.")


if __name__ == '__main__':
    run_monitor()
