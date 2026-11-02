"""Run a separate official-TabM development experiment, never the primary CDSS."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.nhanes_tabm_evaluation import (
    OUTPUT_DIR, SEARCH, TabMConfig, load_development_data, outer_splits,
    training_partition, fit_tabm, predict_probability, score_probability, run_experiment,
    finalize_tabm_report,
)
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--compact-search', action='store_true',
                        help='Use only predeclared compact configs, excluding the timed official default')
    args = parser.parse_args()
    if args.smoke:
        frame = load_development_data()
        train, test = outer_splits(frame.diabetes)[0]
        fit, stop = training_partition(train, frame.diabetes.to_numpy(), 143)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rows = []
        for config in SEARCH[1:] if args.compact_search else [SEARCH[0], SEARCH[1]]:
            model, prep, diagnostics, history = fit_tabm(
                frame.iloc[fit], frame.diabetes.iloc[fit], config, device=device,
                max_epochs=3, patience=2, threads=args.threads,
                stopping_frame=frame.iloc[stop], stopping_target=frame.diabetes.iloc[stop])
            probability = predict_probability(model, prep, frame.iloc[stop], device)
            rows.append(dict(device=device, **diagnostics,
                             metrics=score_probability(frame.diabetes.iloc[stop], probability)))
            print(rows[-1], flush=True)
        (OUTPUT_DIR/'tabm_smoke.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    else:
        run_experiment(search=SEARCH[1:] if args.compact_search else SEARCH,
                        max_epochs=args.max_epochs, patience=args.patience, threads=args.threads)
        finalize_tabm_report()


if __name__ == '__main__':
    main()
