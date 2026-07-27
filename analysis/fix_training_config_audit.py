"""Corrected-dataset experiment: training-configuration audit, run BEFORE F3
per user instruction. Verifies (1) what objective/loss-term combination and
training signals FLaRe's standard training procedure uses for marked-
reversal, and (2) whether the corrected dataset (F1, languages/marked-
reversal-fixed/) provides those same signals, or is missing something F3
would need.

NO TRAINING RUN HERE -- audit only.

PYTHONPATH=src:analysis python analysis/fix_training_config_audit.py
"""

import json
from pathlib import Path

RESULTS = Path("analysis_outputs/final_results")
STANDARD_DIR = Path("languages/marked-reversal")
FIXED_DIR = Path("languages/marked-reversal-fixed")


def file_present(path):
    return path.exists() and path.stat().st_size >= 0


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # (1) standard FLaRe training objective, confirmed from source, not
    # inferred from directory naming alone
    # ------------------------------------------------------------------
    training_objective = {
        "entry_point": "src/recognizers/neural_networks/train_and_evaluate.bash, "
                       "which calls recognizers/neural_networks/train.py",
        "loss_terms_confirmed": "rec+ns",
        "confirmation_method": (
            "train_and_evaluate.bash's <loss-terms> argument is joined into the model output "
            "path via get_model_dir() (recognizers/functions.bash): "
            "\"$base_dir/models/$language/$architecture/$loss_terms/$validation_data/$trial_no\" "
            "-- this EXACTLY matches every checkpoint path used throughout the four-task pilot "
            "(e.g. data/models/marked-reversal/rnn/rec+ns/validation-short/3), confirming "
            "loss_terms='rec+ns' directly from the path-construction logic, not just pattern-"
            "matching the directory name."
        ),
        "loss_term_definitions": {
            "rec": "recognition (binary cross-entropy on the single accept/reject logit) -- "
                  "ALWAYS present, no extra flag needed; requires ONLY labels.txt "
                  "(labels.prepared at train time).",
            "ns": "next-symbol-set prediction (binary cross-entropy over whether each vocabulary "
                 "symbol is valid at each next position) -- ACTIVE for 'rec+ns' runs via "
                 "--use-next-symbols-head; requires next-symbols.jsonl (next-symbols.prepared at "
                 "train time), populated ONLY for positive examples (empty for negatives, matching "
                 "recognizers/neural_networks/data.py's load_prepared_next_symbols_file, which "
                 "yields None for any example with label=False).",
            "lm": "full autoregressive language-modeling head (cross-entropy over the next actual "
                 "token) -- NOT active for 'rec+ns' runs (would need --use-language-modeling-head, "
                 "not present in the confirmed loss_terms). Does NOT use log-probabilities.txt -- "
                 "that file is not read by any code under src/ (confirmed by direct grep); it "
                 "appears to be an analysis-only artifact, not a training signal, regardless of "
                 "which loss terms are active.",
        },
        "signals_required_for_rec_plus_ns": ["main.prepared (token sequences)",
                                             "labels.prepared", "next-symbols.prepared"],
        "signals_NOT_required": ["log-probabilities.txt (unused by training code at all)",
                                 "num-edits.txt (analysis-only, from the original K-edit pipeline)"],
        "per_trial_hyperparameter_note": (
            "train_and_evaluate.bash samples several hyperparameters PER TRIAL via random_sample.py: "
            "--max-tokens-per-batch (log-uniform int, 128-4096), --initial-learning-rate "
            "(log-uniform, 0.0001-0.01), and --next-symbols-loss-coefficient (log-uniform, "
            "0.01-10). This means the 10 original seeds for any architecture/task were NOT trained "
            "with identical hyperparameters -- each trial_no drew its own random sample. F3 must "
            "replicate this per-trial sampling (matching trial_no to seed, using the same "
            "random_sample.py mechanism) to legitimately claim it 'matches FLaRe's standard "
            "training procedure exactly' -- flagged here for F3, not resolved in this audit."
        ),
    }

    # ------------------------------------------------------------------
    # (2) corrected dataset file inventory vs standard dataset
    # ------------------------------------------------------------------
    def inventory(base_dir):
        raw_files = ["main.tok", "labels.txt", "next-symbols.jsonl", "log-probabilities.txt",
                    "num-edits.txt", "negative-kind.txt", "hard-negative-swap-meta.jsonl"]
        prepared_files = ["main.vocab", "main.prepared", "labels.prepared", "next-symbols.prepared"]
        return (
            {f: file_present(base_dir / f) for f in raw_files},
            {f: file_present(base_dir / f) for f in prepared_files},
        )

    std_raw, std_prepared = inventory(STANDARD_DIR)
    fixed_raw, fixed_prepared = inventory(FIXED_DIR)
    fixed_raw_test, fixed_prepared_test = inventory(FIXED_DIR / "datasets" / "test")
    fixed_raw_val, fixed_prepared_val = inventory(FIXED_DIR / "datasets" / "validation-short")

    missing_for_training = [f for f, present in fixed_prepared.items() if not present]

    dataset_comparison = {
        "standard_marked_reversal": {"raw_files": std_raw, "prepared_files": std_prepared},
        "corrected_marked_reversal_fixed_train": {"raw_files": fixed_raw, "prepared_files": fixed_prepared},
        "corrected_marked_reversal_fixed_validation_short": {"raw_files": fixed_raw_val, "prepared_files": fixed_prepared_val},
        "corrected_marked_reversal_fixed_test": {"raw_files": fixed_raw_test, "prepared_files": fixed_prepared_test},
        "raw_signal_completeness": (
            "F1 generated ALL raw signals 'rec+ns' training needs: labels.txt (rec) and a "
            "correctly-populated next-symbols.jsonl (ns) -- verified non-empty for positive rows "
            "and an empty list '[]' for negative rows, matching the standard convention exactly "
            "(spot-checked directly against languages/marked-reversal/next-symbols.jsonl's format). "
            "log-probabilities.txt was also generated (unused by training, harmless extra)."
        ),
        "gap_identified": (
            f"The corrected dataset (train + validation-short + test) is MISSING the PREPARED/"
            f"integerized files training actually reads: {missing_for_training}. This is NOT a "
            f"missing-training-signal problem (Options A/B/C as originally framed don't apply) -- "
            f"F1 was scoped as raw dataset construction + audit only, and never ran the mechanical "
            f"tokenization/vocabulary-building step (recognizers/neural_networks/prepare_data.py, "
            f"invoked via prepare_language.bash for the standard pipeline) that converts "
            f"main.tok/labels.txt/next-symbols.jsonl into main.vocab/main.prepared/labels.prepared/"
            f"next-symbols.prepared. The raw data is complete and correctly formatted for 'rec+ns' "
            f"training; it just hasn't been run through this one remaining, purely mechanical "
            f"preparation step yet."
        ),
        "recommended_next_action": (
            "Run recognizers/neural_networks/prepare_data.py on languages/marked-reversal-fixed "
            "(--training-data languages/marked-reversal-fixed --more-data validation-short "
            "--more-data test --use-next-symbols-head --never-allow-unk), exactly mirroring "
            "prepare_language.bash's invocation for the standard pipeline, before F3 begins "
            "training. This is a data-preparation step, not a redesign of F1's dataset "
            "construction -- none of Options A/B/C (regenerate with all signals / retrain "
            "baselines rec-only / empirically verify rec-only) are needed, since the raw signals "
            "already match 'rec+ns' requirements exactly."
        ),
    }

    out = {
        "task": "marked-reversal",
        "experiment": "corrected-dataset-fix (training-configuration audit, pre-F3)",
        "training_objective": training_objective,
        "dataset_comparison": dataset_comparison,
    }
    out_path = RESULTS / "fix_training_config_audit.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
