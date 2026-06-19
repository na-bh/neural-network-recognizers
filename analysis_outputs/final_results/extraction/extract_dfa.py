import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


CONFIGS = {
    "even-pairs": {
        "hidden_file": "analysis_outputs/even_pairs_mamba_test_hidden_states.pt",
        "seq_file": "languages/even-pairs/datasets/test/main.prepared",
    },
    "repeat-01": {
        "hidden_file": "analysis_outputs/repeat01_mamba_test_hidden_states.pt",
        "seq_file": "languages/repeat-01/datasets/test/main.prepared",
    },
    "parity": {
        "hidden_file": "analysis_outputs/parity_mamba_test_hidden_states.pt",
        "seq_file": "languages/parity/datasets/test/main.prepared",
    },
}


def load_data(language):
    cfg = CONFIGS[language]
    data = torch.load(cfg["hidden_file"])
    seqs = torch.load(cfg["seq_file"])

    X = data["X"].numpy()
    y = data["y"].numpy()

    return X, y, seqs


def train_decoder(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=3000)
    clf.fit(Xs, y)

    decoded = clf.predict(Xs)
    recovery = np.mean(decoded == y)

    return decoded, recovery


def extract_transitions(decoded, seqs):
    transitions = defaultdict(Counter)

    offset = 0

    for seq in seqs:
        n = len(seq)

        if n < 2:
            offset += n
            continue

        states = decoded[offset:offset + n]

        for t in range(n - 1):
            curr_state = int(states[t])
            next_token = int(seq[t + 1])
            next_state = int(states[t + 1])

            transitions[(curr_state, next_token)][next_state] += 1

        offset += n

    return transitions


def summarize_transitions(transitions):
    rows = []

    for key in sorted(transitions):
        curr_state, token = key
        counts = transitions[key]
        total = sum(counts.values())
        majority_next, majority_count = counts.most_common(1)[0]

        rows.append({
            "curr_state": int(curr_state),
            "token": int(token),
            "next_state": int(majority_next),
            "count": int(majority_count),
            "total": int(total),
            "consistency": majority_count / total,
            "full_counts": dict(counts),
        })

    return rows


def print_results(language, recovery, rows):
    print()
    print("=" * 80)
    print(f"EXTRACTED DFA: {language}")
    print("=" * 80)

    states = sorted(set([r["curr_state"] for r in rows] + [r["next_state"] for r in rows]))

    print(f"Decoded states: {states}")
    print(f"Number of decoded states: {len(states)}")
    print(f"State recovery: {recovery:.4f}")
    print()

    print("TRANSITION TABLE")
    print("----------------")

    consistencies = []

    for r in rows:
        consistencies.append(r["consistency"])
        print(
            f"state {r['curr_state']} --{r['token']}--> "
            f"state {r['next_state']} "
            f"({r['count']}/{r['total']} = {r['consistency']:.4f})"
        )

    print()
    print("SUMMARY")
    print("-------")
    print(f"Average transition consistency: {np.mean(consistencies):.4f}")
    print(f"Minimum transition consistency: {np.min(consistencies):.4f}")

    print()
    print("FULL COUNTS")
    print("-----------")
    for r in rows:
        print((r["curr_state"], r["token"]), r["full_counts"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "language",
        choices=sorted(CONFIGS.keys()),
        help="Language to extract DFA for."
    )

    args = parser.parse_args()

    X, y, seqs = load_data(args.language)
    decoded, recovery = train_decoder(X, y)
    transitions = extract_transitions(decoded, seqs)
    rows = summarize_transitions(transitions)
    print_results(args.language, recovery, rows)


if __name__ == "__main__":
    main()
