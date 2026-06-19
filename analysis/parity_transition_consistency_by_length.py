import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = torch.load(
    "analysis_outputs/parity_mamba_test_hidden_states.pt"
)

X = data["X"].numpy()
y = data["y"].numpy()
positions = data["positions"].numpy()

seqs = torch.load(
    "languages/parity/datasets/test/main.prepared"
)

# --------------------------------------------------
# train decoder
# --------------------------------------------------

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

clf = LogisticRegression(max_iter=3000)
clf.fit(Xs, y)

decoded = clf.predict(Xs)

# --------------------------------------------------
# buckets
# --------------------------------------------------

buckets = [
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 300),
    (300, 400),
    (400, 500),
]

results = []

# --------------------------------------------------
# collect transitions
# --------------------------------------------------

offset = 0

for lo, hi in buckets:

    transitions = defaultdict(Counter)

    offset = 0

    for seq in seqs:

        n = len(seq)

        if n < 2:
            offset += n
            continue

        states = decoded[offset:offset+n]
        pos = positions[offset:offset+n]

        for t in range(n - 1):

            if not (lo <= pos[t] < hi):
                continue

            curr_state = int(states[t])
            token = int(seq[t + 1])
            next_state = int(states[t + 1])

            transitions[(curr_state, token)][next_state] += 1

        offset += n

    consistencies = []

    for key in transitions:

        counts = transitions[key]

        total = sum(counts.values())

        majority = counts.most_common(1)[0][1]

        consistencies.append(majority / total)

    results.append(
        (
            f"{lo}-{hi}",
            np.mean(consistencies),
            np.min(consistencies)
        )
    )

print()
print("range        avg_consistency   min_consistency")
print("-" * 55)

for r, avgc, minc in results:
    print(
        f"{r:10s}   {avgc:.4f}           {minc:.4f}"
    )
