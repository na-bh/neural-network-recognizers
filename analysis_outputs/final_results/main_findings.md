# Main Findings

## DFA recovery

Even-pairs:
- State recovery: 1.000
- Transition consistency: 0.98+

Repeat-01:
- State recovery: 0.999
- Transition consistency: 0.98+

## Parity

State recovery by position:
0-50: 0.968
400-500: 0.599

Transition consistency:
0-50: 0.971
400-500: 0.843

## Layer analysis

Layer 4 signal:
0-50: 13.80
400-500: 0.50

Observation:
Parity representation collapses with sequence length.

# Mamba Interpretability Results (June 18, 2026)

## Research Question

Why does Mamba outperform Transformers on formal language recognition tasks?

Specifically:

1. Does Mamba learn automaton-like internal states?
2. How do these representations compare to Transformers?
3. Why does Mamba succeed on some languages while struggling on parity?

---

# 1. DFA State Recovery

## Even-Pairs

State recovery remains perfect across all position ranges:

| Range   | Recovery |
| ------- | -------: |
| 0-50    |   1.0000 |
| 50-100  |   1.0000 |
| 100-150 |   1.0000 |
| 150-200 |   1.0000 |
| 200-300 |   1.0000 |
| 300-400 |   1.0000 |
| 400-500 |   1.0000 |

Transition consistency:

| Range   | Consistency |
| ------- | ----------: |
| 0-50    |      0.9978 |
| 50-100  |      0.9977 |
| 100-150 |      0.9974 |
| 150-200 |      0.9970 |
| 200-300 |      0.9961 |
| 300-400 |      0.9933 |
| 400-500 |      0.9801 |

**Observation:** Mamba learns an almost perfect internal representation of the even-pairs DFA and maintains it across long sequence lengths.

---

## Repeat-01

State recovery:

| Range   | Recovery |
| ------- | -------: |
| 0-50    |   0.9980 |
| 50-100  |   0.9983 |
| 100-150 |   0.9983 |
| 150-200 |   0.9987 |
| 200-300 |   0.9986 |
| 300-400 |   0.9991 |
| 400-500 |   0.9990 |

Transition consistency:

| Range   | Consistency |
| ------- | ----------: |
| 0-50    |      0.9944 |
| 50-100  |      0.9945 |
| 100-150 |      0.9946 |
| 150-200 |      0.9946 |
| 200-300 |      0.9937 |
| 300-400 |      0.9916 |
| 400-500 |      0.9785 |

**Observation:** Mamba also learns a nearly perfect representation of the Repeat-01 automaton.

---

# 2. Parity Failure Analysis

## State Recovery

| Range   | Recovery |
| ------- | -------: |
| 0-50    |   0.9685 |
| 50-100  |   0.7152 |
| 100-150 |   0.6675 |
| 150-200 |   0.6488 |
| 200-300 |   0.6267 |
| 300-400 |   0.6099 |
| 400-500 |   0.5993 |

## Transition Consistency

| Range   | Consistency |
| ------- | ----------: |
| 0-50    |       0.971 |
| 50-100  |       0.819 |
| 100-150 |       0.823 |
| 150-200 |       0.835 |
| 200-300 |       0.844 |
| 300-400 |       0.845 |
| 400-500 |       0.843 |

**Observation:** Parity semantics degrade substantially with length, but transition structure remains partially intact. This suggests that Mamba continues to perform a parity-related computation even when parity labels become difficult to recover.

---

# 3. Layer Analysis

## Parity

State recovery by layer (positions 0-50):

| Layer | Recovery |
| ----- | -------: |
| 0     |    0.615 |
| 1     |    0.712 |
| 2     |    0.822 |
| 3     |    0.914 |
| 4     |    0.969 |

**Observation:** Parity is progressively constructed through depth. Each layer strengthens the parity representation.

---

## Even-Pairs

State recovery by layer (positions 0-50):

| Layer | Recovery |
| ----- | -------: |
| 0     |    0.612 |
| 1     |    0.626 |
| 2     |    0.631 |
| 3     |    0.639 |
| 4     |    1.000 |

**Observation:** Unlike parity, the DFA state is not linearly recoverable until the final layer. The state appears almost entirely in the last layer.

---

# 4. Signal Strength Collapse

Parity probe signal magnitude:

| Range   | Mean Absolute Score |
| ------- | ------------------: |
| 0-50    |              13.804 |
| 50-100  |               1.460 |
| 100-150 |               0.648 |
| 150-200 |               0.558 |
| 200-300 |               0.517 |
| 300-400 |               0.495 |
| 400-500 |               0.497 |

**Observation:** The parity representation becomes dramatically weaker with increasing sequence length.

---

# 5. Localization of the Parity Representation

Most important dimensions identified by causal ablation:

```
[0, 28, 36, 6, 32, 9, 27, 30, 4, 13]
```

## Ablation

| Condition                | Accuracy |
| ------------------------ | -------: |
| Full representation      |    0.713 |
| Top-5 dimensions removed |    0.462 |

Removing a small set of dimensions destroys most parity information.

---

## Top-k Only Recovery

| Dimensions Kept     | Accuracy |
| ------------------- | -------: |
| Top-1               |    0.623 |
| Top-2               |    0.629 |
| Top-3               |    0.683 |
| Top-5               |    0.691 |
| Top-10              |    0.703 |
| Full representation |    0.713 |

**Observation:** Ten dimensions recover 98.6% of full parity probe performance. Parity is concentrated in a low-dimensional subspace.

---

# 6. Geometry

## Parity

Findings:

* Hidden states form a structured parity subspace.
* The representation drifts with sequence length.
* Long sequences collapse toward a central region.
* Recovery decreases as drift increases.

## Even-Pairs

Findings:

* Four clear DFA-state clusters emerge.
* Clusters remain stable across sequence lengths.
* State recovery remains perfect.

**Observation:** Stable latent geometry corresponds to stable automaton recovery.

---

# 7. Transformer Comparison

Parity probe accuracy:

| Model       | Accuracy |
| ----------- | -------: |
| Transformer |    0.578 |
| Mamba       |    0.713 |

Transformer top-k recovery:

| Dimensions Kept     | Accuracy |
| ------------------- | -------: |
| Top-1               |    0.551 |
| Top-2               |    0.551 |
| Top-3               |    0.562 |
| Top-5               |    0.571 |
| Top-10              |    0.575 |
| Full representation |    0.578 |

**Observation:** Transformer also contains a parity subspace, but it is substantially weaker than Mamba's. Transformer geometry is more diffuse and less separable.

---

# Current Hypothesis

Mamba learns compact automaton-like latent representations for formal languages.

For even-pairs and repeat-01, these representations correspond closely to DFA states and remain stable across long sequences.

For parity, Mamba develops a localized parity subspace, but the representation drifts with sequence length, causing parity recovery to collapse.

Compared to Transformers, Mamba learns stronger, more structured, and more recoverable internal representations of automaton state.

---

# Next Steps

1. Verify results across additional random seeds.
2. Perform stronger causal interventions on parity dimensions.
3. Analyze Transformer geometry in greater depth.
4. Investigate why parity exhibits drift while even-pairs remains stable.
5. Develop a circuit-level interpretation of the parity subspace.
