# Parity Interpretability Notes

## Main finding

Mamba and Transformer have similar parity test accuracy, but their internal representations differ substantially.

Mamba forms a strong parity-like internal representation on short strings. However, this representation degrades with sequence length. Interestingly, Mamba's transition structure remains more coherent than its semantic alignment with true parity.

## Behavioral parity accuracy

- RNN: 1.000
- LSTM: 1.000
- Transformer: ~0.633
- Mamba: ~0.641

## Short-string representation results

- Mamba final-layer probe: ~0.999
- Mamba final-state probe: ~0.990
- RNN final-state probe: 1.000
- LSTM final-state probe: 1.000
- Transformer final-state probe: 0.680

## Long test-set representation results

- Mamba test final-state probe: ~0.696
- Transformer test final-state probe: ~0.580
- RNN test final-state probe: 1.000
- LSTM test final-state probe: 1.000

## Mamba length-binned results

| Position range | Probe accuracy | Transition consistency | Mean abs probe score |
|---|---:|---:|---:|
| 0-50 | 0.968 | 0.971 | 13.804 |
| 50-100 | 0.715 | 0.819 | 1.460 |
| 100-150 | 0.668 | 0.823 | 0.648 |
| 150-200 | 0.649 | 0.835 | 0.558 |
| 200-300 | 0.627 | 0.844 | 0.517 |
| 300-400 | 0.610 | 0.845 | 0.495 |
| 400-500 | 0.599 | 0.843 | 0.497 |

## Transformer length-binned results

| Position range | Probe accuracy | Transition consistency | Mean abs probe score |
|---|---:|---:|---:|
| 0-50 | 0.711 | 0.679 | 1.377 |
| 50-100 | 0.562 | 0.540 | 0.253 |
| 100-150 | 0.553 | 0.538 | 0.246 |
| 150-200 | 0.547 | 0.534 | 0.302 |
| 200-300 | 0.540 | 0.537 | 0.241 |
| 300-400 | 0.535 | 0.533 | 0.244 |
| 400-500 | 0.531 | 0.527 | 0.270 |

## Interpretation

Transformer appears to have weak parity semantics and weak transition structure. Both approach chance on long strings.

Mamba initially has strong parity semantics and strong transition structure. On long strings, parity semantics degrade sharply, but transition consistency remains relatively high. This suggests that Mamba may preserve an automaton-like internal update structure even after the recovered state no longer aligns well with true parity.

## Current hypothesis

Mamba learns a coherent parity-like state machine on short strings. Under length extrapolation, the internal state continues to follow a mostly consistent transition rule, but the semantic alignment between recovered states and true parity drifts.

## Next experiments

1. Repeat this pipeline on even-pairs.
2. Repeat on repeat-01.
3. For Mamba, localize which layer first creates the parity state.
4. Track parity-carrying dimensions, especially dimensions 20, 35, and 18, as a function of position.
5. Instrument Mamba internals to identify which state-space update components carry or degrade the parity signal.
