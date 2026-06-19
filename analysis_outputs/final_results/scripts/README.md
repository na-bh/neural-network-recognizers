# Saved Analysis Scripts

## extract_dfa.py

Purpose:
Extract a symbolic automaton from Mamba hidden states.

Outputs:
- decoded states
- transition table
- transition consistency
- symbolic DFA

Used for:
- even-pairs
- repeat-01
- parity

Main finding:
Even-pairs and repeat-01 produce nearly perfect symbolic automata.
Parity produces a partially recoverable automaton.


## parity_transition_consistency_by_length.py

Purpose:
Measure parity transition consistency as a function of position.

Outputs:
- average transition consistency by position bucket
- minimum transition consistency by position bucket

Main finding:

State recovery:
0.9685 -> 0.5993

Transition consistency:
0.9716 -> 0.8446

Interpretation:
Transition dynamics degrade much more slowly than state recoverability.
The parity update rule remains partially intact even when parity state decoding becomes difficult.
