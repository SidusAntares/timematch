# Source Structure Research Baseline

The active research baseline is original TimeMatch with an optional
source-stage raw-global compactness loss.

## Active modes

```text
off
raw_global
```

`off` dispatches directly to the unchanged supervised source-training loop.

Raw-global acts on the raw PSE output `H [B,T,D]`, before LTAE. It mean-pools
`H` over time, computes a center for each source class represented by at least
two batch samples, and averages the class compactness losses with equal class
weight. The loss uses source labels only and is never called during TimeMatch
domain adaptation.

The classification path remains:

```text
H -> LTAE -> classifier
```

Smooth/timepoint, elastic, UMSC, DCT, routing, and selector implementations
are historical experiments. They are not exposed by the active CLI or the
launchers in `launchers/rawglobal/`.

Future position-aware temporal decomposition may be inserted at the boundary
where both `H` and `positions` are available. No decomposition is implemented
on this branch.
