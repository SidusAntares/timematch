# v3.2.1 Local Shift Utilities

This module contains experimental v3.2.1 local-shift utilities.

Stage 3d archived the residual local-shift method as a negative boundary experiment.  It should not be used as an active training method without a new proposal.

The utilities remain useful for diagnostics and reproducibility:

```text
source_reference.py
target_partition.py
soft_alignment.py
local_position.py
train_local_shift.py
```

Do not confuse this module with v3.1 stage contrast.  Stage contrast is not active in the cleaned parser path.

