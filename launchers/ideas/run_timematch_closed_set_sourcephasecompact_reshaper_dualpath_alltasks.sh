#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for PART in A B; do
  PARTITION="$PART" bash "$SCRIPT_DIR/run_timematch_closed_set_sourcephasecompact_reshaper_dualpath_partition.sh"
done
