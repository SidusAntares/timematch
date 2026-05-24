#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

export STAMP
export RUN_TAG="${RUN_TAG:-v27b_elastic_blend_probe_${STAMP}}"
export SOURCE_EPOCHS="${SOURCE_EPOCHS:-20}"
export DA_EPOCHS="${DA_EPOCHS:-20}"

# v27a used radius 2/3 with full replacement. This conservative probe keeps the
# original hard compactness as the anchor and adds only a small elastic residual.
export VARIANTS="${VARIANTS:-original:False:0:0.10:1.00 elastic_r1_a025_t010:True:1:0.10:0.25 elastic_r1_a050_t010:True:1:0.10:0.50}"

echo "v2.7b conservative elastic-blend probe"
echo "RUN_TAG=${RUN_TAG}"
echo "SOURCE_EPOCHS=${SOURCE_EPOCHS} DA_EPOCHS=${DA_EPOCHS}"
echo "VARIANTS=${VARIANTS}"

exec bash "${SCRIPT_DIR}/run_v27a_elastic_structure_probe.sh"
