#!/usr/bin/env bash
# Point DA-SAM3 to the bundled Medical-SAM3 SAM3 source tree.
export SAM3_ROOT="$(cd "$(dirname "$0")/../Medical-SAM3/Medical-SAM3-main" && pwd)"
export PYTHONPATH="${SAM3_ROOT}:${SAM3_ROOT}/sam3:${PYTHONPATH}"
echo "SAM3_ROOT=${SAM3_ROOT}"
