#!/usr/bin/env bash
set -euo pipefail

# Canonical environment loader for this repo on NERSC-like systems.
# Usage: source ./env.sh
module load pytorch/2.6.0-1

# ---- ORCA configuration (optional) ------------------------------------------
# Set one of these before sourcing, or edit defaults here:
#   export ORCA_HOME="/path/to/orca"
#   export ORCA_BIN="/path/to/orca/orca"
#
# If ORCA_HOME is set and ORCA_BIN is not, assume $ORCA_HOME/orca.
if [[ -n "${ORCA_HOME:-}" && -z "${ORCA_BIN:-}" ]]; then
  export ORCA_BIN="$ORCA_HOME/orca"
fi

# If ORCA_BIN points to an existing file, add its directory to PATH
# so calls to `orca` work transparently.
if [[ -n "${ORCA_BIN:-}" && -f "${ORCA_BIN}" ]]; then
  export PATH="$(dirname "${ORCA_BIN}"):${PATH}"
fi


