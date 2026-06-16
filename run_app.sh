#!/usr/bin/env bash
# Launch Streamlit with the PitchTip conda env Python (not system streamlit).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "/opt/anaconda3/envs/PitchTip/bin/python" ]]; then
  PY="/opt/anaconda3/envs/PitchTip/bin/python"
else
  echo "PitchTip env not found. Run: conda env create -f environment.yml" >&2
  exit 1
fi

exec "$PY" -m streamlit run "$ROOT/app.py" "$@"
