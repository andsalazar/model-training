#!/bin/bash
# Install the imports Ovi actually needs but does not declare.
#
# Ovi's requirements.txt does not list einops, even though ovi/modules/vae.py imports
# it at module scope, so `from ovi.modules import ...` fails on a clean install. Rather
# than guess the rest, import the top-level package and fix each failure in turn.
set -uo pipefail

source /local/ovi-env/bin/activate
export PYTHONPATH="/local/Ovi:${PYTHONPATH:-}"

pip install --no-cache-dir einops > /tmp/einops.log 2>&1
echo "einops install exit: $?"
tail -2 /tmp/einops.log

echo "=== import attempts ==="
for i in 1 2 3 4 5 6 7 8; do
  OUT=$(python3 -c "import ovi.ovi_fusion_engine; print('IMPORT OK')" 2>&1 | tail -3)
  echo "attempt $i: $OUT"
  MODULE=$(printf '%s' "$OUT" | grep -oP "No module named '\K[^'.]+" | head -1)
  if [ -z "$MODULE" ]; then
    break
  fi
  echo "  -> installing $MODULE"
  pip install --no-cache-dir "$MODULE" > /tmp/mod_install.log 2>&1 || true
done
