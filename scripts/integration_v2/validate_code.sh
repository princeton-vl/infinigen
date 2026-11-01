#!/usr/bin/env bash
# Code-level CI checks for infinigen2. Run from the repo root.
set -euo pipefail

# Commits from dead/force-overwritten lineages that must never be merged.
BANNED=$(cat <<'EOF'
a81a0eb9254e4d0125f72aa507ae6f8d19492a16
d7bf1f6ccbf011b5b8590cec6b281ec55a9f29f9
7148e0bad3da2eddce21bda378bef73270f9a7a7
c0e2a0c1496cd753ca48d7b66609252d79fe3e7c
6601e9b7a4f31bccbf6c79db1c634f7be5288641
63dee4d441fcbb8db9af12739a43a62a443e13c1
7b515778cdd359df63ead6a8f651d9af2bd73422
cfab1c645c4eee31ddb26e808eeace8421fe8822
ec9c1b1076366456de6894307a84d2df1cbf1522
EOF
)

history=$(git rev-list HEAD)
failed=0
while read -r sha; do
  [ -n "$sha" ] || continue
  if grep -qx "$sha" <<< "$history"; then
    echo "::error::Banned commit $sha is in this PR's history"
    failed=1
  fi
done <<< "$BANNED"

if [ "$failed" -ne 0 ]; then
  echo "This branch contains a banned commit. Rebase it onto the current develop and drop the stale lineage."
  exit 1
fi
echo "No banned commits found in history."
