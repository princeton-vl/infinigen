#!/usr/bin/env bash
# Code-level CI checks for infinigen2. Run from the repo root.
set -euo pipefail

# Commits from dead/force-overwritten lineages that must never be merged.
# One "<full-sha> <reason>" per line.
BANNED=$(cat <<'EOF'
a81a0eb9254e4d0125f72aa507ae6f8d19492a16 develop_materials (do not branch off; rebase onto develop2)
d7bf1f6ccbf011b5b8590cec6b281ec55a9f29f9 pre-msg-rewrite develop2 (force-overwritten; rebase onto current develop2)
EOF
)

history=$(git rev-list HEAD)
failed=0
while read -r sha reason; do
  [ -n "$sha" ] || continue
  if grep -qx "$sha" <<< "$history"; then
    echo "::error::Banned commit $sha is in this PR's history: $reason"
    failed=1
  fi
done <<< "$BANNED"

if [ "$failed" -ne 0 ]; then
  echo "This branch contains a banned commit. Rebase it onto the current develop2 and drop the stale lineage."
  exit 1
fi
echo "No banned commits found in history."
