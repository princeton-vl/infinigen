#!/usr/bin/env bash
# Code-level CI checks for infinigen2. Run from the repo root.
set -euo pipefail

# Commits from dead/force-overwritten lineages that must never be merged.
# One "<full-sha> <reason>" per line.
BANNED=$(cat <<'EOF'
a81a0eb9254e4d0125f72aa507ae6f8d19492a16 develop_materials (do not branch off; rebase onto develop2)
d7bf1f6ccbf011b5b8590cec6b281ec55a9f29f9 pre-msg-rewrite develop2 (force-overwritten; rebase onto current develop2)
7148e0bad3da2eddce21bda378bef73270f9a7a7 Claude co-author trailer on old main lineage (dropped; rebase onto current develop2)
c0e2a0c1496cd753ca48d7b66609252d79fe3e7c stale integration-render gate (superseded by squashed 29facb4e4 on develop; rebase to drop)
6601e9b7a4f31bccbf6c79db1c634f7be5288641 stale gate coverage-flush follow-up (superseded on develop; rebase to drop)
63dee4d441fcbb8db9af12739a43a62a443e13c1 unapproved Blender add-on bindings pushed directly to develop (removed by history rewrite; rebase to drop)
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
