#!/usr/bin/env bash
# Code-level CI checks for infinigen2. Run from the repo root.
set -euo pipefail

# Commits from dead/force-overwritten lineages that must never be merged.
BANNED_FILE="$(dirname "$0")/banned_commits.txt"

history=$(git rev-list HEAD)
failed=0
while read -r sha; do
  [ -n "$sha" ] || continue
  if grep -qx "$sha" <<< "$history"; then
    echo "::error::Banned commit $sha is in this PR's history"
    failed=1
  fi
done < "$BANNED_FILE"

if [ "$failed" -ne 0 ]; then
  echo "This branch contains a banned commit. Rebase it onto the current develop and drop the stale lineage."
  exit 1
fi
echo "No banned commits found in history."
