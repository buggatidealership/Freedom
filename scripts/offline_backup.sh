#!/usr/bin/env bash
# Save the whole repository, including commits GitHub has not accepted yet, as a
# self-contained git bundle that can be restored with `git fetch <bundle>` or
# `git clone <bundle>`. Used while the GitHub account is suspended and pushes fail.
#
#   scripts/offline_backup.sh [OUT_DIR]
#
# Prints the bundle path and the commits not yet on the remote branch.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
cd "$repo"
branch=$(git rev-parse --abbrev-ref HEAD)
out_dir=${1:-"${FREEDOM_BACKUP_DIR:-$repo/reports/backups}"}
mkdir -p "$out_dir"

stamp=$(date -u +%Y-%m-%dT%H%MZ)
head=$(git rev-parse --short HEAD)
bundle="$out_dir/freedom-$stamp-$head.bundle"

git bundle create "$bundle" --all >/dev/null
git bundle verify "$bundle" >/dev/null

echo "bundle: $bundle ($(du -h "$bundle" | cut -f1))"
if git rev-parse --verify -q "origin/$branch" >/dev/null; then
    n=$(git rev-list --count "origin/$branch..HEAD")
    echo "unpushed commits on $branch: $n"
    git log --oneline "origin/$branch..HEAD" | sed 's/^/  /'
else
    echo "remote branch origin/$branch unknown locally; bundle carries everything"
fi
