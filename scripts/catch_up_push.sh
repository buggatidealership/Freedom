#!/usr/bin/env bash
# Push everything GitHub missed while the account was suspended.
#
#   scripts/catch_up_push.sh [BUNDLE]
#
# 1. If BUNDLE is given (a file made by scripts/offline_backup.sh), fetch the
#    branch from it first and fast-forward the local branch to it. This is the
#    path for a fresh checkout that no longer holds the local-only commits.
# 2. Push the branch with `git push -u origin <branch>`, retrying network
#    failures with exponential backoff (2, 4, 8, 16 s). A 403 means the account
#    is still suspended: the script says so and exits 2 without retrying.
#
# Never force-pushes and never rewrites history.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
cd "$repo"
branch=$(git rev-parse --abbrev-ref HEAD)
bundle=${1:-}

if [[ -n "$bundle" ]]; then
    git bundle verify "$bundle" >/dev/null
    git fetch "$bundle" "refs/heads/$branch:refs/bundles/$branch"
    if git merge-base --is-ancestor HEAD "refs/bundles/$branch"; then
        git merge --ff-only "refs/bundles/$branch"
        echo "fast-forwarded $branch to the bundle's $(git rev-parse --short HEAD)"
    elif git merge-base --is-ancestor "refs/bundles/$branch" HEAD; then
        echo "local $branch already contains the bundle"
    else
        echo "local $branch and the bundle have diverged; resolve by hand" >&2
        exit 3
    fi
fi

git fetch origin "$branch" >/dev/null 2>&1 || true
if git rev-parse --verify -q "origin/$branch" >/dev/null; then
    echo "commits to push:"
    git log --oneline "origin/$branch..HEAD" | sed 's/^/  /'
fi

delay=2
for attempt in 1 2 3 4 5; do
    if out=$(git push -u origin "$branch" 2>&1); then
        echo "$out"
        echo "pushed $branch"
        exit 0
    fi
    if grep -q "403" <<<"$out"; then
        echo "$out" | tail -2
        echo "GitHub still refuses writes (403): the account suspension has not been lifted."
        exit 2
    fi
    echo "push attempt $attempt failed:" >&2
    echo "$out" | tail -2 >&2
    [[ $attempt -lt 5 ]] && sleep "$delay" && delay=$((delay * 2))
done
echo "push failed after 5 attempts" >&2
exit 1
