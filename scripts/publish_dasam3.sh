#!/usr/bin/env bash
# Create clean single-commit history and push to Reconsider80/DASAM3.
# Usage: bash scripts/publish_dasam3.sh <GITHUB_TOKEN>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
TOKEN="${1:-}"

if [[ -z "$TOKEN" ]]; then
  echo "Usage: bash scripts/publish_dasam3.sh <GITHUB_TOKEN>"
  exit 1
fi

export GIT_AUTHOR_NAME="Reconsider80"
export GIT_AUTHOR_EMAIL="chenying980812@gmail.com"
export GIT_COMMITTER_NAME="Reconsider80"
export GIT_COMMITTER_EMAIL="chenying980812@gmail.com"

git config http.version HTTP/1.1
git config http.postBuffer 524288000

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  TREE=$(git write-tree)
  COMMIT=$(printf '%s\n' \
    'Initial release of DA-SAM3 implementation.' \
    '' \
    'Dual-Adaptive MoE (DER + DPE) fusion layers with training, inference,' \
    'and validation scripts for parameter-efficient medical image segmentation.' \
    | git commit-tree "$TREE")
  git reset --hard "$COMMIT"
fi

if git log -1 --format=%B | grep -qi cursor; then
  echo "ERROR: commit message contains cursor reference."
  exit 1
fi

AUTH="Authorization: token ${TOKEN}"
REPO="Reconsider80/DASAM3"

if ! curl -sf -H "$AUTH" "https://api.github.com/repos/${REPO}" >/dev/null; then
  echo "Creating repository ${REPO}..."
  curl -sf -X POST -H "$AUTH" -H "Accept: application/vnd.github+json" \
    "https://api.github.com/user/repos" \
    -d '{"name":"DASAM3","description":"Dual-Adaptive SAM3 for parameter-efficient medical image segmentation","private":false}' \
    >/dev/null
fi

REMOTE="https://x-access-token:${TOKEN}@github.com/${REPO}.git"
git remote remove dasam3 2>/dev/null || true
git remote add dasam3 "$REMOTE"

echo "Pushing to https://github.com/${REPO}"
GIT_TERMINAL_PROMPT=0 git push --force dasam3 main

echo "Done: https://github.com/${REPO}"
