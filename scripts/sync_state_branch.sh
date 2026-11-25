#!/usr/bin/env bash
# Sync an isolated Git orphan branch (e.g. delivery-state) into a local worktree directory.
# Usage:
#   scripts/sync_state_branch.sh pull <branch> <state_dir>
#   scripts/sync_state_branch.sh push <branch> <state_dir> <commit_message>

set -euo pipefail

ACTION="${1:-}"
BRANCH="${2:-delivery-state}"
STATE_DIR="${3:-.delivery-state}"
COMMIT_MSG="${4:-chore(state): update delivery state [skip ci]}"

if [ -z "$ACTION" ]; then
  echo "Usage: $0 <pull|push> <branch> <state_dir> [commit_message]" >&2
  exit 2
fi

case "$ACTION" in
  pull)
    if [ -d "$STATE_DIR" ]; then
      git worktree remove --force "$STATE_DIR" >/dev/null 2>&1 || rm -rf "$STATE_DIR"
    fi
    git worktree prune >/dev/null 2>&1 || true

    if git fetch origin "$BRANCH" --depth=20 >/dev/null 2>&1; then
      git worktree add -B "$BRANCH" "$STATE_DIR" "origin/$BRANCH"
    else
      empty_tree="$(git hash-object -t tree /dev/null)"
      init_commit="$(
        GIT_AUTHOR_NAME="ai-daily-bot" \
        GIT_AUTHOR_EMAIL="ai-daily-bot@users.noreply.github.com" \
        GIT_COMMITTER_NAME="ai-daily-bot" \
        GIT_COMMITTER_EMAIL="ai-daily-bot@users.noreply.github.com" \
        git commit-tree "$empty_tree" -m "chore(state): initialize ${BRANCH} branch [skip ci]"
      )"
      git branch -f "$BRANCH" "$init_commit"
      git worktree add "$STATE_DIR" "$BRANCH"
    fi
    ;;

  push)
    if [ ! -d "$STATE_DIR" ]; then
      echo "State worktree directory not found: $STATE_DIR" >&2
      exit 1
    fi

    git -C "$STATE_DIR" config user.name "ai-daily-bot"
    git -C "$STATE_DIR" config user.email "ai-daily-bot@users.noreply.github.com"
    git -C "$STATE_DIR" add -A

    if git -C "$STATE_DIR" diff --cached --quiet; then
      echo "No delivery state changes to push on branch ${BRANCH}."
      exit 0
    fi

    git -C "$STATE_DIR" commit -m "$COMMIT_MSG"

    max_attempts=3
    attempt=1
    while [ "$attempt" -le "$max_attempts" ]; do
      if git -C "$STATE_DIR" push -u origin "$BRANCH"; then
        echo "Successfully pushed delivery state to branch ${BRANCH} (attempt ${attempt})."
        exit 0
      fi
      echo "Push to ${BRANCH} failed (attempt ${attempt}/${max_attempts}); attempting fetch + rebase..." >&2
      if git -C "$STATE_DIR" fetch origin "$BRANCH" >/dev/null 2>&1; then
        git -C "$STATE_DIR" rebase -X theirs "origin/$BRANCH" || git -C "$STATE_DIR" rebase --abort || true
      fi
      sleep $((attempt * 2))
      attempt=$((attempt + 1))
    done

    echo "ERROR: Failed to push delivery state to branch ${BRANCH} after ${max_attempts} attempts." >&2
    exit 1
    ;;

  *)
    echo "Unknown action: $ACTION (expected pull or push)" >&2
    exit 2
    ;;
esac
