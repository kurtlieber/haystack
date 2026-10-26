#!/usr/bin/env bash
# Symlink the repo's units into ~/.config/systemd/user and reload. Does NOT enable anything.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
dest="$HOME/.config/systemd/user"
mkdir -p "$dest"
for f in "$repo"/systemd/*.service "$repo"/systemd/*.timer; do
  ln -sfn "$f" "$dest/$(basename "$f")"
done
systemctl --user daemon-reload
echo "linked: $(ls "$repo"/systemd | tr '\n' ' ')"
echo "token-free jobs:   systemctl --user enable --now haystack-library.timer haystack-harvest.timer \\"
echo "                   haystack-harvest-broad.timer"
echo "after auth:        systemctl --user enable --now haystack-sync.timer haystack-plays.timer haystack-batch.timer"
