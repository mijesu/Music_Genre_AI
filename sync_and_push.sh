#!/bin/bash
# Sync documentation from SSD to GitHub repo and push

echo "=== Syncing Music_ReClass Documentation ==="

# Source and destination
SRC="/media/mijesu_970/SSD_Data/Kiro_Projects/Music_Reclass"
DEST="/home/mijesu_970/Music_ReClass/docs"

# Sync markdown files
echo "Syncing .md files..."
rsync -av --delete "$SRC"/*.md "$DEST/" 2>/dev/null

# Sync Reference folder
echo "Syncing Reference folder..."
rsync -av --delete "$SRC/Reference/" "$DEST/Reference/" 2>/dev/null

# Git operations
cd /home/mijesu_970/Music_ReClass

echo "Checking for changes..."
if [[ -n $(git status -s) ]]; then
    echo "Changes detected. Committing..."
    git add -A
    git commit -m "Auto-sync: Update documentation from SSD ($(date '+%Y-%m-%d %H:%M'))"
    
    echo "Pushing to GitHub..."
    git push origin main
    
    echo "✓ Sync and push complete!"
else
    echo "No changes to sync."
fi

echo "=== Done ==="
