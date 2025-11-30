#!/bin/bash
# Music_ReClass Daily Startup Script
# Run this at the beginning of each work session

clear
echo "=========================================="
echo "  Music_ReClass - Daily Startup"
echo "=========================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)
echo "📁 Project Directory: $PROJECT_DIR"
echo ""

# Check Git status
echo "📊 Git Status:"
git status -s
echo ""

# Show recent commits
echo "📝 Recent Commits:"
git log --oneline -3
echo ""

# Check GPU status
echo "🖥️  GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s | Memory: %s/%s MB (%.1f%%) | Usage: %s%%\n", $1, $2, $3, $4, ($3/$4)*100, $5}'
else
    echo "  ⚠️  nvidia-smi not available"
fi
echo ""

# Check Python environment
echo "🐍 Python Environment:"
python3 --version
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo ""

# Check feature files
echo "📦 Feature Files:"
if [ -d "features" ]; then
    ls -lh features/*.npy 2>/dev/null | awk '{printf "  %s: %s\n", $9, $5}' || echo "  No .npy files found"
else
    echo "  features/ directory not found"
fi
echo ""

# Check recent logs
echo "📋 Recent Training Logs:"
if [ -f "logs/training.log" ]; then
    echo "  Last 3 lines from training.log:"
    tail -3 logs/training.log | sed 's/^/    /'
else
    echo "  No training.log found"
fi
echo ""

# Show available scripts
echo "🔧 Quick Commands:"
echo "  Training:"
echo "    python3 training/train_msd.py              # Fast (2 min, 77%)"
echo "    python3 training/train_gtzan_v2.py         # Balanced (45 min, 70-80%)"
echo "    python3 training/train_gtzan_enhanced.py   # Best (4 hrs, 80-90%)"
echo ""
echo "  Feature Extraction:"
echo "    python3 extractors/extract_fma_features.py /path/to/audio/"
echo "    python3 extractors/extract_all_features.py"
echo ""
echo "  Classification:"
echo "    python3 classification/classify_music_tbc.py --input /path/to/music"
echo ""
echo "  Analysis:"
echo "    python3 analysis/analyze_data.py"
echo "    python3 utils/gpu_monitor.py"
echo ""

# Show TODO items if file exists
if [ -f "TODO.md" ]; then
    echo "✅ TODO Items:"
    grep -E "^\- \[ \]" TODO.md | head -5 | sed 's/^/  /'
    echo ""
fi

# Prompt for Kiro CLI
echo "=========================================="
echo "💬 Start Kiro CLI:"
echo "   kiro-cli chat"
echo ""
echo "📖 View Daily Checklist:"
echo "   cat DAILY_STARTUP.md"
echo ""
echo "🚀 Ready to work!"
echo "=========================================="
