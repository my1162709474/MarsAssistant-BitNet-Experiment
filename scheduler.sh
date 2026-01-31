#!/bin/bash
# BitNet Performance Optimization Scheduler
# 替代 cron，每10分钟执行一次优化

REPO_DIR="/Users/mars/.openclaw/workspace/MarsAssistant-BitNet-Experiment"
LOG_FILE="$REPO_DIR/experiments/OPTIMIZATION_LOG.md"
cd "$REPO_DIR"

echo "🚀 BitNet 优化调度器启动 - $(date)"

while true; do
    # 检查是否在工作时间 (8:00 - 23:00)
    HOUR=$(date +%H)
    if [ "$HOUR" -ge 8 ] && [ "$HOUR" -lt 23 ]; then
        if [ -f "optimize_bitnet.sh" ]; then
            ./optimize_bitnet.sh
        fi
    fi
    sleep 600  # 10分钟
done
