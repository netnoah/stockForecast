#!/bin/bash

if [ $# -eq 0 ]; then
    echo "用法: ./run.sh <股票代码1> [股票代码2] ..."
    echo ""
    echo "示例:"
    echo "  ./run.sh 002602            # 分析世纪华通"
    echo "  ./run.sh 002602 600519      # 分析多只股票"
    echo "  ./run.sh --review           # 查看预测自检报告"
    echo "  ./run.sh 002602 --refresh   # 强制刷新缓存数据"
    exit 0
fi

python forecast.py "$@"
