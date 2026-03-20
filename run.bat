@echo off
chcp 65001 >nul 2>&1

if "%~1"=="" (
    echo 用法: run.bat ^<股票代码1^> [股票代码2] ...
    echo.
    echo 示例:
    echo   run.bat 002602            # 分析世纪华通
    echo   run.bat 002602 600519      # 分析多只股票
    echo   run.bat -l                # 读取 config.json 中的 stock_list
    echo   run.bat --review           # 查看预测自检报告
    echo   run.bat 002602 --refresh   # 强制刷新缓存数据
    exit /b 0
)

python forecast.py %*
