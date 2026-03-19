import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ---------------- 配置 ----------------
# 修复 Windows 下可能存在的编码或连接问题
def get_sina_data(url):
    headers = {
        "Referer": "http://finance.sina.com.cn",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    return requests.get(url, headers=headers, timeout=10)

def fetch_history(symbol):
    """获取新浪历史日线数据"""
    s_type = "sh" if symbol.startswith('6') else "sz"
    full_code = f"{s_type}{symbol}"
    
    # 新浪 K 线接口
    url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={full_code}&scale=240&ma=no&datalen=100"
    
    r = get_sina_data(url)
    data = r.json()
    
    # 转换为 DataFrame
    df = pd.DataFrame(data)
    
    # 显式转换数值列，避开 apply(pd.to_numeric) 的报错
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') # 'coerce' 会把无法转换的转为 NaN
        
    return df

def fetch_realtime(symbol):
    """获取新浪实时快照数据"""
    s_type = "sh" if symbol.startswith('6') else "sz"
    url = f"http://hq.sinajs.cn/list={s_type}{symbol}"
    
    r = get_sina_data(url)
    content = r.text
    # 解析新浪格式: var hq_str_sz002602="...,open,prev_close,now,high,low,..."
    try:
        data = content.split('"')[1].split(',')
        if len(data) < 10: return None
        return {
            'open': float(data[1]),
            'high': float(data[4]),
            'low': float(data[5]),
            'close': float(data[3]),
            'volume': float(data[8]),
            'date': data[30]
        }
    except:
        return None

# ---------------- 预测引擎 ----------------
def predict_logic(df):
    """基于趋势、量价和形态的概率模型"""
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 55  # 基础分
    reasons = []

    # 1. 均线状态
    if last['close'] > last['ma20']:
        score += 10
        reasons.append("✅ 价格站稳 20 日均线（中期趋势向好）")
    else:
        score -= 10
        reasons.append("❌ 价格处于 20 日均线下方（中期承压）")

    # 2. 动能判断
    if last['close'] > prev['close'] and last['volume'] > df['volume'].tail(5).mean():
        score += 12
        reasons.append("🔥 量增价涨，资金入场积极")
    
    # 3. 形态：K线重心
    avg_price = (last['high'] + last['low']) / 2
    if last['close'] > avg_price:
        score += 8
        reasons.append("📈 收盘价位于当日高位，多头力量占优")

    # 4. 波动率修正
    change = (last['close'] - prev['close']) / prev['close']
    if abs(change) > 0.05:
        score -= 5 # 波动过大，次日往往会震荡修复
        reasons.append("⚠️ 今日波幅较大，需提防明早的回撤")

    prob = min(max(score, 10), 90)
    return prob, reasons

# ---------------- 主程序 ----------------
def run_analysis(symbol):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在获取 {symbol} 的数据...")
    
    # 1. 获取历史
    df = fetch_history(symbol)
    
    # 2. 检查是否需要融合实时数据
    now = datetime.now()
    is_trading = (9 <= now.hour <= 15) and now.weekday() < 5
    
    if is_trading:
        real = fetch_realtime(symbol)
        if real and real['date'] != df.iloc[-1]['date']:
            # 如果今天的数据还没在 K 线里，手动加进去
            new_row = pd.DataFrame([real])
            df = pd.concat([df, new_row], ignore_index=True)
            print(">>> 已融合当前盘中实时数据进行预测")

    # 3. 运行预测
    prob, reasons = predict_logic(df)
    
    print("-" * 30)
    print(f"股票代码: {symbol} | 当前价格: {df.iloc[-1]['close']}")
    print(f"预测明日常规上涨概率: {prob}%")
    print("分析依据:")
    for r in reasons:
        print(f" - {r}")

if __name__ == "__main__":
    # 还是以你关注的这只票为例
    run_analysis("002602")