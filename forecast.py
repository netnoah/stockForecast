import argparse
import json
import os
import sys


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="A-Share Stock Quantitative Analyzer")
    parser.add_argument("symbols", nargs="*", help="Stock codes to analyze (e.g. 002602)")
    parser.add_argument("--review", action="store_true", help="Show self-reflection report only")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cached data")
    args = parser.parse_args()

    config = load_config()

    if args.review:
        print("Self-reflection report (not yet implemented)")
        return

    symbols = args.symbols or config.get("stocks", [])
    if not symbols:
        print("No stocks specified. Use positional args or configure stocks in config.json")
        sys.exit(1)

    for symbol in symbols:
        print(f"Analyzing {symbol}... (not yet implemented)")


if __name__ == "__main__":
    main()
