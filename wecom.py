"""WeChat Work (企业微信) webhook push module.

Provides functionality to push stock analysis reports to a WeChat Work
group robot via webhook. Supports markdown and plain text message formats.
"""

from __future__ import annotations

import re
from datetime import datetime

import requests


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# WeChat Work markdown message size limit (bytes)
_WECOM_MAX_SIZE = 4096


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return _ANSI_RE.sub("", text)


def report_to_wecom_markdown(report: str) -> str:
    """Convert a plain-text report to WeChat Work markdown format.

    Parses the structured output of ``analyzer.format_report()`` and
    converts each section to appropriate markdown syntax supported by
    WeChat Work (headings, bold, font color, blockquote, lists).
    """
    plain = strip_ansi(report)
    lines = plain.split("\n")
    md_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Separator line (===...) -> horizontal rule
        if stripped.startswith("=") and set(stripped) <= {"="}:
            md_lines.append("---")
            continue

        # Section header (--- xxx ---) -> heading
        if stripped.startswith("---") and stripped.endswith("---") and len(stripped) > 4:
            title = stripped.strip("-").strip()
            if title:
                md_lines.append(f"## {title}")
            continue

        # Stock title line: "  Name (symbol) | date"
        if " | " in stripped and "(" in stripped and ")" in stripped:
            parts = stripped.split("|", 1)
            name_part = parts[0].strip()
            date_part = parts[1].strip()
            md_lines.append(f"## {name_part}")
            md_lines.append(f"> {date_part}")
            continue

        # Signal rating line -> colored text
        if "[信号评级]" in stripped:
            for signal in ("强烈买入", "买入", "强烈卖出", "卖出"):
                if signal in stripped:
                    color = "warning" if "买入" in signal else "info"
                    stripped = stripped.replace(
                        signal,
                        f'<font color="{color}">{signal}</font>',
                    )
                    break
            md_lines.append(stripped)
            continue

        # Score line -> colored score
        if "[综合评分]" in stripped:
            score_match = re.search(r"(-?\d+)分", stripped)
            if score_match:
                score_val = int(score_match.group(1))
                if score_val >= 15:
                    color = "warning"
                elif score_val <= -15:
                    color = "info"
                else:
                    color = "comment"
                score_str = f"{score_val}分"
                stripped = stripped.replace(
                    score_str,
                    f'<font color="{color}">{score_str}</font>',
                )
            md_lines.append(stripped)
            continue

        # Indented list items (indicator lines, market lines)
        if line.startswith("  ") and stripped and not stripped.startswith("["):
            md_lines.append(f"- {stripped}")
            continue

        # Risk alert lines ([!] prefix) -> blockquote
        if "[!]" in stripped:
            stripped = stripped.replace("[!]", "⚠️")
            md_lines.append(f"> {stripped}")
            continue

        # Score reference line (dim text at bottom)
        if "评分参考:" in stripped:
            md_lines.append(f"> {stripped}")
            continue

        # Default: keep as-is, trim leading whitespace
        md_lines.append(stripped)

    return "\n".join(md_lines)


def send_to_wecom(webhook_url: str, content: str, msg_format: str = "markdown") -> bool:
    """Send a message to WeChat Work webhook.

    Args:
        webhook_url: The webhook URL for the WeChat Work group robot.
        content: The message content to send.
        msg_format: Either "markdown" or "text".

    Returns:
        True if sent successfully, False otherwise.
    """
    # Check size limit
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > _WECOM_MAX_SIZE:
        content = content[:_WECOM_MAX_SIZE - 20].encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        content += "\n\n[内容过长已截断]"

    if msg_format == "markdown":
        payload = {
            "msgtype": "markdown",
            "markdown": {"content": content},
        }
    else:
        payload = {
            "msgtype": "text",
            "text": {"content": content},
        }

    try:
        resp = requests.post(
            webhook_url,
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("errcode", -1) != 0:
            print(f"  [推送失败] 企业微信返回错误: {result.get('errmsg', '未知')}")
            return False
        print("  [推送成功] 已发送到企业微信")
        return True
    except requests.RequestException as e:
        print(f"  [推送失败] 网络请求异常: {e}")
        return False


def push_reports(config: dict, sections: list[str], title: str = "分析报告") -> bool:
    """Optionally push one or more report sections to WeChat Work.

    Asks the user **once** for confirmation, then sends all sections.
    If the total content exceeds the WeChat Work 4096-byte limit,
    each section is sent as a separate message.

    If ``wecom_webhook`` is not configured in *config*, silently returns
    without prompting.

    Args:
        config: The loaded config dict.
        sections: List of report strings (ANSI-formatted or plain).
        title: A short title for the push confirmation prompt.

    Returns:
        True if all messages were sent successfully, False otherwise or skipped.
    """
    webhook_url = config.get("wecom_webhook", "")
    if not webhook_url or not webhook_url.strip():
        return False

    msg_format = config.get("wecom_msg_format", "markdown")

    answer = input(f"  推送「{title}」到企业微信? [y/N]: ").strip().lower()
    if answer != "y":
        print("  已取消推送。")
        return False

    # Convert all sections
    if msg_format == "markdown":
        converted = [report_to_wecom_markdown(s) for s in sections]
    else:
        converted = [strip_ansi(s) for s in sections]

    # Prepend generation timestamp
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if msg_format == "markdown":
        header = f"## 分析报告\n> 生成时间: {now_str}"
    else:
        header = f"分析报告\n生成时间: {now_str}"

    # Try sending as a single message first
    combined = "\n\n".join([header] + converted)
    if len(combined.encode("utf-8")) <= _WECOM_MAX_SIZE:
        return send_to_wecom(webhook_url, combined, msg_format)

    # Too large — send each section separately
    ok = True
    for content in converted:
        if not send_to_wecom(webhook_url, content, msg_format):
            ok = False
    return ok
