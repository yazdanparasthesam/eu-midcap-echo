"""Send today's trades via Telegram bot.

Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env to enable. Silently skips
otherwise so the pipeline doesn't fail without credentials.
"""
import os, requests
from dotenv import load_dotenv
load_dotenv()

def send_message(text):
    tok = os.getenv("TELEGRAM_TOKEN")
    cid = os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not cid:
        print("[telegram] credentials not set; skipping."); return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": cid, "text": text, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print(f"[telegram] send failed: {e}")

def send_trades(path="reports/latest_trades.csv"):
    if not os.path.exists(path): return
    import pandas as pd
    df = pd.read_csv(path).tail(10)
    text = "*eu-midcap-echo: latest signals*\n```\n" + df.to_string(index=False) + "\n```"
    send_message(text)
