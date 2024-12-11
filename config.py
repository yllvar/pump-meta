import logging

PUMP_PORTAL_WS = "wss://pumpportal.fun/api/data"
TELEGRAM_TOKEN = "your telegram bot token"
PUBLIC_GROUP_ID = "your telegram public group id"
LATEST_COINS_API = "https://frontend-api.pump.fun/coins/latest"
SOL_PRICE_API = "https://frontend-api.pump.fun/sol-price"
TRADE_API = "https://frontend-api.pump.fun/trades/latest"

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )