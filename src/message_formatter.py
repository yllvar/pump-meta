from telegram.helpers import escape_markdown
import logging

class MessageFormatter:
    @staticmethod
    def format_latest_token(token_data: dict, sol_price: float) -> str:
        try:
            # Compute market cap in SOL and USDT
            market_cap_sol = (
                float(token_data["Market Cap"]) / sol_price if token_data["Market Cap"] != "Unknown" else "N/A"
            )
            market_cap_usdt = (
                float(token_data["Market Cap"]) if token_data["Market Cap"] != "Unknown" else "N/A"
            )

            # Dynamic links
            telegram_link = token_data.get("telegram", "https://t.me/pumpfun")  # Fallback to default if not available
            trading_page_link = f"https://pump.fun/trades/{token_data['CA Address']}"  # Adjusted to CA Address

            # Escape Markdown characters in token data
            token_name = escape_markdown(token_data['Name'])
            token_symbol = escape_markdown(token_data['Symbol'])
            contract_address = escape_markdown(token_data['CA Address'])

            # Message formatting with bold descriptions and hyperlinks
            return (
                f"🚀 **Latest Token on Pump.Fun** 🚀\n\n"
                f"🌟 **Name**: [{token_name}]({telegram_link})\n"
                f"💎 **Symbol**: [{token_symbol}]({trading_page_link})\n"
                f"📜 **Contract Address**: `{contract_address}`\n"
                f"📈 **Market Cap**: **{market_cap_sol:.6f} SOL** (~${market_cap_usdt:.2f} USDT)\n"
            )
        except Exception as e:
            logging.error(f"Error formatting latest token message: {e}")
            return "Error formatting latest token message."
