import asyncio
import logging
from config import configure_logging, TELEGRAM_TOKEN, PUBLIC_GROUP_ID
from src.data_fetcher import DataFetcher
from src.sentiment_analyzer import SentimentAnalyzer
from src.trend_analyzer import TrendAnalyzer
from src.message_formatter import MessageFormatter
from src.telegram_bot import TelegramBot

async def main():
    configure_logging()
    data_fetcher = DataFetcher()
    sentiment_analyzer = SentimentAnalyzer()
    trend_analyzer = TrendAnalyzer()
    message_formatter = MessageFormatter()
    telegram_bot = TelegramBot(TELEGRAM_TOKEN, PUBLIC_GROUP_ID)

    try:
        async with data_fetcher:
            while True:
                data_fetcher.sol_price = await data_fetcher.fetch_sol_price()
                if data_fetcher.sol_price == 0.0:
                    logging.error("SOL price unavailable. Skipping this iteration...")
                    await asyncio.sleep(60)
                    continue

                token_data = await data_fetcher.fetch_latest_token()
                if token_data:
                    sentiment_data = sentiment_analyzer.analyze(token_data)
                    data_fetcher.update_tokens_df(sentiment_data)
                    trend_data = trend_analyzer.analyze(data_fetcher.tokens_df)
                    
                    overall_trends = trend_data['overall_trends']
                    keyword_insights = trend_data['keyword_insights']
                    
                    latest_token_message = message_formatter.format_latest_token(token_data, data_fetcher.sol_price)
                    final_message = f"{overall_trends}\n\n{keyword_insights}"
                    
                    await telegram_bot.send_message(final_message)
                    await telegram_bot.send_message(latest_token_message)
                else:
                    logging.warning("No new token data fetched. Skipping this iteration...")

                await asyncio.sleep(60)
    except KeyboardInterrupt:
        logging.info("Program interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())