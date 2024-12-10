import asyncio
import aiohttp
import pandas as pd
import logging
from datetime import datetime
from textblob import TextBlob
from tabulate import tabulate
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from typing import Tuple, List
import nltk
from collections import Counter
from typing import Tuple
import telebot

# Telegram Bot Token
TELEGRAM_TOKEN = "7851602145:AAFxh70OSmMqStSFqtIrhz94Gs8Dnb4VlDc"
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Group ID for the public group (replace with actual group ID)
PUBLIC_GROUP_ID = "@pumpfunsentiment"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# API endpoint
LATEST_COINS_API = "https://frontend-api.pump.fun/coins/latest"

# Initialize an empty DataFrame to store fetched data
columns = ["Name", "Symbol", "Timestamp", "Sentiment", "Sentiment Description", "Market Cap", "CA Address", "USD Market Cap", "Image URI"]
tokens_df = pd.DataFrame(columns=columns)

# Function to convert timestamp to a human-readable format
def format_timestamp(timestamp: int) -> str:
    """Convert a timestamp to human-readable datetime."""
    try:
        return datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logging.error(f"Error formatting timestamp: {e}")
        return "Invalid Timestamp"

# Function to analyze sentiment
def analyze_sentiment(text: str) -> Tuple[float, str]:
    """Analyze sentiment of the given text using TextBlob."""
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity  # Polarity: -1 (negative) to 1 (positive)
        if polarity > 0.2:
            description = "Positive 😊"
        elif polarity < -0.2:
            description = "Negative 😢"
        else:
            description = "Neutral 😐"
        return polarity, description
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return 0.0, "Unknown"
    
def analyze_trends(df: pd.DataFrame) -> str:
    """Analyze trends based on token names, symbols, and sentiment."""
    # Name and Symbol analysis
    names = df['Name'].str.lower().str.split().explode().tolist()
    symbols = df['Symbol'].str.lower().tolist()
    all_text = names + symbols
    
    # Count occurrences
    name_counter = Counter(all_text)
    top_names = name_counter.most_common(5)
    
    # Sentiment Analysis Overview
    sentiment_counts = df['Sentiment Description'].value_counts().to_dict()
    
    # Metadata Analysis (assuming metadata includes additional info like categories or themes)
    metadata = df['Market Cap'].apply(lambda x: 'High Cap' if x != 'Unknown' else 'Unknown Cap').tolist()
    meta_counter = Counter(metadata)
    top_meta = meta_counter.most_common(3)

    # Prepare summary with improved formatting for Telegram
    summary = "🌟 **Current Trends on Pump.Fun** 🌟\n\n"
    
    summary += "**🔥 Popular Names & Symbols**\n"
    summary += '\n'.join([f"- `{name}`: {count} times" for name, count in top_names]) + "\n\n"
    
    summary += "**🎭 Sentiment Overview**\n"
    summary += '\n'.join([f"- {sentiment}: {count} tokens" for sentiment, count in sentiment_counts.items()]) + "\n\n"
    
    summary += "**📊 Market Cap Trends**\n"
    summary += '\n'.join([f"- {meta}: {count}" for meta, count in top_meta]) + "\n\n"

    # TF-IDF analysis should be performed here before using feature_names
    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform([' '.join(all_text)])
    feature_names = vectorizer.get_feature_names_out()  # Assign feature_names here

    summary += "**🔍 Top Keywords by TF-IDF**\n"
    summary += '\n'.join([f"- `{word}`: {score:.2f}" for word, score in sorted(zip(feature_names, X.sum(axis=0).A1), key=lambda x: x[1], reverse=True)[:5]])

    return summary

def send_telegram_message(message: str, chat_id=None):
    """Send a message to the Telegram bot."""
    try:
        if chat_id is None:
            chat_ids = ["6616771329", PUBLIC_GROUP_ID]
        else:
            chat_ids = [chat_id]

        for id in chat_ids:
            bot.send_message(chat_id=id, text=message, parse_mode="Markdown")
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

def format_telegram_message(dataframe: pd.DataFrame) -> str:
    """Format the DataFrame content as a Telegram message."""
    try:
        latest_data = dataframe.tail(10)
        message = "🚀 **Latest Tokens on Pump.Fun** 🚀\n\n"

        for _, row in latest_data.iterrows():
            message += (
                f"🌟 **Name**: {row['Name']}\n"
                f"💎 **Symbol**: {row['Symbol']}\n"
                f"📜 **Contract**: [View on Explorer](https://solscan.io/token/{row['CA Address']})\n"
                f"📈 **USD Market Cap**: ${row['USD Market Cap']}\n"
                f"💬 **Sentiment**: {row['Sentiment Description']}\n\n"
            )

        return message
    except Exception as e:
        logging.error(f"Error formatting Telegram message: {e}")
        return "Error formatting message."

# Async function to fetch data from the API
async def fetch_latest_token(session: aiohttp.ClientSession):
    """Fetch the latest meme token from the API."""
    try:
        async with session.get(LATEST_COINS_API) as response:
            if response.status == 200:
                data = await response.json()
                name = data.get("name", "Unknown")
                symbol = data.get("symbol", "Unknown")
                timestamp = data.get("created_timestamp", 0)
                market_cap = data.get("market_cap", "Unknown")
                usd_market_cap = data.get("usd_market_cap", "Unknown")
                ca_address = data.get("mint", "Unknown")
                image_uri = data.get("image_uri", "")

                # Format the timestamp
                readable_timestamp = format_timestamp(timestamp)

                # Analyze sentiment for Name and Symbol
                sentiment, sentiment_desc = analyze_sentiment(name + " " + symbol)

                return {
                    "Name": name,
                    "Symbol": symbol,
                    "Timestamp": readable_timestamp,
                    "Sentiment": sentiment,
                    "Sentiment Description": sentiment_desc,
                    "Market Cap": market_cap,
                    "CA Address": ca_address,
                    "USD Market Cap": usd_market_cap,
                    "Image URI": image_uri,
                }
            else:
                logging.error(f"Failed to fetch data. Status code: {response.status}")
                return None
    except Exception as e:
        logging.error(f"Exception occurred while fetching data: {e}")
        return None
            
async def fetch_tokens_continuously():
    """Continuously fetch latest tokens every 5 seconds and analyze trends."""
    global tokens_df
    async with aiohttp.ClientSession() as session:
        while True:
            token_data = await fetch_latest_token(session)
            if token_data:
                if any(token_data.values()):
                    tokens_df = pd.concat(
                        [tokens_df, pd.DataFrame([token_data])], ignore_index=True
                    )
                    tokens_df.drop_duplicates(subset=["Name", "Symbol"], keep="last", inplace=True)

                    trend_summary = analyze_trends(tokens_df)
                    bulk_sentiment = format_telegram_message(tokens_df)
                    
                    # Combine trend summary with bulk sentiment analysis
                    final_message = f"{trend_summary}\n\n{bulk_sentiment}"

                    send_telegram_message(final_message)

            await asyncio.sleep(15)
# Run the script
if __name__ == "__main__":
    try:
        nltk.download("stopwords")
        asyncio.run(fetch_tokens_continuously())
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Exiting...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")