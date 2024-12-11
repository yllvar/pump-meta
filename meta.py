import asyncio
import aiohttp
import pandas as pd
import logging
import websockets
import json
from datetime import datetime
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import telebot
import nltk
from typing import Tuple, Dict
<<<<<<< HEAD
from collections import Counter
from tenacity import retry, stop_after_attempt, wait_fixed

# New PumpPortal WebSocket URL
PUMP_PORTAL_WS = "wss://pumpportal.fun/api/data"
=======
>>>>>>> origin/main

# Telegram Bot Token
TELEGRAM_TOKEN = "your-telegram-token"
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Group ID for the public group (replace with actual group ID)
PUBLIC_GROUP_ID = "your-telegram-PUBLIC-group-id"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# API endpoints
LATEST_COINS_API = "https://frontend-api.pump.fun/coins/latest"
SOL_PRICE_API = "https://frontend-api.pump.fun/sol-price"
TRADE_API = "https://frontend-api.pump.fun/trades/latest"

# Sentiment weights
COMMENT_WEIGHT = 0.5
TRADE_WEIGHT = 0.3
MARKET_ACTIVITY_WEIGHT = 0.2

# Initialize an empty DataFrame to store fetched data
columns = ["Name", "Symbol", "Timestamp", "Sentiment", "Sentiment Description", "Market Cap", "CA Address", "USD Market Cap", "Image URI"]
tokens_df = pd.DataFrame(columns=columns)

# Ensure necessary columns exist with default values
tokens_df = tokens_df.assign(
    **{
        "Market Cap Change": None,  # Initialize Market Cap Change column
        "Sentiment Description": "Unknown"  # Initialize Sentiment Description column
    }
)

# Safely fill missing data in case columns are modified later
tokens_df["Market Cap Change"] = tokens_df["Market Cap Change"].fillna(0)
tokens_df["Sentiment Description"] = tokens_df["Sentiment Description"].fillna("Unknown")

# Function to convert timestamp to a human-readable format
def format_timestamp(timestamp: int) -> str:
    """Convert a timestamp to human-readable datetime."""
    try:
        return datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logging.error(f"Error formatting timestamp: {e}")
        return "Invalid Timestamp"
    
# Function to initialize WebSocket connection
async def init_websocket_connection():
    try:
        websocket = await websockets.connect(PUMP_PORTAL_WS)
        return websocket
    except Exception as e:
        logging.error(f"Failed to connect to PumpPortal WebSocket: {e}")
        return None
    
# Function to subscribe to token events via WebSocket
async def subscribe_to_events(websocket, token_address=None):
    if token_address:
        payload = {"method": "subscribeTokenTrade", "keys": [token_address]}
    else:
        payload = {"method": "subscribeNewToken"}
    await websocket.send(json.dumps(payload))

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

# Analyze trends
def analyze_trends(df: pd.DataFrame) -> str:
<<<<<<< HEAD
    """Analyze trends based on token names, symbols, descriptions, and sentiment."""
    try:
        if df.empty:
            return "No data available for analysis."

        # Check if 'Description' column exists before using it
        if 'Description' in df.columns:
            descriptions = df['Description'].dropna().str.lower()
            if not descriptions.empty:
                all_text = ' '.join(df['Name'].dropna().str.lower() + ' ' +
                                    df['Symbol'].dropna().str.lower() + ' ' +
                                    descriptions)
            else:
                all_text = ' '.join(df['Name'].dropna().str.lower() + ' ' +
                                    df['Symbol'].dropna().str.lower())
                logging.warning("No valid descriptions available.")
        else:
            all_text = ' '.join(df['Name'].dropna().str.lower() + ' ' +
                                df['Symbol'].dropna().str.lower())
            logging.warning("Description column is missing from DataFrame.")

        vectorizer = TfidfVectorizer(max_features=10)
        X = vectorizer.fit_transform([all_text])
        top_keywords = vectorizer.get_feature_names_out()

        # Sentiment Analysis Overview
        if 'Sentiment Description' in df.columns:
            sentiment_counts = df['Sentiment Description'].value_counts().to_dict()
            # Convert iterrows() to list to avoid 'generator' object error
            sentiment_examples = {
                sentiment: ', '.join([
                    f"**{row['Symbol']}**" for _, row in list(df[df['Sentiment Description'] == sentiment].iterrows())[:3]
                ])
                for sentiment in sentiment_counts.keys()
            }
        else:
            sentiment_counts = {}
            sentiment_examples = {"Unknown": "N/A"}
            logging.warning("Sentiment Description column is missing from DataFrame.")

        # Top Tokens by Market Cap
        top_tokens = df[df['Market Cap'].notna()].nlargest(5, 'Market Cap')
        top_tokens_details = [
            f"**{row['Symbol']}**: {row['Name']} - {row['Market Cap']:.2f} SOL\n"
            f"  📜 Contract: `{row['CA Address']}`\n"
            f"  💰 Price: {row.get('Price', 'N/A')} SOL\n"
            f"  🪙 Supply: {row.get('Supply', 'N/A')}"
            for _, row in top_tokens.iterrows()
        ]

        # High Cap Analysis
        high_cap_tokens = df[df['Market Cap'].notna()]
        high_cap_avg = high_cap_tokens['Market Cap'].mean() if not high_cap_tokens.empty else 0
        high_cap_count = len(high_cap_tokens)

        # Significant Growth Analysis
        growth_threshold = 0.1
        significant_growth_tokens = high_cap_tokens[high_cap_tokens['Market Cap Change'] > growth_threshold]
        significant_growth_details = [
            f"**{row['Symbol']}**: {row['Name']} - Growth: {row['Market Cap Change']:.2f} SOL\n"
            f"  📜 Contract: `{row['CA Address']}`\n"
            f"  💰 Price: {row.get('Price', 'N/A')} SOL\n"
            f"  🪙 Supply: {row.get('Supply', 'N/A')}"
            for _, row in significant_growth_tokens.iterrows()
        ]

        # Summary Construction
        summary = "🌟 **Current Trends on Pump.Fun** 🌟\n\n"
        summary += "**🔥 *Popular Keywords***\n"
        summary += '\n'.join([f"- `{keyword}`" for keyword in top_keywords]) + "\n\n"

        summary += "**🎭 *Sentiment Overview***\n"
        for sentiment, count in sentiment_counts.items():
            example_symbols = sentiment_examples.get(sentiment, "N/A")
            summary += f"- {sentiment}: {count} tokens (e.g., {example_symbols})\n"
        summary += "\n"

        summary += "**📊 *Market Cap Trends***\n"
        summary += f"- **Top Tokens by *Market Cap***:\n"
        summary += '\n'.join([f"  - {details}" for details in top_tokens_details]) + "\n"
        summary += f"- **High Cap Tokens**: {high_cap_count} tokens\n"
        summary += f"  - 📈 *Average Market Cap*: {high_cap_avg:.2f} SOL\n"
        summary += f"- **Significant Growth**:\n"
        if significant_growth_details:
            summary += '\n'.join([f"  - {details}" for details in significant_growth_details]) + "\n\n"
        else:
            summary += "  - 🚫 No tokens showed significant growth.\n\n"

=======
    """Analyze trends based on token names, symbols, and sentiment."""
    try:
        # Data validation: Ensure necessary columns exist
        if df.empty:
            return "No data available for analysis."

        # Name and Symbol analysis
        all_text = ' '.join(df['Name'].dropna().str.lower() + ' ' + df['Symbol'].dropna().str.lower())
        vectorizer = TfidfVectorizer(max_features=10)
        X = vectorizer.fit_transform([all_text])
        top_keywords = vectorizer.get_feature_names_out()

        # Sentiment Analysis Overview
        sentiment_counts = df['Sentiment Description'].value_counts().to_dict()

        # Top Tokens by Market Cap
        top_tokens = df[df['Market Cap'].notna()].nlargest(5, 'Market Cap')
        top_tokens_details = [
            f"**{row['Symbol']}**: {row['Name']} - {row['Market Cap']:.2f} SOL\n"
            f"  📜 Contract: `{row['CA Address']}`\n"
            f"  💰 Price: {row.get('Price', 'N/A')} SOL\n"
            f"  🪙 Supply: {row.get('Supply', 'N/A')}"
            for _, row in top_tokens.iterrows()
        ]

        # High Cap Analysis
        high_cap_tokens = df[df['Market Cap'].notna()]
        high_cap_avg = high_cap_tokens['Market Cap'].mean() if not high_cap_tokens.empty else 0
        high_cap_count = len(high_cap_tokens)

        # Significant Growth Analysis
        growth_threshold = 0.1  # 10% growth
        significant_growth_tokens = high_cap_tokens[high_cap_tokens['Market Cap Change'] > growth_threshold]
        significant_growth_details = [
            f"**{row['Symbol']}**: {row['Name']} - Growth: {row['Market Cap Change']:.2f} SOL\n"
            f"  📜 Contract: `{row['CA Address']}`\n"
            f"  💰 Price: {row.get('Price', 'N/A')} SOL\n"
            f"  🪙 Supply: {row.get('Supply', 'N/A')}"
            for _, row in significant_growth_tokens.iterrows()
        ]

        # Sentiment Example Symbols
        sentiment_examples = {
            sentiment: ', '.join([
                f"**{row['Symbol']}**: {row['Name']} - 📜 `{row['CA Address']}`"
                for _, row in df[df['Sentiment Description'] == sentiment].iterrows()[:5]
            ])
            for sentiment in sentiment_counts.keys()
        }

        # Summary Construction
        summary = "🌟 **Current Trends on Pump.Fun** 🌟\n\n"
        
        summary += "**🔥 *Popular Keywords***\n"
        summary += '\n'.join([f"- `{keyword}`" for keyword in top_keywords]) + "\n\n"

        summary += "**🎭 *Sentiment Overview***\n"
        for sentiment, count in sentiment_counts.items():
            example_symbols = sentiment_examples[sentiment]
            summary += f"- {sentiment}: {count} tokens (e.g., {example_symbols})\n"
        summary += "\n"

        summary += "**📊 *Market Cap Trends***\n"
        summary += f"- **Top Tokens by *Market Cap***:\n"
        summary += '\n'.join([f"  - {details}" for details in top_tokens_details]) + "\n"
        summary += f"- **High Cap Tokens**: {high_cap_count} tokens\n"
        summary += f"  - 📈 *Average Market Cap*: {high_cap_avg:.2f} SOL\n"
        summary += f"- **Significant Growth**:\n"
        if significant_growth_details:
            summary += '\n'.join([f"  - {details}" for details in significant_growth_details]) + "\n\n"
        else:
            summary += "  - 🚫 No tokens showed significant growth.\n\n"

>>>>>>> origin/main
        return summary
    except Exception as e:
        logging.error(f"Error analyzing trends: {e}")
        return "Error analyzing trends."
<<<<<<< HEAD
    
# Function to fetch SOL price
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
async def fetch_sol_price(session: aiohttp.ClientSession) -> float:
    """Fetch the current SOL price."""
=======

# Function to fetch SOL price
async def fetch_sol_price(session: aiohttp.ClientSession) -> float:
>>>>>>> origin/main
    try:
        async with session.get(SOL_PRICE_API) as response:
            if response.status == 200:
                data = await response.json()
                return float(data.get("solPrice", 0.0))
<<<<<<< HEAD
            else:
                logging.error(f"Failed to fetch SOL price. Status: {response.status}")
                return 0.0
    except Exception as e:
        logging.error(f"Error fetching SOL price: {e}")
        raise  # Raise the exception to trigger the retry mechanism

# Function to fetch trade data
async def fetch_trade_data(session: aiohttp.ClientSession) -> Dict:
    try:
        async with session.get(TRADE_API) as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        logging.error(f"Error fetching trade data: {e}")
    return {}

# Analyze comment sentiment
def analyze_comment_sentiment(text: str) -> float:
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity  # -1 (negative) to 1 (positive)
    except Exception as e:
        logging.error(f"Error analyzing comment sentiment: {e}")
        return 0.0

# Analyze trade data sentiment
def analyze_trade_sentiment(trade_data: Dict) -> float:
    try:
        buy_trades = sum(1 for trade in trade_data if trade.get("is_buy", False))
        sell_trades = len(trade_data) - buy_trades

        if sell_trades == 0:
            buy_sell_ratio = 1.0
        else:
            buy_sell_ratio = buy_trades / sell_trades

        # Sentiment modifiers based on buy/sell ratios
        if buy_sell_ratio > 1.5:
            return 0.8  # Bullish sentiment
        elif buy_sell_ratio < 0.7:
            return -0.8  # Bearish sentiment
        else:
            return 0.0  # Neutral sentiment
    except Exception as e:
        logging.error(f"Error analyzing trade sentiment: {e}")
        return 0.0

# Analyze market activity sentiment
def analyze_market_activity(market_cap_change: float) -> float:
    try:
        if market_cap_change > 0.1:
            return 0.6  # Positive sentiment for growth > 10%
        elif market_cap_change < -0.1:
            return -0.6  # Negative sentiment for decline > 10%
        else:
            return 0.0  # Neutral sentiment
    except Exception as e:
        logging.error(f"Error analyzing market activity: {e}")
        return 0.0

# Combine sentiments with weighted averaging
def aggregate_sentiment(comment_sentiment: float, trade_sentiment: float, market_sentiment: float, data_quality: Dict[str, int]) -> float:
    try:
        total_weight = COMMENT_WEIGHT * data_quality['comments'] + \
                       TRADE_WEIGHT * data_quality['trades'] + \
                       MARKET_ACTIVITY_WEIGHT * data_quality['market_activity']

        # Normalize weights
        comment_weight = COMMENT_WEIGHT * data_quality['comments'] / total_weight
        trade_weight = TRADE_WEIGHT * data_quality['trades'] / total_weight
        market_weight = MARKET_ACTIVITY_WEIGHT * data_quality['market_activity'] / total_weight

        total_sentiment = (
            comment_weight * comment_sentiment +
            trade_weight * trade_sentiment +
            market_weight * market_sentiment
        )
        return total_sentiment
    except Exception as e:
        logging.error(f"Error aggregating sentiment: {e}")
        return 0.0


# Fetch and process token data
async def fetch_and_analyze_sentiment():
    async with aiohttp.ClientSession() as session:
        sol_price = await fetch_sol_price(session)
        trade_data = await fetch_trade_data(session)

        # Example: Comment and market cap sentiment for a token
        token_comment = "This token is mooning! Buy now!"
        market_cap_change = 0.12  # Example: 12% growth in market cap

        # Sentiment analysis
        comment_sentiment = analyze_comment_sentiment(token_comment)
        trade_sentiment = analyze_trade_sentiment(trade_data)
        market_sentiment = analyze_market_activity(market_cap_change)

        # Aggregate sentiments
        final_sentiment = aggregate_sentiment(comment_sentiment, trade_sentiment, market_sentiment)
        
        logging.info(f"Aggregated Sentiment Score: {final_sentiment:.2f}")
        
# Function to analyze trends based on token descriptions
def analyze_trends_from_descriptions(df: pd.DataFrame) -> str:
    """Analyze trends based on token descriptions, linking keywords to market trends."""
    try:
        if df.empty:
            return "No data available for description-based analysis."

        if 'Description' not in df.columns:
            return "Description column is missing from DataFrame."

        descriptions = df['Description'].dropna().str.lower()

        # Check if there are any valid descriptions after removing NA values
        if descriptions.empty:
            return "No valid descriptions available for analysis."

        # Extract keywords using TF-IDF with adjusted settings
        vectorizer = TfidfVectorizer(stop_words=None, min_df=1, max_features=10)
        X = vectorizer.fit_transform(descriptions)
        keywords = vectorizer.get_feature_names_out()

        # If no keywords are found, return an informative message
        if not keywords:
            return "No keywords could be extracted from descriptions; all might be stop words or too rare."

        # Count keyword occurrences
        keyword_counter = Counter()
        keyword_to_tokens = {keyword: [] for keyword in keywords}

        for keyword in keywords:
            # Use .any() to check if the keyword is in the description
            for _, row in df.iterrows():
                if isinstance(row['Description'], str):  # Check if Description is a string
                    if keyword in row['Description'].lower().split():
                        keyword_counter[keyword] += 1
                        keyword_to_tokens[keyword].append(row['Symbol'])
                else:  # If Description is not a string, skip or handle differently
                    logging.warning(f"Description for token {row['Symbol']} is not a string: {type(row['Description'])}")

        # Compute average market cap and sentiment for each keyword
        keyword_insights = []
        for keyword in keywords:
            tokens = df[df['Symbol'].isin(keyword_to_tokens[keyword])]
            avg_market_cap = tokens['Market Cap'].mean()
            avg_sentiment = tokens['Sentiment'].mean()
            top_tokens = tokens.nlargest(3, 'Market Cap')[['Symbol', 'Market Cap']]

            # Format top tokens as a string
            top_tokens_str = ', '.join(
                [f"`{row['Symbol']}` ({row['Market Cap']:.2f} SOL)" for _, row in top_tokens.iterrows()]
            )

            # Append insights for this keyword
            keyword_insights.append(
                f"🔹 **Keyword**: `{keyword}`\n"
                f"   - 📄 **Occurrences**: {keyword_counter[keyword]}\n"
                f"   - 📈 **Avg Market Cap**: {avg_market_cap:.2f} SOL\n"
                f"   - 💬 **Avg Sentiment**: {avg_sentiment:.2f}\n"
                f"   - 🌟 **Top Tokens**: {top_tokens_str if top_tokens_str else 'None'}"
            )

        # Format the output
        summary = "🔑 **Contextual Keyword Insights**\n\n"
        summary += '\n\n'.join(keyword_insights)  # Add a blank line between keyword blocks
        return summary
    except Exception as e:
        logging.error(f"Error analyzing trends from descriptions: {e}")
        return "Error analyzing trends from descriptions."
=======
    except Exception as e:
        logging.error(f"Error fetching SOL price: {e}")
    return 0.0

# Function to fetch trade data
async def fetch_trade_data(session: aiohttp.ClientSession) -> Dict:
    try:
        async with session.get(TRADE_API) as response:
            if response.status == 200:
                return await response.json()
    except Exception as e:
        logging.error(f"Error fetching trade data: {e}")
    return {}

# Analyze comment sentiment
def analyze_comment_sentiment(text: str) -> float:
    try:
        analysis = TextBlob(text)
        return analysis.sentiment.polarity  # -1 (negative) to 1 (positive)
    except Exception as e:
        logging.error(f"Error analyzing comment sentiment: {e}")
        return 0.0

# Analyze trade data sentiment
def analyze_trade_sentiment(trade_data: Dict) -> float:
    try:
        buy_trades = sum(1 for trade in trade_data if trade.get("is_buy", False))
        sell_trades = len(trade_data) - buy_trades

        if sell_trades == 0:
            buy_sell_ratio = 1.0
        else:
            buy_sell_ratio = buy_trades / sell_trades

        # Sentiment modifiers based on buy/sell ratios
        if buy_sell_ratio > 1.5:
            return 0.8  # Bullish sentiment
        elif buy_sell_ratio < 0.7:
            return -0.8  # Bearish sentiment
        else:
            return 0.0  # Neutral sentiment
    except Exception as e:
        logging.error(f"Error analyzing trade sentiment: {e}")
        return 0.0

# Analyze market activity sentiment
def analyze_market_activity(market_cap_change: float) -> float:
    try:
        if market_cap_change > 0.1:
            return 0.6  # Positive sentiment for growth > 10%
        elif market_cap_change < -0.1:
            return -0.6  # Negative sentiment for decline > 10%
        else:
            return 0.0  # Neutral sentiment
    except Exception as e:
        logging.error(f"Error analyzing market activity: {e}")
        return 0.0

# Combine sentiments with weighted averaging
def aggregate_sentiment(comment_sentiment: float, trade_sentiment: float, market_sentiment: float, data_quality: Dict[str, int]) -> float:
    try:
        total_weight = COMMENT_WEIGHT * data_quality['comments'] + \
                       TRADE_WEIGHT * data_quality['trades'] + \
                       MARKET_ACTIVITY_WEIGHT * data_quality['market_activity']

        # Normalize weights
        comment_weight = COMMENT_WEIGHT * data_quality['comments'] / total_weight
        trade_weight = TRADE_WEIGHT * data_quality['trades'] / total_weight
        market_weight = MARKET_ACTIVITY_WEIGHT * data_quality['market_activity'] / total_weight

        total_sentiment = (
            comment_weight * comment_sentiment +
            trade_weight * trade_sentiment +
            market_weight * market_sentiment
        )
        return total_sentiment
    except Exception as e:
        logging.error(f"Error aggregating sentiment: {e}")
        return 0.0


# Fetch and process token data
async def fetch_and_analyze_sentiment():
    async with aiohttp.ClientSession() as session:
        sol_price = await fetch_sol_price(session)
        trade_data = await fetch_trade_data(session)

        # Example: Comment and market cap sentiment for a token
        token_comment = "This token is mooning! Buy now!"
        market_cap_change = 0.12  # Example: 12% growth in market cap

        # Sentiment analysis
        comment_sentiment = analyze_comment_sentiment(token_comment)
        trade_sentiment = analyze_trade_sentiment(trade_data)
        market_sentiment = analyze_market_activity(market_cap_change)

        # Aggregate sentiments
        final_sentiment = aggregate_sentiment(comment_sentiment, trade_sentiment, market_sentiment)
        
        logging.info(f"Aggregated Sentiment Score: {final_sentiment:.2f}")
>>>>>>> origin/main

# Format the latest token
def format_latest_token(token_data: dict, sol_price: float) -> str:
    """Format the latest token details without backlinks."""
    try:
        market_cap_sol = float(token_data["Market Cap"]) / sol_price if token_data["Market Cap"] != "Unknown" else "N/A"
        market_cap_usdt = float(token_data["Market Cap"]) if token_data["Market Cap"] != "Unknown" else "N/A"

        return (
            f"🚀 **Latest Token on Pump.Fun** 🚀\n\n"
            f"🌟 **Name**: {token_data['Name']}\n"  # No backlink
            f"💎 **Symbol**: {token_data['Symbol']}\n"  # No backlink
            f"📜 **Contract Address**: {token_data['CA Address']}\n"
            f"📈 **Market Cap**: {market_cap_sol} SOL (~${market_cap_usdt:.2f} USDT)\n"
        )
    except Exception as e:
        logging.error(f"Error formatting latest token message: {e}")
        return "Error formatting latest token message."

<<<<<<< HEAD
# Update fetch_latest_token to use WebSocket as backup
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
async def fetch_latest_token(session, websocket):
=======
# Fetch latest token
async def fetch_latest_token(session: aiohttp.ClientSession):
    """Fetch the latest meme token from the API."""
>>>>>>> origin/main
    try:
        async with session.get(LATEST_COINS_API) as response:
            if response.status == 200:
                data = await response.json()
<<<<<<< HEAD
                token_data = {
=======
                return {
>>>>>>> origin/main
                    "Name": data.get("name", "Unknown"),
                    "Symbol": data.get("symbol", "Unknown"),
                    "Timestamp": format_timestamp(data.get("created_timestamp", 0)),
                    "Market Cap": data.get("market_cap", "Unknown"),
                    "USD Market Cap": data.get("usd_market_cap", "Unknown"),
                    "CA Address": data.get("mint", "Unknown"),
                    "Image URI": data.get("image_uri", ""),
                }
                if token_data['Market Cap'] == 'Unknown':
                    raise ValueError("Market Cap is Unknown")
                
                # Fetch metadata (description, supply, and price)
                metadata = await fetch_token_metadata(session, token_data['CA Address'])
                token_data.update(metadata)

                # Analyze sentiment based on the token's description
                sentiment, sentiment_desc = analyze_sentiment(token_data.get('Description', ''))
                token_data['Sentiment'] = sentiment
                token_data['Sentiment Description'] = sentiment_desc

                return token_data
            else:
                logging.error(f"Failed to fetch token. Status: {response.status}")
<<<<<<< HEAD
                raise ValueError("Failed to fetch token")
    except (ValueError, Exception) as e:
        logging.warning(f"Primary API failed: {e}. Attempting WebSocket backup.")
        if websocket:
            try:
                await subscribe_to_events(websocket, token_data.get("CA Address"))
                async for message in websocket:
                    msg_data = json.loads(message)
                    if msg_data.get('method') == 'tokenTrade' and msg_data.get('mint') == token_data['CA Address']:
                        token_data['Market Cap'] = msg_data.get('vSolInBondingCurve', 'Unknown')
                        token_data['USD Market Cap'] = msg_data.get('marketCapSol', 'Unknown')
                        
                        # If you can get description or another text from WebSocket, analyze sentiment here too
                        if 'description' in msg_data:
                            sentiment, sentiment_desc = analyze_sentiment(msg_data['description'])
                            token_data['Sentiment'] = sentiment
                            token_data['Sentiment Description'] = sentiment_desc
                        
                        return token_data
            except Exception as ws_e:
                logging.error(f"WebSocket fetch failed: {ws_e}")
        return None

# Function to fetch token metadata (description and other details)
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
async def fetch_token_metadata(session: aiohttp.ClientSession, ca_address: str) -> dict:
    """Fetch additional metadata for a token."""
    url = f"https://frontend-api.pump.fun/coins/{ca_address}"
    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "Description": data.get("description", "No description available."),
                    "Supply": data.get("total_supply", "Unknown"),
                    "Price": data.get("market_cap", 0) / data.get("total_supply", 1) if data.get("total_supply", 1) > 0 else "Unknown"
                }
            elif response.status == 500:
                logging.warning(f"Server error (500) while fetching metadata for CA {ca_address}. Skipping...")
                return {"Description": "Server error: Metadata unavailable.", "Supply": "Unknown", "Price": "Unknown"}
            else:
                logging.error(f"Failed to fetch token metadata for CA {ca_address}. Status: {response.status}")
                return {"Description": "Metadata fetch failed.", "Supply": "Unknown", "Price": "Unknown"}
    except Exception as e:
        logging.error(f"Error fetching token metadata for CA {ca_address}: {e}")
        raise  # Raise the exception to trigger the retry mechanism

# Update fetch_tokens_continuously to manage WebSocket
async def fetch_tokens_continuously():
=======
                return None
    except Exception as e:
        logging.error(f"Error fetching latest token: {e}")
        return None

# Continuously fetch tokens
async def fetch_tokens_continuously():
    """Continuously fetch latest tokens and analyze trends."""
>>>>>>> origin/main
    global tokens_df
    async with aiohttp.ClientSession() as session:
        sol_price = await fetch_sol_price(session)
        if sol_price == 0.0:
            logging.error("SOL price unavailable. Exiting...")
            return
<<<<<<< HEAD
        
        websocket = await init_websocket_connection()
        if not websocket:
            logging.error("WebSocket connection failed. Continuing with primary API only.")
        
=======

>>>>>>> origin/main
        while True:
            token_data = await fetch_latest_token(session, websocket)
            if token_data:
<<<<<<< HEAD
                # Ensure all necessary columns are in token_data
                for col in columns + ['Description', 'Supply', 'Price']:
                    if col not in token_data:
                        token_data[col] = None  # or appropriate default

                # Convert Market Cap to float if it's valid
                token_data['Market Cap'] = (
                    float(token_data['Market Cap']) if token_data['Market Cap'] != "Unknown" else None
                )

                # Append new token data to the DataFrame
                new_row = pd.DataFrame([token_data])
                tokens_df = pd.concat([tokens_df, new_row], ignore_index=True)
                tokens_df.drop_duplicates(subset=["Name", "Symbol"], inplace=True)


                # Compute Market Cap Change
                if 'Market Cap Change' not in tokens_df.columns:
                    tokens_df['Market Cap Change'] = None  # Initialize the column
                tokens_df['Market Cap Change'] = tokens_df.groupby('Symbol')['Market Cap'].diff()

                # Analyze trends and descriptions
                overall_trends = analyze_trends(tokens_df)  # Includes name and symbol-based analysis
                keyword_insights = analyze_trends_from_descriptions(tokens_df)  # Contextual keyword insights

                # Format the latest token message
                latest_token_message = format_latest_token(token_data, sol_price)

                # Combine messages
                final_message = f"{overall_trends}\n\n{keyword_insights}"
                bot.send_message(PUBLIC_GROUP_ID, final_message, parse_mode="Markdown")

            await asyncio.sleep(60)
=======
                # Convert Market Cap to float if it's valid
                token_data['Market Cap'] = (
                    float(token_data['Market Cap']) if token_data['Market Cap'] != "Unknown" else None
                )
                
                # Append new token data to the DataFrame
                tokens_df = pd.concat([tokens_df, pd.DataFrame([token_data])], ignore_index=True)
                tokens_df.drop_duplicates(subset=["Name", "Symbol"], inplace=True)

                # Compute Market Cap Change
                if 'Market Cap Change' not in tokens_df.columns:
                    tokens_df['Market Cap Change'] = None  # Initialize the column
                
                tokens_df['Market Cap Change'] = tokens_df.groupby('Symbol')['Market Cap'].diff()

                # Generate messages
                latest_token_message = format_latest_token(token_data, sol_price)
                trend_summary = analyze_trends(tokens_df)
                final_message = f"{trend_summary}\n\n{latest_token_message}"
                bot.send_message(PUBLIC_GROUP_ID, final_message, parse_mode="Markdown")

            await asyncio.sleep(35)
>>>>>>> origin/main

# Main execution
if __name__ == "__main__":
    try:
        nltk.download("stopwords")
        asyncio.run(fetch_tokens_continuously())
    except KeyboardInterrupt:
        logging.info("Program interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
