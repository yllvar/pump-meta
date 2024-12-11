import aiohttp
import websockets
import json
import pandas as pd
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed
from config import PUMP_PORTAL_WS, LATEST_COINS_API, SOL_PRICE_API, TRADE_API

class DataFetcher:
    def __init__(self):
        self.session = None
        self.websocket = None
        columns = ["Name", "Symbol", "Timestamp", "Sentiment", "Sentiment Description", "Market Cap", "CA Address", "USD Market Cap", "Image URI", "Description", "Supply", "Price", "Market Cap Change"]
        self.tokens_df = pd.DataFrame(columns=columns)
        self.sol_price = 0.0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        self.websocket = await self.init_websocket_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()

    async def init_websocket_connection(self):
        try:
            return await websockets.connect(PUMP_PORTAL_WS)
        except Exception as e:
            logging.error(f"Failed to connect to PumpPortal WebSocket: {e}")
            return None

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    async def fetch_sol_price(self):
        try:
            async with self.session.get(SOL_PRICE_API) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get("solPrice", 0.0))
                else:
                    logging.error(f"Failed to fetch SOL price. Status: {response.status}")
                    return 0.0
        except Exception as e:
            logging.error(f"Error fetching SOL price: {e}")
            raise

    async def fetch_trade_data(self):
        try:
            async with self.session.get(TRADE_API) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logging.error(f"Error fetching trade data: {e}")
        return {}

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    async def fetch_latest_token(self):
        try:
            async with self.session.get(LATEST_COINS_API) as response:
                if response.status == 200:
                    data = await response.json()
                    token_data = self.process_token_data(data)
                    if token_data['Market Cap'] == 'Unknown':
                        raise ValueError("Market Cap is Unknown")
                    
                    metadata = await self.fetch_token_metadata(token_data['CA Address'])
                    token_data.update(metadata)
                    
                    self.update_tokens_df(token_data)
                    return token_data
                else:
                    logging.error(f"Failed to fetch token. Status: {response.status}")
                    raise ValueError("Failed to fetch token")
        except Exception as e:
            logging.warning(f"Primary API failed: {e}. Attempting WebSocket backup.")
            return await self.fetch_token_from_websocket()

    async def fetch_token_from_websocket(self):
        if self.websocket:
            try:
                await self.subscribe_to_events(self.websocket)
                async for message in self.websocket:
                    msg_data = json.loads(message)
                    if msg_data.get('method') == 'tokenTrade':
                        return self.process_websocket_data(msg_data)
            except Exception as ws_e:
                logging.error(f"WebSocket fetch failed: {ws_e}")
        return None

    @staticmethod
    def process_token_data(data):
        return {
            "Name": data.get("name", "Unknown"),
            "Symbol": data.get("symbol", "Unknown"),
            "Timestamp": DataFetcher.format_timestamp(data.get("created_timestamp", 0)),
            "Market Cap": data.get("market_cap", "Unknown"),
            "USD Market Cap": data.get("usd_market_cap", "Unknown"),
            "CA Address": data.get("mint", "Unknown"),
            "Image URI": data.get("image_uri", ""),
        }

    @staticmethod
    def format_timestamp(timestamp):
        try:
            return datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logging.error(f"Error formatting timestamp: {e}")
            return "Invalid Timestamp"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    async def fetch_token_metadata(self, ca_address):
        url = f"https://frontend-api.pump.fun/coins/{ca_address}"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "Description": data.get("description", "No description available."),
                        "Supply": data.get("total_supply", "Unknown"),
                        "Price": data.get("market_cap", 0) / data.get("total_supply", 1) if data.get("total_supply", 1) > 0 else "Unknown",
                        "telegram": data.get("telegram", "https://t.me/pumpfun"),  # Default if not found
                        "trading_page": f"https://pump.fun/trades/{ca_address}",  # Constructed link
                    }
                elif response.status == 500:
                    logging.warning(f"Server error (500) while fetching metadata for CA {ca_address}. Skipping...")
                    return {
                        "Description": "Server error: Metadata unavailable.",
                        "Supply": "Unknown",
                        "Price": "Unknown",
                        "telegram": "https://t.me/pumpfun",  # Default fallback
                        "trading_page": f"https://pump.fun/trades/{ca_address}",
                    }
                else:
                    logging.error(f"Failed to fetch token metadata for CA {ca_address}. Status: {response.status}")
                    return {
                        "Description": "Metadata fetch failed.",
                        "Supply": "Unknown",
                        "Price": "Unknown",
                        "telegram": "https://t.me/pumpfun",  # Default fallback
                        "trading_page": f"https://pump.fun/trades/{ca_address}",
                    }
        except Exception as e:
            logging.error(f"Error fetching token metadata for CA {ca_address}: {e}")
            raise


    def update_tokens_df(self, token_data):
        new_row = pd.DataFrame([token_data], columns=self.tokens_df.columns)
        self.tokens_df = pd.concat([self.tokens_df, new_row], ignore_index=True)
        self.tokens_df.drop_duplicates(subset=["Name", "Symbol"], keep="last", inplace=True)
        
        # Ensure 'Market Cap' is float
        self.tokens_df['Market Cap'] = pd.to_numeric(self.tokens_df['Market Cap'], errors='coerce')
        
        # Calculate Market Cap Change
        self.tokens_df['Market Cap Change'] = self.tokens_df.groupby('Symbol')['Market Cap'].diff()

    @staticmethod
    async def subscribe_to_events(websocket, token_address=None):
        if token_address:
            payload = {"method": "subscribeTokenTrade", "keys": [token_address]}
        else:
            payload = {"method": "subscribeNewToken"}
        await websocket.send(json.dumps(payload))

    @staticmethod
    def process_websocket_data(msg_data):
        return {
            "Name": msg_data.get("name", "Unknown"),
            "Symbol": msg_data.get("symbol", "Unknown"),
            "Timestamp": DataFetcher.format_timestamp(msg_data.get("timestamp", 0)),
            "Market Cap": msg_data.get("vSolInBondingCurve", "Unknown"),
            "USD Market Cap": msg_data.get("marketCapSol", "Unknown"),
            "CA Address": msg_data.get("mint", "Unknown"),
            "Image URI": msg_data.get("image_uri", ""),
            "Description": msg_data.get("description", "No description available."),
        }