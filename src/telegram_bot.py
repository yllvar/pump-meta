import telebot
import logging
from telegram.helpers import escape_markdown

class TelegramBot:
    def __init__(self, token, group_id):
        self.bot = telebot.TeleBot(token)
        self.group_id = group_id

    async def send_message(self, message):
        try:
            # Escape markdown characters
            escaped_message = escape_markdown(message, version=2)
            
            # Log the message content for debugging
            logging.debug(f"Sending message: {escaped_message}")
            
            # Send the message
            self.bot.send_message(self.group_id, escaped_message, parse_mode='MarkdownV2')
        except Exception as e:
            logging.error(f"Error sending Telegram message: {e}")
            try:
                # Try sending without Markdown if parsing fails
                self.bot.send_message(self.group_id, message)
            except Exception as e2:
                logging.error(f"Error sending plain text Telegram message: {e2}")