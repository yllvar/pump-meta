# This is just a method to get chatId

import telebot

TELEGRAM_TOKEN = "your-telegram-token"
bot = telebot.TeleBot(TELEGRAM_TOKEN)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, f"Your Chat ID is: {message.chat.id}")

bot.polling()
