import telebot

TELEGRAM_TOKEN = "7851602145:AAFxh70OSmMqStSFqtIrhz94Gs8Dnb4VlDc"
bot = telebot.TeleBot(TELEGRAM_TOKEN)

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, f"Your Chat ID is: {message.chat.id}")

bot.polling()
