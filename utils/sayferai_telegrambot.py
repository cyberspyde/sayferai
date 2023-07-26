import telebot, uuid, logging, threading, os
from flask import Flask, request

TOKEN = os.getenv('SAYFERAI_TELEGRAM_BOT_TOKEN')
passwords = ['password1', 'password2', 'password3']
family_size_limit = 4

bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

user_sessions = {}
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@bot.message_handler(commands=['start'])
def start(message):
    """Handler for the /start command"""
    user_id = message.from_user.id

    if user_id in user_sessions:
        bot.send_message(message.chat.id, 'You are already authenticated.')
    else:
        session_id = str(uuid.uuid4())
        if len(user_sessions) < family_size_limit:
            user_sessions[user_id] = session_id
            bot.send_message(message.chat.id, f'Please enter the password. Session ID: {session_id}')
            logger.info(f'User {user_id} started a new session with ID {session_id}')
        else:
            bot.send_message(message.chat.id, 'The maximum number of people has been reached in this session.')
            logger.warning(f'User {user_id} tried to start a new session but the maximum number of people has been reached.')

@bot.message_handler(func=lambda message: True)
def check_password(message):
    """Handler for checking the entered password"""
    user_id = message.from_user.id
    user_password = message.text.strip()

    if user_id in user_sessions:
        bot.send_message(message.chat.id, 'You are already authenticated.')
        logger.info(f'User {user_id} tried to authenticate again.')
    elif user_password in passwords:
        session_id = user_sessions.get(user_id)
        if session_id:
            bot.send_message(message.chat.id, 'Another user has been authenticated in this session.')
            logger.warning(f'User {user_id} tried to authenticate but another user has already been authenticated in this session.')
        elif len(user_sessions) < family_size_limit:
            session_id = str(uuid.uuid4())
            user_sessions[user_id] = session_id
            bot.send_message(message.chat.id, f'Authentication successful. You are now authenticated. Session ID: {session_id}')
            logger.info(f'User {user_id} authenticated successfully with session ID {session_id}')
        else:
            bot.send_message(message.chat.id, 'The maximum number of people has been reached in this session.')
            logger.warning(f'User {user_id} tried to authenticate but the maximum number of people has been reached in this session.')
    else:
        bot.send_message(message.chat.id, 'Wrong password.')
        logger.warning(f'User {user_id} entered a wrong password.')

def send_emergency_message(message):
    """Send emergency message to all family members"""
    for user_id in user_sessions:
        bot.send_message(user_id, message)

def run_bot():
    # Start the bot
    bot.remove_webhook()
    bot.polling()

def run_webhook():
    # Set up the webhook server
    if __name__ == '__main__':
        app.run()
    
@app.route('/activate_emergency', methods=['POST'])
def activate_emergency():
    if request.method == 'POST':
        print(request.json)
        #synthesize_once("Favqulotda holat aktivlashtirildi, uyni egalariga xabar berilmoqda")
        send_emergency_message("Favqulotda holat aktivlashtirildi")
        return 'Test Success', 200
    else:
        abort(400)

@app.route('/deactivate_emergency', methods=['POST'])
def deactivate_emergency():
    if request.method == 'POST':
        print(request.json)
        #synthesize_once("Favqulotda holat o'chirildi")
        send_emergency_message("Favqulotda holat o'chirildi")
        return 'Test Success', 200
    else:
        abort(400)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_webhook)
    bot_thread = threading.Thread(target=run_bot)

    flask_thread.start()
    bot_thread.start()

    flask_thread.join()
    bot_thread.join()
