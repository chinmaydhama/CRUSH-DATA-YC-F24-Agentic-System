import ssl
import urllib3
import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
ALLOWED_CHANNELS = os.getenv("ALLOWED_CHANNELS", "")
ALLOWED_USERS = os.getenv("ALLOWED_USERS", "")
allowed_channels_set = set(ALLOWED_CHANNELS.split(",")) if ALLOWED_CHANNELS else set()
allowed_users_set = set(ALLOWED_USERS.split(",")) if ALLOWED_USERS else set()
if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    raise ValueError("Slack tokens missing in .env (SLACK_BOT_TOKEN, SLACK_APP_TOKEN).")

# 3) Create Slack Bolt App
app = App(token=SLACK_BOT_TOKEN)
@app.event("message")
def handle_message_events(event, say, logger):
    try:
        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "").strip()
        if not text:
            return
        if allowed_channels_set and (channel_id not in allowed_channels_set):
            logger.info(f"Ignoring message in channel {channel_id} (not allowed).")
            return
        if allowed_users_set and (user_id not in allowed_users_set):
            logger.info(f"Ignoring message from user {user_id} (not allowed).")
            return
        say(f"Received your message: '{text}'")
    except Exception as e:
        logger.error(f"Error handling message event: {e}")

if __name__ == "__main__":
    print("[INFO] Starting Slack Agent with globally disabled SSL verification.")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
