from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler, exceptions
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
import os
from dotenv import load_dotenv

load_dotenv()
access_token = os.environ['ACCESS_TOKEN']
channel_secret = os.environ['CHANNEL_SECRET']

# print(access_token)
# print(channel_secret)

line_bot_api = LineBotApi(access_token)

handler = WebhookHandler(channel_secret)
