from line_bot_api import *


def about_us_event(event):
    emojis = [
        {
            "index": 0,
            "productId": "5ac21a18040ab15980c9b43e",
            "emojiId": "009"
        },
        {
            "index": 12,
            "productId": "5ac21a18040ab15980c9b43e",
            "emojiId": "014"
        }
    ]

    info_message = TextSendMessage(
        text="$ 史都客 stock $\n"
             "我被設定為財經小幫手~\n"
             "我的功能有：\n"
             "股票查詢、指標選股、\n股票清單、股票預測\n"
             "✔股票查詢:\n按下後輸入欲查詢的股票代號即可查詢相關資訊\n\n"
             "✔超錢股(指標選股):\n按下後選擇想要股票的指標即會排出該排名前15筆\n\n"
             "✔股票清單:\n輸入'加'+'股票代號'，即可儲存想關注的股票"
             "再點選圖文選單中的股票清單即會呈現先前關注的股票\n\n"
             "✔股票預測:\n能夠預測未來股市價格\n\n"
             "✔使用說明:\n選單中及記事本裡都有詳細的使用說明\n\n"
             "使用上有任何問題可以參考使用說明",
        emojis=emojis
    )

    sticker_message = StickerSendMessage(
        package_id="11537", sticker_id="52002735"
    )

    line_bot_api.reply_message(
        event.reply_token,
        [info_message, sticker_message]
    )

