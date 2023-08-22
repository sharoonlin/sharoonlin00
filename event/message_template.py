from line_bot_api import *
from event.mongodb_connection import *
from event.request_event import *


def create_buttons_template():
    # 定義 Postback Action 的 data 參數
    postback_data = {
        "action": "button_clicked",
        "input_option": "openKeyboard",
        "fill_in_text": "股票查詢#"
    }

    # 將 data 參數轉換為字串形式
    postback_data_str = "&".join([f"{key}={value}" for key, value in postback_data.items()])

    # 創建一個 Postback Action 按鈕，將 data 參數添加進去
    input_postback_action = PostbackAction(
        label='輸入"#" + "股票代號"',
        data=postback_data_str
    )

    finish_postback_action = PostbackAction(
        label='結束',
        data=postback_data_str
    )

    message_template = TemplateSendMessage(
        alt_text='按鈕範本',
        template=ButtonsTemplate(
            text='請選擇下一步',
            actions=[
                input_postback_action,
                finish_postback_action
            ]
        )
    )
    return message_template


def create_quick_reply():
    message_template = TextSendMessage(
        text="下一步：下方按鈕選擇功能",
        quick_reply=QuickReply(
            items=[
                QuickReplyButton(action=PostbackAction(label="請輸入股票代號", data="data=None",
                                                       input_option="openKeyboard", fill_in_text="股票查詢#")),
                QuickReplyButton(action=MessageAction(label='結束', text='結束')),
            ]
        )
    )

    return message_template


def create_image_carousel_template(user_id, code, url):
    carousel_columns = []

    # 創建 6 個 Image Carousel Columns
    # 1
    image_url_1 = "https://i.imgur.com/3daCk3V.jpg"  # 替換為實際圖片的 URL
    column_1 = ImageCarouselColumn(
        image_url=image_url_1,
        action=URITemplateAction(
            label=f'即時股價 & K線圖',
            uri=url)
    )
    carousel_columns.append(column_1)

    # 2
    basic_postback_data = {
        "action": "image_clicked",
        "data": "fetch_basic_information",
        "userID": user_id,
        "code": code
    }
    postback_data_str = "&".join([f"{key}={value}" for key, value in basic_postback_data.items()])

    image_url = "https://i.imgur.com/3daCk3V.jpg"  # 替換為實際圖片的 URL
    column = ImageCarouselColumn(
        image_url=image_url,
        action=PostbackAction(label="基本資訊", data=postback_data_str)
    )
    carousel_columns.append(column)

    # 創建 Image Carousel Template
    image_carousel_template = ImageCarouselTemplate(columns=carousel_columns)

    return TemplateSendMessage(
        alt_text='Image Carousel Template',
        template=image_carousel_template
    )


def create_flex_template(user_id, code, basic_data):
    """
    建立flex message template
    :param user_id: 使用這ID
    :param code: 股票代碼
    :param basic_data: 基本資訊
    :return: flex message
    """

    trade_market = load_stock_market(code)[0]

    if trade_market["上市/櫃"] == "上市":
        market = "TWSE"
    else:
        market = "TPEX"

    img_list = ["https://i.imgur.com/9X5aSVv.png/img",
                "https://i.imgur.com/ybzwxiX.png/img",
                "https://i.imgur.com/bcbzNaR.png/img",
                "https://i.imgur.com/T9wNGM7.png/img",
                ]
    title_list = ["即時股價日K",
                  "財務總攬",
                  "新聞",
                  "三大法人",
                  ]
    button_action_list = [
        {
            "type": "uri",
            "label": "前往TradingView",
            "uri": f"https://tw.tradingview.com/chart/?symbol={market}%3A{code}"
        },
        {
            "type": "uri",
            "label": "前往財務總攬",
            "uri":  f"https://tw.tradingview.com/symbols/{market}-{code}/financials-overview/"
        },
        {
            "type": "uri",
            "label": "前往相關新聞",
            "uri": f"https://tw.tradingview.com/symbols/{market}-{code}/news/"
        },
        {
            "type": "postback",
            "label": "三大法人彙整",
            "data": f"action=button_clicked&data=fetch_3_insti&userID={user_id}&code={code}&market={trade_market['上市/櫃']}"
        },
    ]
    bg_list = ["#03303Acc", "#9C8E7Ecc", "#03303Acc", "#9C8E7Ecc"]

    flex_content = []
    for i in range(len(img_list)):
        flex_page = {
                    "type": "bubble",
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "image",
                                "url": img_list[i],
                                "size": "full",
                                "aspectMode": "cover",
                                "aspectRatio": "2:3",
                                "gravity": "top"
                            },
                            {
                                "type": "box",
                                "layout": "vertical",
                                "contents": [
                                    {
                                        "type": "box",
                                        "layout": "vertical",
                                        "contents": [
                                            {
                                                "type": "text",
                                                "text": title_list[i],
                                                "size": "xl",
                                                "color": "#ffffff",
                                                "weight": "bold"
                                            },
                                            {
                                                "type": "button",
                                                "action": button_action_list[i],
                                                "style": "primary"
                                            }
                                        ]
                                    }
                                ],
                                "position": "absolute",
                                "offsetBottom": "0px",
                                "offsetStart": "0px",
                                "offsetEnd": "0px",
                                "backgroundColor": bg_list[i],
                                "paddingAll": "20px",
                                "paddingTop": "18px"
                            }
                        ],
                        "paddingAll": "0px"
                    }
                }

        flex_content.append(flex_page)

    if basic_data != None:
        basic_info = []
        for key, value in basic_data.items():
            info = {
                "type": "text",
                "text": f"{key}: {value}",
                "offsetTop": "70px",
                "offsetStart": "4px",
                "size": "22px",
                "color": "#333333",
                "weight": "bold"
            }

            basic_info.append(info)

        text_page = {
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "image",
                        "url": "https://i.imgur.com/QYwU3Zd.png/img",
                        "size": "full",
                        "aspectMode": "cover",
                        "aspectRatio": "2:3",
                        "gravity": "top"
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "box",
                                "layout": "vertical",
                                "contents": [
                                    {
                                        "type": "text",
                                        "text": "基本資訊",
                                        "size": "xl",
                                        "color": "#ffffff",
                                        "weight": "bold"
                                    },
                                    {
                                        "type": "button",
                                        "action": {
                                            "type": "postback",
                                            "label": "傳至聊天室",
                                            "data": f"action=button_clicked&data=fetch_basic_information&userID={user_id}&code={code}"
                                        },
                                        "style": "primary"
                                    }
                                ]
                            }
                        ],
                        "position": "absolute",
                        "offsetBottom": "0px",
                        "offsetStart": "0px",
                        "offsetEnd": "0px",
                        "backgroundColor": "#03303Acc",
                        "paddingAll": "20px",
                        "paddingTop": "18px"
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "contents": basic_info,
                        "position": "absolute",
                        "flex": 0,
                        "width": "264px",
                        "height": "294px",
                        "offsetTop": "18px",
                        "offsetStart": "18px",
                        "cornerRadius": "10px"
                    }
                ],
                "paddingAll": "0px"
            }
        }

        flex_content.append(text_page)

    flex_message = FlexSendMessage(
        alt_text="股票查詢",
        contents={
            "type": "carousel",
            "contents": flex_content
        }
    )

    return flex_message


def show_Button(user_id):
    flex_message = FlexSendMessage(
        alt_text="指標選股",
        contents={
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "指標選股",
                        "weight": "bold",
                        "size": "xl",
                        "color": "#A9D9D0"
                    }
                ],
                "backgroundColor": "#038C7F"
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": "上市",
                        "size": "xl",
                        "color": "#027373",
                        "weight": "bold"
                    },
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "contents": [
                            {
                                "type": "button",
                                "action": {
                                    "type": "postback",
                                    "label": "本益比",
                                    "data": f"action=button_clicked&data=PE&userID={user_id}"
                                },
                                "gravity": "center",
                                "style": "primary",
                                "color": "#038C7F",
                                "margin": "sm"
                            },
                            {
                                "type": "button",
                                "action": {
                                    "type": "postback",
                                    "label": "淨值比",
                                    "data": f"action=button_clicked&data=PB&userID={user_id}"
                                },
                                "gravity": "center",
                                "style": "primary",
                                "color": "#038C7F",
                                "margin": "sm"
                            },
                            {
                                "type": "button",
                                "action": {
                                    "type": "postback",
                                    "label": "殖利率",
                                    "data": f"action=button_clicked&data=DY&userID={user_id}"
                                },
                                "gravity": "center",
                                "style": "primary",
                                "color": "#038C7F",
                                "margin": "sm"
                            }
                        ]
                    },
                    {
                        "type": "separator",
                        "margin": "md",
                        "color": "#D9D0C7"
                    },
                    {
                        "type": "text",
                        "text": "上櫃",
                        "color": "#027373",
                        "weight": "bold",
                        "size": "xl"
                    },
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "contents": [
                            {
                                "type": "button",
                                "action": {
                                    "type": "postback",
                                    "label": "本益比",
                                    "data": f"action=button_clicked&data=PE2&userID={user_id}"
                                },
                                "gravity": "center",
                                "style": "primary",
                                "color": "#038C7F",
                                "margin": "sm"
                            },
                            {
                                "type": "button",
                                "action": {
                                    "type": "postback",
                                    "label": "淨值比",
                                    "data": f"action=button_clicked&data=PB2&userID={user_id}"
                                },
                                "gravity": "center",
                                "style": "primary",
                                "color": "#038C7F",
                                "margin": "sm"
                            },
                            {
                                "type": "button",
                                "action": {
                                    "type": "postback",
                                    "label": "殖利率",
                                    "data": f"action=button_clicked&data=DY2&userID={user_id}"
                                },
                                "gravity": "center",
                                "style": "primary",
                                "color": "#038C7F",
                                "margin": "sm"
                            }
                        ]
                    }
                ],
                "backgroundColor": "#A9D9D0"
            },
            "styles": {
                "header": {
                    "backgroundColor": "#CEECF2"
                }
            }
        }

    )
    return flex_message


def create_stock_list_v2(user_name, user_id):
    """
    建立關注清單的flex message
    :param user_name: 使用這名稱
    :param user_id: 使用者ID
    :return: flex message
    """
    user_list = show_stock_setting(user_name, user_id)

    bubble_list = []
    while len(user_list) > 0:
        flex_stock_list = []
        while len(flex_stock_list) < 4 and len(user_list) != 0:
            user_list_pop = user_list.pop()

            if user_list_pop["market"] == "上市":
                market = "TWSE"
            else:
                market = "TPEX"

            price_data = fetch_instance_price(user_list_pop['favorite_stock'])
            diff = float(price_data[0]) - float(price_data[-1])

            if diff > 0:
                indicator = "▲"
                color = "#FF2A00"
            elif diff < 0:
                indicator = "▼"
                color = "#008408"
            else:
                indicator = "-"
                color = "#393838"

            stock_box = {
                "type": "box",
                "layout": "vertical",
                "margin": "xxl",
                "spacing": "sm",
                "contents": [
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "contents": [
                            {
                                "type": "box",
                                "layout": "vertical",
                                "contents": [
                                    {
                                        "type": "text",
                                        "text": f"{user_list_pop['favorite_stock']}",
                                        "size": "xl",
                                        "color": "#272829",
                                        "flex": 0,
                                        "weight": "bold",
                                        "align": "center"
                                    },
                                    {
                                        "type": "text",
                                        "text": f"{user_list_pop['company_name']}",
                                        "size": "xl",
                                        "weight": "bold",
                                        "align": "center"
                                    }
                                ]
                            },
                            {
                                "type": "box",
                                "layout": "vertical",
                                "contents": [
                                    {
                                        "type": "text",
                                        "text": f"{price_data[0]} {indicator} {price_data[1]}",
                                        "align": "center",
                                        "color": color,
                                        "size": "sm"
                                    },
                                    {
                                        "type": "text",
                                        "text": f"{price_data[2]}",
                                        "align": "center",
                                        "color": color,
                                        "gravity": "top",
                                        "size": "xl"
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "type": "box",
                        "layout": "horizontal",
                        "contents": [
                            {
                                "type": "button",
                                "action": {
                                    "type": "uri",
                                    "label": "查詢詳情",
                                    "uri": f"https://tw.tradingview.com/chart/?symbol={market}%3A{user_list_pop['favorite_stock']}"
                                },
                                "style": "primary",
                                "margin": "2px",
                                "color": "#33BBC5"
                            },
                            {
                                "type": "button",
                                "action": {
                                    "type": "postback",
                                    "label": "移除",
                                    "data": f"action=button_clicked&data=delete_my_stock&name={user_name}&userID={user_id}&code={user_list_pop['favorite_stock']}"
                                },
                                "style": "primary",
                                "color": "#ED7B7B",
                                "margin": "2px"
                            }
                        ]
                    },
                    {
                        "type": "separator",
                        "margin": "lg",
                        "color": "#435B66"
                    }
                ]
            }

            flex_stock_list.append(stock_box)
        bubble_list.append(flex_stock_list)

    stack_stock_box = []
    for i in range(len(bubble_list)):
        stack_stock = {
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": f"關注清單{i+1}",
                        "weight": "bold",
                        "color": "#272829",
                        "size": "xxl",
                        "align": "center"
                    },
                    {
                        "type": "separator",
                        "margin": "xxl"
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "contents": bubble_list[i]
                    }
                ]
            }
        }

        stack_stock_box.append(stack_stock)

    flex_message = FlexSendMessage(
        alt_text="股票查詢",
        contents={
            "type": "carousel",
            "contents": stack_stock_box
        }
    )

    return flex_message


def create_forcast_template():
    URL = "https://6182dac9942401bf92.gradio.live"
    content = {
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": "https://i.imgur.com/3cNkNQ6.gif/",
            "size": "full",
            "aspectRatio": "20:13",
            "aspectMode": "cover",
            "action": {
                "type": "uri",
                "uri": URL
            }
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {
                    "type": "text",
                    "text": "Forcasting",
                    "weight": "bold",
                    "size": "xl"
                }
            ]
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {
                        "type": "uri",
                        "label": "Go",
                        "uri": URL
                    }
                }
            ],
            "flex": 0
        }
    }

    flex_message = FlexSendMessage(
        alt_text="大盤預測",
        contents=content
    )

    return flex_message


if __name__ == "__main__":
    user_name = "苗金儒"
    user_id = "U77e9f3ddd382a56a50801e59221056c9"
    create_stock_list(user_name, user_id)
