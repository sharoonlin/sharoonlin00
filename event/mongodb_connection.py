from pymongo import MongoClient
import datetime
from dotenv import load_dotenv
import os

stockDB = "line_bot_project"
# collection = "stock"


def constructor_stock():
    """
    建立MongoDB的連線
    :return: db
    """
    load_dotenv()
    key = os.environ['MONGODB_URI']
    client = MongoClient(key)
    db = client[stockDB]
    return db


def load_stock_market(code):
    """
    處理上市、上櫃分類
    :param: code: 股票代碼
    :return: db collect
    """
    db = constructor_stock()

    collect = db["market_type"]
    company = collect.find({"代號": f"{code}"})

    return company


def show_stock_setting(user_name, user_id):
    """
    抓出使用這目前的關注清單
    :param user_name: 使用者名稱
    :param user_id: 使用者ID
    :return:
    """
    db = constructor_stock()

    collect = db[user_name]
    dataList = list(collect.find({'userID': user_id}))

    if not dataList: return '您的股票清單為空，請透過指令新增股票至清單中'

    # content = '您清單中的選股條件為: \n'
    # for i in range(len(dataList)):
    #     content += f"{dataList[i]['favorite_stock']} {dataList[i]['condition']} {dataList[i]['price']}\n"
    # print(user_name, user_id)

    return dataList


def save_my_stock(user_id, user_name, code, condition=None, target_price=None):
    """
    新增使用者的股票
    :param user_id: 使用者ID
    :param user_name: 使用者名稱
    :param code: 股票代號
    :param condition: 到價通知的條件，預設：None
    :param target_price: 指定價格，預設：None
    :return: 加入成功訊息
    """
    db = constructor_stock()

    info_collect = db["market_type"]
    information = info_collect.find_one({"代號": code})

    collect = db[user_name]
    is_exist = collect.find_one({"favorite_stock": code})

    if is_exist != None:
        content = remind_my_stock(user_name, code, condition, target_price)
        return content
    else:
        collect.insert_one(
            {
                "userID": user_id,
                "favorite_stock": code,
                "company_name": information["簡稱"],
                "market": information["上市/櫃"],
                "condition": condition,
                "price": target_price,
                "tag": "stock",
                "date_info": datetime.datetime.now()
            }
        )

        return f"{code}已新增至您的股票清單"


def remind_my_stock(user_name, code, condition, target_price):
    """
    更新暫存的股票名稱
    :param user_name: 使用者ID
    :param code: 使用者名稱
    :param condition: 到價通知的條件，預設：None
    :param target_price: 指定價格，預設：None
    :return: 更新成功訊息
    """
    db = constructor_stock()
    collection = db[user_name]
    collection.update_many(
        {"favorite_stock": code},
        {"$set": {"condition": condition, "price": target_price}}
    )
    content = f"股票{code}更新成功"

    return content


def delete_my_stock(user_name, code):
    db = constructor_stock()

    collect = db[user_name]
    collect.delete_one({"favorite_stock": code})

    return code + "刪除成功"


if __name__ == "__main__":
    user_name = "苗金儒"
    user_id = "U77e9f3ddd382a56a50801e59221056c9"
    print(show_stock_setting(user_name, user_id))
