import requests
import twstock
import pandas as pd
import numpy as np
import datetime
import json
from bs4 import BeautifulSoup


def fetch_stock_basic_info(code):
    """
    從twstock.codes取得股票資訊
    :param code: 股票代號
    :return: {"股票代號": "", "名稱": "", "上市/櫃日期": "", "上市/櫃": "", "類別": ""}
    """
    try:
        info = twstock.codes[f'{code}']
        useful = {
            "股票代號": info[1],
            "名稱": info[2],
            "上市/櫃日期": info[4],
            "上市/櫃": info[5],
            "類別": info[6]
        }
        return useful
    except KeyError:
        return None


def fetch_twse_3institution(code):
    """
    抓取上市股票三大法人買賣超資訊
    :param code: 股票代號
    :return:
    """
    today = datetime.datetime.now()
    # today = datetime.date(2023, 8, 5)
    # print(today)
    # print(today.weekday())
    if today.weekday() == 5:
        today -= datetime.timedelta(days=1)
    elif today.weekday() == 6:
        today -= datetime.timedelta(days=2)
    else:
        if today.hour < 15:
            today -= datetime.timedelta(days=1)

    print(today.strftime('%Y%m%d'))

    query_url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={today.strftime('%Y%m%d')}&selectType=ALL&response=json&_=1691826664719"

    res = requests.get(query_url)

    inv_json = res.json()
    # print(inv_json)

    df = pd.DataFrame.from_dict(inv_json["data"])

    df.columns = ['證券代號', '證券名稱',
                  '外陸資買進股數(不含外資自營商)', '外陸資賣出股數(不含外資自營商)', '外陸資買賣超股數(不含外資自營商)',
                  '外資自營商買進股數', '外資自營商賣出股數', '外資自營商買賣超股數',
                  '投信買進股數', '投信賣出股數', '投信買賣超股數', '自營商買賣超股數',
                  '自營商買進股數(自行買賣)', '自營商賣出股數(自行買賣)', '自營商買賣超股數(自行買賣)',
                  '自營商買進股數(避險)', '自營商賣出股數(避險)', '自營商買賣超股數(避險)', '三大法人買賣超股數']

    filtered_df = df[df["證券代號"] == code]
    # print(filtered_df)

    target_columns = ['證券代號', '證券名稱', '外陸資買進股數(不含外資自營商)', '外陸資賣出股數(不含外資自營商)',
                      '外陸資買賣超股數(不含外資自營商)', '投信買進股數', '投信賣出股數', '投信買賣超股數',
                      '自營商買賣超股數',
                      '自營商買進股數(自行買賣)', '自營商賣出股數(自行買賣)', '自營商買賣超股數(自行買賣)',
                      '自營商買進股數(避險)', '自營商賣出股數(避險)', '自營商買賣超股數(避險)'
                      ]

    filtered_df_todict = filtered_df[target_columns].to_dict('records')[0]

    process_dict = {
        '日期': today.strftime('%Y/%m/%d'),
        '證券代號': filtered_df_todict["證券代號"],
        '證券名稱': filtered_df_todict["證券名稱"].replace(" ", ""),
        '外資買入張數': f'{int(np.round((float(filtered_df_todict["外陸資買進股數(不含外資自營商)"].replace(",", "")) / 1000)))}張',
        '外資賣出張數': f'{int(np.round((float(filtered_df_todict["外陸資賣出股數(不含外資自營商)"].replace(",", "")) / 1000)))}張',
        '外資買賣超': f'{int(np.round((float(filtered_df_todict["外陸資買賣超股數(不含外資自營商)"].replace(",", "")) / 1000)))}張',
        '投信買入張數': f'{int(np.round((float(filtered_df_todict["投信買進股數"].replace(",", "")) / 1000)))}張',
        '投信賣出張數': f'{int(np.round((float(filtered_df_todict["投信賣出股數"].replace(",", "")) / 1000)))}張',
        '投信買賣超': f'{int(np.round((float(filtered_df_todict["投信買賣超股數"].replace(",", "")) / 1000)))}張',
        '自營商買入張數': f'{int(np.round((float(filtered_df_todict["自營商買進股數(自行買賣)"].replace(",", "")) + float(filtered_df_todict["自營商買進股數(避險)"].replace(",", ""))) / 1000))}張',
        '自營商賣出張數': f'{int(np.round((float(filtered_df_todict["自營商賣出股數(自行買賣)"].replace(",", "")) + float(filtered_df_todict["自營商賣出股數(避險)"].replace(",", ""))) / 1000))}張',
        '自營商買賣超': f'{int(np.round((float(filtered_df_todict["自營商買賣超股數"].replace(",", "")) / 1000)))}張',
        '自營商(避險)買入張數': f'{int(np.round((float(filtered_df_todict["自營商買進股數(避險)"].replace(",", "")) / 1000)))}張',
        '自營商(避險)賣出張數': f'{int(np.round((float(filtered_df_todict["自營商賣出股數(避險)"].replace(",", "")) / 1000)))}張',
        '自營商(避險)買賣超': f'{int(np.round((float(filtered_df_todict["自營商買賣超股數(避險)"].replace(",", "")) / 1000)))}張',
    }

    return process_dict


def fetch_tpex_3institution(code):
    """
    抓取上櫃股票三大法人買賣超資訊
    :param code: 股票代號
    :return:
    """
    query_url = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&se=EW&t=D&_=1691917586205"

    df = pd.read_json(query_url)
    # print(len(df["aaData"]))

    date = df["reportDate"]

    columns_list = ['證券代號', '證券名稱',
                    '外陸資買進股數(不含外資自營商)', '外陸資賣出股數(不含外資自營商)', '外陸資買賣超股數(不含外資自營商)',
                    '外資自營商買進股數', '外資自營商賣出股數', '外資自營商買賣超股數',
                    '外資及陸資買進股數', '外資及陸資賣出股數', '外資及陸資買賣超股數',
                    '投信買進股數', '投信賣出股數', '投信買賣超股數',
                    '自營商買進股數(自行買賣)', '自營商賣出股數(自行買賣)', '自營商買賣超股數(自行買賣)',
                    '自營商買進股數(避險)', '自營商賣出股數(避險)', '自營商買賣超股數(避險)',
                    '自營商買進股數', '自營商賣出股數', '自營商買賣超股數',
                    '三大法人買賣超股數', "EE"
                    ]

    processed_df = pd.DataFrame(columns=columns_list)
    # print(processed_df)

    # for i in zip(columns_list, df["aaData"][0]):
    #     print(i)
    # print(len(df["aaData"][0]))
    #
    # print(len(columns_list))

    for i, row in df.iterrows():
        nd_array = np.array(row["aaData"])
        nd_array.reshape((1, 25))

        processed_df.loc[len(processed_df.index)] = nd_array

    # print(processed_df.to_string())

    filtered_df = processed_df[processed_df["證券代號"] == code]
    # print(filtered_df.to_string())

    target_columns = ['證券代號', '證券名稱',
                      '外陸資買進股數(不含外資自營商)', '外陸資賣出股數(不含外資自營商)', '外陸資買賣超股數(不含外資自營商)',
                      '投信買進股數', '投信賣出股數', '投信買賣超股數',
                      '自營商買進股數(自行買賣)', '自營商賣出股數(自行買賣)', '自營商買賣超股數(自行買賣)',
                      '自營商買進股數(避險)', '自營商賣出股數(避險)', '自營商買賣超股數(避險)',
                      '自營商買賣超股數',
                      ]

    filtered_df_todict = filtered_df[target_columns].to_dict('records')[0]
    # print(filtered_df_todict)

    process_dict = {
        '日期': date[0],
        '證券代號': filtered_df_todict["證券代號"],
        '證券名稱': filtered_df_todict["證券名稱"].replace(" ", ""),
        '外資買入張數': f'{int(np.round((float(filtered_df_todict["外陸資買進股數(不含外資自營商)"].replace(",", "")) / 1000)))}張',
        '外資賣出張數': f'{int(np.round((float(filtered_df_todict["外陸資賣出股數(不含外資自營商)"].replace(",", "")) / 1000)))}張',
        '外資買賣超': f'{int(np.round((float(filtered_df_todict["外陸資買賣超股數(不含外資自營商)"].replace(",", "")) / 1000)))}張',
        '投信買入張數': f'{int(np.round((float(filtered_df_todict["投信買進股數"].replace(",", "")) / 1000)))}張',
        '投信賣出張數': f'{int(np.round((float(filtered_df_todict["投信賣出股數"].replace(",", "")) / 1000)))}張',
        '投信買賣超': f'{int(np.round((float(filtered_df_todict["投信買賣超股數"].replace(",", "")) / 1000)))}張',
        '自營商買入張數': f'{int(np.round((float(filtered_df_todict["自營商買進股數(自行買賣)"].replace(",", "")) + float(filtered_df_todict["自營商買進股數(避險)"].replace(",", ""))) / 1000))}張',
        '自營商賣出張數': f'{int(np.round((float(filtered_df_todict["自營商賣出股數(自行買賣)"].replace(",", "")) + float(filtered_df_todict["自營商賣出股數(避險)"].replace(",", ""))) / 1000))}張',
        '自營商買賣超': f'{int(np.round((float(filtered_df_todict["自營商買賣超股數"].replace(",", "")) / 1000)))}張',
        '自營商(避險)買入張數': f'{int(np.round((float(filtered_df_todict["自營商買進股數(避險)"].replace(",", "")) / 1000)))}張',
        '自營商(避險)賣出張數': f'{int(np.round((float(filtered_df_todict["自營商賣出股數(避險)"].replace(",", "")) / 1000)))}張',
        '自營商(避險)買賣超': f'{int(np.round((float(filtered_df_todict["自營商買賣超股數(避險)"].replace(",", "")) / 1000)))}張',
    }

    return process_dict


def fetch_instance_price(code):
    """

    :param code:
    :return:
    """
    query_url = f"https://scantrader.com/v2/stock/{code}"

    response = requests.get(query_url)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.select("ul li p")
    # print(table)

    price_info = []
    for i, content in enumerate(table):
        # print(i)
        # print(content.text.replace(" ", "").strip("\n"))
        price_info.append(content.text.replace(" ", "").strip("\n"))

    return price_info


def PE():
    query_url = 'https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL'
    response = requests.get(query_url)
    data = response.json()
    df = pd.DataFrame(data)
    column_mapping = {
        'Code': '股票代號',
        'Name': '公司名稱',
        'PEratio': '本益比',
        'DividendYield': '殖利率',
        'PBratio': '股價淨值比'
    }
    df = df.rename(columns=column_mapping)
    filter_ = df["本益比"] != ""
    sorted_df = df[["股票代號", "公司名稱", "本益比"]].where(filter_).sort_values(by='本益比', ascending=True).head(15)
    # 設定每個欄位的寬度，並使用 str.format() 調整對齊
    # format_string = "{:<15} {:<15} {:>15}"
    # header = format_string.format("股票代號", "公司名稱", "本益比")
    # result_text = [header]
    # for index, row in sorted_df.iterrows():
    #     formatted_row = format_string.format(row["股票代號"], row["公司名稱"], row["本益比"])
    #
    #     result_text.append(formatted_row)

    # return "\n".join(result_text)

    top_10_peratio = sorted_df

    result_text = f'上市Top 15\n{"股票代號":<7}{"公司名稱":<7}{"本益比":}\n'
    for i, row in top_10_peratio.iterrows():
        result_text += f'{row["股票代號"]:<10}{row["公司名稱"]:<8}{row["本益比"]}\n'

    return result_text


def DY():
    # 列出殖利率最高前15
    query_url = f'https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL'
    response = requests.get(query_url)
    data = response.json()
    df = pd.DataFrame(data)
    column_mapping = {
        'Code': '股票代號',
        'Name': '公司名稱',
        'PEratio': '本益比',
        'DividendYield': '殖利率',
        'PBratio': '股價淨值比'
    }
    df = df.rename(columns=column_mapping)
    sorted_df = df.sort_values(by='殖利率', ascending=False).head(15)
    # format_string = "{:<15} {:<15} {:>15}"
    # header = format_string.format("股票代號", "公司名稱", "殖利率")
    # result_text = [header]
    #
    # for index, row in sorted_df.iterrows():
    #     formatted_row = format_string.format(row["股票代號"], row["公司名稱"], row["殖利率"])
    #
    #     result_text.append(formatted_row)
    #
    # return "\n".join(result_text)

    top_10_yield = sorted_df

    result_text = f'上市Top 15\n{"股票代號":<7}{"公司名稱":<7}{"殖利率"}\n'
    for i, row in top_10_yield.iterrows():
        result_text += f'{row["股票代號"]:<10}{row["公司名稱"]:<8}{row["殖利率"]}\n'

    return result_text


def PB():
    query_url = 'https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL'
    response = requests.get(query_url)
    data = response.json()
    df = pd.DataFrame(data)
    column_mapping = {
        'Code': '股票代號',
        'Name': '公司名稱',
        'PEratio': '本益比',
        'DividendYield': '殖利率',
        'PBratio': '股價淨值比'
    }
    df = df.rename(columns=column_mapping)
    sorted_df = df.sort_values(by='股價淨值比', ascending=True).head(15)
    # 設定每個欄位的寬度，並使用 str.format() 調整對齊
    # format_string = "{:<15} {:<15} {:>15}"
    # header = format_string.format("股票代號", "公司名稱", "股價淨值比")
    # result_text = [header]
    # for index, row in sorted_df.iterrows():
    #     formatted_row = format_string.format(row["股票代號"], row["公司名稱"], row["股價淨值比"])
    #
    #     result_text.append(formatted_row)
    #
    # return "\n".join(result_text)

    top_10_pbratio = sorted_df

    result_text = f'上市Top 15\n{"股票代號":<7}{"公司名稱":<7}{"股價淨值比"}\n'
    for i, row in top_10_pbratio.iterrows():
        result_text += f'{row["股票代號"]:<10}{row["公司名稱"]:<8}{row["股價淨值比"]}\n'

    return result_text


def PE2():
    query_url = f'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis'
    response = requests.get(query_url)
    data = response.json()
    df = pd.DataFrame(data)
    column_mapping = {
        'Date': '資料日期',
        'SecuritiesCompanyCode': '股票代號',
        'CompanyName': '公司名稱',
        'PriceEarningRatio': '本益比',
        'DividendPerShare': '每股股利',
        'YieldRatio': '殖利率',
        'PriceBookRatio': '股價淨值比'
    }
    df = df.rename(columns=column_mapping)
    filter_ = df["本益比"] != ""
    sorted_df = df[["股票代號", "公司名稱", "本益比"]].where(filter_).sort_values(by='本益比', ascending=True).head(15)
    # format_string = "{:<10} {:<10} {:>10}"
    # header = format_string.format("股票代號", "公司名稱", "本益比")
    # result_text = [header]
    # for index, row in sorted_df.iterrows():
    #     formatted_row = format_string.format(row["股票代號"], row["公司名稱"], row["本益比"])
    #
    #     result_text.append(formatted_row)
    #
    # return ("\n".join(result_text))

    top_10_peratio = sorted_df

    result_text = f'上櫃Top 15\n{"股票代號":<7}{"公司名稱":<7}{"本益比":}\n'
    for i, row in top_10_peratio.iterrows():
        result_text += f'{row["股票代號"]:<10}{row["公司名稱"]:<8}{row["本益比"]}\n'

    return result_text


def DY2():
    query_url = f'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis'
    response = requests.get(query_url)
    data = response.json()
    df = pd.DataFrame(data)
    column_mapping = {
        'Date': '資料日期',
        'SecuritiesCompanyCode': '股票代號',
        'CompanyName': '公司名稱',
        'PriceEarningRatio': '本益比',
        'DividendPerShare': '每股股利',
        'YieldRatio': '殖利率',
        'PriceBookRatio': '股價淨值比'
    }
    df = df.rename(columns=column_mapping)
    sorted_df = df.sort_values(by='殖利率', ascending=False).head(15)
    # format_string = "{:<10} {:<10} {:>10}"
    # header = format_string.format("股票代號", "公司名稱", "殖利率")
    # result_text = [header]
    # for index, row in sorted_df.iterrows():
    #     formatted_row = format_string.format(row["股票代號"], row["公司名稱"], row["殖利率"])
    #
    #     result_text.append(formatted_row)
    #
    # return "\n".join(result_text)

    top_10_yield = sorted_df

    result_text = f'上櫃Top 15\n{"股票代號":<7}{"公司名稱":<7}{"殖利率"}\n'
    for i, row in top_10_yield.iterrows():
        result_text += f'{row["股票代號"]:<10}{row["公司名稱"]:<8}{row["殖利率"]}\n'

    return result_text


def PB2():
    query_url = f'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis'
    response = requests.get(query_url)
    data = response.json()
    df = pd.DataFrame(data)
    column_mapping = {
        'Date': '資料日期',
        'SecuritiesCompanyCode': '股票代號',
        'CompanyName': '公司名稱',
        'PriceEarningRatio': '本益比',
        'DividendPerShare': '每股股利',
        'YieldRatio': '殖利率',
        'PriceBookRatio': '股價淨值比'
    }
    df = df.rename(columns=column_mapping)
    sorted_df = df.sort_values(by='股價淨值比', ascending=False).head(15)
    # format_string = "{:<10} {:<10} {:>10}"
    # header = format_string.format("股票代號", "公司名稱", "股價淨值比")
    # result_text = [header]
    # for index, row in sorted_df.iterrows():
    #     formatted_row = format_string.format(row["股票代號"], row["公司名稱"], row["股價淨值比"])
    #
    #     result_text.append(formatted_row)
    #
    # return "\n".join(result_text)

    top_10_pbratio = sorted_df

    result_text = f'上櫃Top 15\n{"股票代號":<7}{"公司名稱":<7}{"股價淨值比"}\n'
    for i, row in top_10_pbratio.iterrows():
        result_text += f'{row["股票代號"]:<10}{row["公司名稱"]:<8}{row["股價淨值比"]}\n'

    return result_text


def fetch_weight_index():
    """

    :return:
    """
    query_url = f'https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?response=json'
    response = requests.get(query_url)
    data = response.json()
    # print(data["data"])

    df = pd.read_csv("歷年台股大盤指數.csv", index_col=0)
    # print(df.tail()["Date"].iloc[-1])
    #
    # print(df.tail()["Date"].iloc[-1] == data["data"][7][0])

    append_data = {
        'Date': [],
        'Open': [],
        'High': [],
        'Low': [],
        'Close': []
    }
    for i in data["data"]:
        if int(df.tail()["Date"].iloc[-1].replace(r"/", "")) < int(i[0].replace(r"/", "")):
            print(True)
            append_data["Date"].append(i[0])
            append_data["Open"].append(i[1])
            append_data["High"].append(i[3])
            append_data["Low"].append(i[3])
            append_data["Close"].append(i[4])
        else:
            print(False)

    # Make data frame of above data
    append_df = pd.DataFrame(append_data)

    # append data frame to CSV file
    append_df.to_csv('歷年台股大盤指數.csv', mode='a', index=True, header=False)

    # print message
    print("Data appended successfully.")


if __name__ == "__main__":
    # code = "2330"
    # print(fetch_twse_3institution(code))
    # print(fetch_tpex_3institution(code))

    # flex_message_dy = DY()
    # flex_message_pe = PE()
    # flex_message_pb = PB()
    #
    # print(flex_message_dy)
    # print(flex_message_pe)
    # print(flex_message_pb)

    # print(fetch_instance_price('2330'))

    fetch_weight_index()
