#  -*- coding: utf-8 -*-

from pymongo import ASCENDING
from handle_data.database_utils import DB_CONN
from datetime import datetime, timedelta
import pandas as pd


def pro_get_all_codes(date=None):
    """
    get the new code
    :param date:
    :return:
    """
    datetime_obj = datetime.now()

    fmt = "%Y%m%d"
    if date is None:
        date = datetime_obj.strftime(fmt)

    codes = []
    while len(codes)==0:
        code_cursor = DB_CONN.pro_basic.find(
            {"trade_date":date},
            projection={"ts_code":True,"_id":False}
        )
        codes = [x['ts_code'] for x in code_cursor]
        datetime_obj = datetime_obj - timedelta(days=1)
        date = datetime_obj.strftime(fmt)
    return codes

def get_stock_code_all_data(code):
    """
    get stock code all data
    :param code:
    :return:
    """
    datas = []
    code_datas = DB_CONN.pro_daily.find(
        {"ts_code":code,"index":False},
        sort=[('trade_date', ASCENDING)],
        projection={"_id":False,"trade_date":True,"close":True,"high":True,"low":True,"open":True,"pre_close":True,"au_factor":True}
    )
    print()
    for ele in code_datas:
        datas.append(ele)
    return pd.DataFrame(data=datas)



if __name__ == '__main__':
    codes = pro_get_all_codes()
    # for code in codes:
    #     print(code)
    code = "603999.SH"
    data = get_stock_code_all_data(code)
    print(data)