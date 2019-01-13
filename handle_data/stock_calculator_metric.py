#  -*- coding: utf-8 -*-
from handle_data.stock_utils import get_stock_code_all_data,pro_get_all_codes


def metric_calculator(code):
    code_data = get_stock_code_all_data(code)
    metric_mean_or_std(code_data,'close','close_mean10',10,"mean")
    metric_mean_or_std(code_data,'close','close_std10',10,"std")
    cci(code_data,'close','cci',20)
    atr(code_data,'high','low','pre_close','atr',14)
    boll(code_data,"close")
    mtm(code_data,'close',12)
    roc(code_data,'close',5)
    return code_data

def show_data(data):
    for index in data.index:
        print(dict(data.loc[index]))

def metric_mean_or_std(data_frame,file_name,mean_name,num,type):
    rolling = data_frame[file_name].rolling(num)
    if(type=="mean"):
        cal_num = rolling.mean()
    elif type=="std":
        cal_num = rolling.std()
    else:
        raise AttributeError
    data_frame[mean_name] = cal_num

def cci(stock_data_df,field_name,metric_name,num):
    max_ = stock_data_df[field_name].rolling(num).max()
    min_ = stock_data_df[field_name].rolling(num).min()
    stock_data_df['max_'] = max_
    stock_data_df['min_'] = min_

    metric_mean_or_std(stock_data_df, field_name,'temp_mean_', num, 'mean')
    metric_mean_or_std(stock_data_df, field_name, 'temp_std_',num, 'std')
    stock_data_df[metric_name] = ((stock_data_df['max_'] + stock_data_df['min_'] + stock_data_df[field_name])/3 - stock_data_df['temp_mean_']) / (stock_data_df['temp_std_'] * 0.15)
    stock_data_df.drop(["max_","min_","temp_mean_","temp_std_"],axis=1,inplace=True)

def atr(stock_data,high_key,low_key,pre_close_key,atr_name,atr_num):
    i = 0
    trs = []
    before_close = 0
    for index in stock_data.index:
        hi = stock_data.loc[index][high_key]
        li = stock_data.loc[index][low_key]
        trs.append(max(abs((hi-li)),abs(before_close-hi),abs(before_close-li)))
        before_close = stock_data.loc[index][pre_close_key]
    stock_data['tr'] = trs

    stock_data[atr_name]= stock_data['tr'].rolling(atr_num).mean()
    #stock_data[matr_name] = stock_data[atr_name].rolling(matr_num).mean()
def boll(stock_data_df,field_name):

    stock_data_df['temp_ma'] = stock_data_df[field_name].rolling(4).mean()
    stock_data_df['temp_md'] = stock_data_df[field_name].rolling(5).std()

    stock_data_df['temp_ma'] = stock_data_df['temp_ma'].shift(1)

    stock_data_df['up'] = stock_data_df['temp_ma'] + 2 * stock_data_df['temp_md']
    stock_data_df['dn'] = stock_data_df['temp_ma'] - 2 * stock_data_df['temp_md']
    stock_data_df.drop(["temp_ma", "temp_md"], axis=1, inplace=True)


def mtm(stock_data_df,field_name,before_day):


    temp_data = []
    indexs = stock_data_df.index
    for index in stock_data_df.index:
        current_close = stock_data_df.loc[index][field_name]
        ago_index = index - before_day

        if ago_index in indexs:
            ago_close = stock_data_df.loc[ago_index][field_name]
            temp_data.append(current_close-ago_close)
        else:
            temp_data.append(None)
    stock_data_df['mtm'] = temp_data


def roc(stock_data_df,field_name,before_day):

    indexs = stock_data_df.index
    temp_data = []
    for index in indexs:
        current_close = stock_data_df.loc[index][field_name]
        ago_index = index - before_day
        if ago_index in indexs:
            ago_close = stock_data_df.loc[ago_index][field_name]
            if ago_close !=0:
                rate = (current_close - ago_close)/ago_close
            else:
                rate = 0
        else:
            rate = None
        temp_data.append(rate)
    stock_data_df['roc'] = temp_data



def macd(stock_data_df, cal_key, short, long, m):
    ema1 = []
    ema2 = []
    i = 0
    for index in stock_data_df.index:
        close = stock_data_df.loc[index][cal_key]
        if i==0:
            ema1.append(close)
            ema2.append(close)
        else:
            ema1.append((close-ema1[i-1])*2/(short+1) + ema1[i-1])
            ema2.append((close-ema2[i-1])*2/(long+1) + ema2[i-1])
        i+=1
    stock_data_df['ema1'] = ema1
    stock_data_df['ema2'] = ema2
    stock_data_df['diff'] = stock_data_df['ema1'] - stock_data_df['ema2']
    i = 0
    diff = []
    for index in stock_data_df.index:
        close = stock_data_df.loc[index]['diff']
        if i == 0:
            diff.append(close)
        else:
            diff.append((close-diff[i-1])*2/(m+1)+diff[i-1])
        i+=1
    stock_data_df['dea'] = diff
    stock_data_df['delta'] = stock_data_df['diff'] - stock_data_df['dea']

def wpr(stock_data_df):
    num = 12
    stock_data_df['max_'] = stock_data_df['f2'].rolling(num).max()
    stock_data_df['min_'] = stock_data_df['f2'].rolling(num).min()

    stock_data_df['wpr'] = (stock_data_df['max_'] - stock_data_df['f2']) / (stock_data_df['max_'] - stock_data_df['min_'])

if __name__ == '__main__':
    metric_calculator()