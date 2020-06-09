'''
選出前五年(2017-2019)每股盈餘>0、營業利益率>0(代表公司有賺錢)、稅後淨利率>0、ROE>5%、負債比率>0、
速動比率>0之公司，並於2019年再以營業活動現金流量(每年)>0、營收成長率(每年)>10%等做篩選。
由於篩選出的公司過多，所以再根據：近五年來每年殖利率>2%、最新殖利率>近五年平均殖利率、最新殖利率>3%、2019全年EPS>2019所發放之現金股利，
進一步篩選，希望能夠減少公司數，根據上述條件與 20200602 的殖利率（只要有新資料就會抓最新的資料），最後篩出的公司有14間，
股票代碼分別為 [1227,2308,2748,3004,3036,3596,3617,4137,4438,4915,5284,5288,9938,9946]，
由於執行程式碼的時間不同，抓下來的殖利率可能是不同天的資料，所以最後篩出的公司跟上述14間公司可能會有些微的變動。
'''

import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import time
import talib
import pandas_datareader.data as web

tej = pd.read_csv('2015-2019比率.csv')
tej.index = tej['代號']
tej.drop(columns=['代號'],inplace=True)
tej['來自營運之現金流量'] = tej['來自營運之現金流量'].apply(lambda x: int(x.replace(',','')) if type(x) == str else x)
tej_2019 = tej[tej['年/月'] == '2019/12']
tej_2018 = tej[tej['年/月'] == '2018/12']
tej_2017 = tej[tej['年/月'] == '2017/12']
tej_2016 = tej[tej['年/月'] == '2016/12']
tej_2015 = tej[tej['年/月'] == '2015/12']

stock2019 = set(tej_2019.index)
stock2018 = set(tej_2018.index)
stock2017 = set(tej_2017.index)
stock2016 = set(tej_2016.index)
stock2015 = set(tej_2015.index)

drop2019 = stock2019.difference(stock2018)|stock2019.difference(stock2017)|stock2019.difference(stock2016)|stock2019.difference(stock2015)
drop2018 = stock2018.difference(stock2019)|stock2018.difference(stock2017)|stock2018.difference(stock2016)|stock2018.difference(stock2015)
drop2017 = stock2017.difference(stock2019)|stock2017.difference(stock2018)|stock2017.difference(stock2016)|stock2017.difference(stock2015)
drop2016 = stock2016.difference(stock2019)|stock2016.difference(stock2018)|stock2016.difference(stock2017)|stock2016.difference(stock2015)
drop2015 = stock2015.difference(stock2019)|stock2015.difference(stock2018)|stock2015.difference(stock2017)|stock2015.difference(stock2016)

df = tej_2015[(tej_2015['營業利益率'] > 0) & (tej_2015['稅後淨利率'] > 0) & (tej_2015['每股盈餘'] > 0) & (tej_2015['ROE(A)－稅後'] > 5)
             & (tej_2015['速動比率'] > 0) & (tej_2015['負債比率'] > 0)]
df = tej_2016.loc[df.index][(tej_2016.loc[df.index]['營業利益率'] > 0) & (tej_2016.loc[df.index]['稅後淨利率'] > 0) & (tej_2016.loc[df.index]['每股盈餘'] > 0) 
                            & (tej_2016.loc[df.index]['ROE(A)－稅後'] > 5) & (tej_2016.loc[df.index]['速動比率'] > 0) & (tej_2016.loc[df.index]['負債比率'] > 0)]
df = tej_2017.loc[df.index][(tej_2017.loc[df.index]['營業利益率'] > 0) & (tej_2017.loc[df.index]['稅後淨利率'] > 0) & (tej_2017.loc[df.index]['每股盈餘'] > 0) 
                            & (tej_2017.loc[df.index]['ROE(A)－稅後'] > 5) & (tej_2017.loc[df.index]['速動比率'] > 0) & (tej_2017.loc[df.index]['負債比率'] > 0)]
df = tej_2018.loc[df.index][(tej_2018.loc[df.index]['營業利益率'] > 0) & (tej_2018.loc[df.index]['稅後淨利率'] > 0) & (tej_2018.loc[df.index]['每股盈餘'] > 0) 
                            & (tej_2018.loc[df.index]['ROE(A)－稅後'] > 5) & (tej_2018.loc[df.index]['速動比率'] > 0) & (tej_2018.loc[df.index]['負債比率'] > 0)]
df = tej_2019.loc[df.index][(tej_2019.loc[df.index]['營業利益率'] > 0) & (tej_2019.loc[df.index]['稅後淨利率'] > 0) & (tej_2019.loc[df.index]['每股盈餘'] > 0) 
                            & (tej_2019.loc[df.index]['ROE(A)－稅後'] > 5) & (tej_2019.loc[df.index]['速動比率'] > 0) & (tej_2019.loc[df.index]['負債比率'] > 0) 
                            & (tej_2019.loc[df.index]['來自營運之現金流量'] > 0) & (tej_2019.loc[df.index]['營收成長率'] > 10)]

'''近五年來每年殖利率>2%'''
df = tej_2015.loc[df.index][(tej_2015.loc[df.index]['股利殖利率'] > 2)]
df = tej_2016.loc[df.index][(tej_2016.loc[df.index]['股利殖利率'] > 2)]
df = tej_2017.loc[df.index][(tej_2017.loc[df.index]['股利殖利率'] > 2)]
df = tej_2018.loc[df.index][(tej_2018.loc[df.index]['股利殖利率'] > 2)]
df = tej_2019.loc[df.index][(tej_2019.loc[df.index]['股利殖利率'] > 2)]

'''爬取每天最新的殖利率，如果今日還未收盤則爬取昨天，如遇未開市日則爬取最後收盤資料'''
def todayDividendyield():
    today = datetime.datetime.today()
    while True:
        try:
            y = str(today.year)
            m = str(today.month)
            if len(m) == 1:
                m = '0' + m
            d = str(today.day)
            url ='https://www.twse.com.tw/exchangeReport/BWIBBU_d?response=html&date=' + y + m + d + '&selectType=ALL'
            r = requests.post(url)
            df = pd.read_html(r.text)[0].fillna("")
            return df
        except ValueError:
            today = today - datetime.timedelta(days = 1)
            time.sleep(5)
            
df1 = todayDividendyield()
df1.columns = df1.columns.droplevel(0)
df2 = df1[['證券代號','殖利率(%)']]
df2 = df2[df2['證券代號'].isin(df.index)]
df2.index = df2['證券代號']
df2.drop(columns=['證券代號'], inplace=True)
df2.columns = ['最新殖利率']
df = pd.concat((df, df2), axis=1, join='inner')

'''最新殖利率>3%'''
'''2019全年EPS>2019所發放之現金股利'''
df = df[(df['最新殖利率'] >3) & (df['每股盈餘'] > df['普通股每股現金股利（盈餘及公積）'])]

'''最新殖利率>近五年平均殖利率'''
df_mean = pd.DataFrame(columns=['五年平均股利殖利率'])
df_mean['五年平均股利殖利率'] = (tej_2015.loc[df.index]['股利殖利率']+tej_2016.loc[df.index]['股利殖利率']+tej_2017.loc[df.index]['股利殖利率']+tej_2018.loc[df.index]['股利殖利率']+tej_2019.loc[df.index]['股利殖利率'])/5
df = pd.concat((df, df_mean), axis=1, join='inner')
df = df[df['最新殖利率']>df['五年平均股利殖利率']]

'''根據上述條件篩選出的公司'''
symbolId = list(df.index)

'''
１．股價　＞　均線 MA10
２．股價　＞　均線 MA20
３．均線 MA10　與　均線 MA20　呈現黃金交叉
'''
output=[]
for stock in symbolId:
    stock = str(stock)
    url = 'https://tw.stock.yahoo.com/q/q?s=' + stock
    list_req = requests.get(url)
    soup = BeautifulSoup(list_req.content, "html.parser")
    get_stock_price= soup.findAll('b')[1].text 
    
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days = 120)

    df2 = web.DataReader([stock+'.TW'], 'yahoo', start, end)
    
    ma20=talib.SMA(df2['Close'][stock+'.TW'],timeperiod=20)
    ma10=talib.SMA(df2['Close'][stock+'.TW'],timeperiod=10)
    avg20 = sum(ma20[-20:].values)/20
    avg10 = sum(ma10[-10:].values)/10
    
    if avg20 < float(get_stock_price):
        if avg10 < float(get_stock_price):
            if avg20 < avg10: #黃金交叉
                output.append(stock)
                
for i in symbolId:
    if str(i) not in output:
        df = df.drop(int(i))
print(output)#最後篩選出來的 '''看df可以看到這幾間的財務比率'''
