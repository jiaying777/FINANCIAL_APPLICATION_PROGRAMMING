# FINANCIAL_APPLICATION_PROGRAMMING

[程式碼](https://github.com/jiaying777/FINANCIAL_APPLICATION_PROGRAMMING/blob/master/project.py)<br>
[0612 程式結果](https://nbviewer.jupyter.org/github/jiaying777/FINANCIAL_APPLICATION_PROGRAMMING/blob/master/0612執行結果.ipynb#)<br>

- 選出前五年(2015-2019)滿足以下條件之公司:

    * 每股盈餘>0
    * 營業利益率>0(代表公司有賺錢)
    * 稅後淨利率>0
    * ROE>5%
    * 負債比率>0
    * 速動比率>0
    * 營業活動現金流量(每年)>0
    * 並於2019年再以營收成長率(每年)>10%等做篩選。
    
    <br>
    
- 由於篩選出的公司過多，所以再根據：**近五年來每年殖利率>2%、最新殖利率>近五年平均殖利率、最新殖利率>3%、2019全年EPS>2019所發放之現金股利**，
進一步篩選，希望能夠減少公司數，根據上述條件與 ***20200612*** 的殖利率（只要有新資料就會抓最新的資料），最後篩出的公司有4間，
股票代碼分別為 [1227, 2395, 4137, 5288]，由於執行程式碼的時間不同，抓下來的殖利率是不同天的資料，
所以最後篩出的公司跟上述4間公司可能會有些微的變動。<br>


- 若是篩出的公司大於等於5間公司，則在加上前五年(2015-2019)**利息保障倍數>5、總資產週轉次數>0.8**的公司。<br>


- 從基本面篩出的公司會在用技術面去分析，挑選較適合的推薦股票：<br>

        １．股價(即時) ＞ 均線 MA10
        ２．股價(即時) ＞ 均線 MA20
        ３．均線 MA10 與均線 MA20 呈現黃金交叉
<br>

最後篩出的股票為 **‘4137麗豐-KY’**，再畫出今年技術面分析圖，以利使用者可利用技術面自行進一步判斷。<br>
<br>
<br>
<img src='https://github.com/jiaying777/FINANCIAL_APPLICATION_PROGRAMMING/blob/master/技術分析圖4137.png'>

