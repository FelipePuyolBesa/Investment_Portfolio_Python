# Investment_Portfolio_Python

This exercise is done under the guidance of the video [How to Invest with Data Science](https://www.youtube.com/watch?v=4jaBKXDqg9U&list=PL0UhIYS-b6hhMFshRCFapwuPq9Dhvn4V8&index=2&t=11383s&ab_channel=DerekBanas) made by Derek Banas, the code is from his authority, this is a purely academic exercise and is not intended to be a financial guide.

## Objective
Generate, through Python, code that allows the extraction of stock investment data, the selection of a set of assets and the creation of an investment portfolio.

![Inversion01](https://user-images.githubusercontent.com/124479181/219982407-147084ce-ec4d-4962-8dde-5ad30cbea8e9.png)

## Data
The data we are going to use is the stock asset data found on Yahoo Finance.
In this case, the Wilshire 5000 asset group will be used.

![Inversion02](https://user-images.githubusercontent.com/124479181/219982413-dbf03bd2-ae34-4707-8245-6d08bd9ff15c.png)

### Wilshire 5000

It is a market capitalization-weighted index of the market value of all US stocks actively traded in the United States. The index purports to measure the performance of the majority of publicly traded companies based in the United States, with readily available price data. (we exclude Bulletin Board/penny stocks and extremely small company stocks).

# Portfolio Generation

A summary of the main functionalities of the code is presented below, to access the files use this [link](https://github.com/FelipePuyolBesa/Investment_Portfolio_Python) 

## 1. Obtaining the information of the actions. Archive: [Portfolio_Investment_Download.py](https://github.com/FelipePuyolBesa/Investment_Portfolio_Python/blob/main/Portfolio_Investment_Download.ipynb)

Download or view stock information from Yahoo Finance ##

    # Variable definition
    PATH = 'C:\Users\Felipe\Desktop\PortafoliodeInversionPython\CSV\Wilshire/'

    # Default start and end dates
    S_DATE = '2017-02-01'
    E_DATE = '2022-06-19'
    S_DATE_DT = pd.to_datetime(S_DATE)
    E_DATE_DT = pd.to_datetime(E_DATE)


Generation of a function to obtain the column data from the CSV

    def get_column_from_csv (file, col_name):
        try:
            df = pd.read_csv(file)
        except FileNotFoundError:
            print('File does not exist')
        else:
            return df[col_name]
    
Obtain the information of the indicative of the actions of a CSV file previously created with the tickers

    tickers = get_column_from_csv(r'C:\Users\Felipe\Desktop\PortafoliodeInversionPython\CSV/Wilshire-5000-stocks.csv', 'Ticker')

Save action data in a CSV

    # Creation of the function that generates a dataframe with the ticker and the start date

    def save_to_csv_from_yahoo(folder, ticker):
        stock = yf.Ticker(ticker)

        try:
            print('Get data for: ', ticker)
            # Obtaining the historical data of the closing price
            df = stock.history(period='5y')

            # Two second wait
            time.sleep(2)

            # Replace point to save CSV file
            # Save the data in a CSV
            # File saving
            the_file = folder + ticker.replace('.','_')+'.csv'
            print(the_file, 'Saved')
            df.to_csv(the_file)
        except Exception as ex:
            print('Could not get data for: ', ticker)

Download all the information of the actions

    for x in range (0, 3481):
        save_to_csv_from_yahoo(PATH, tickers[x])
        print('Finished')


### Result:

Generation of CSV files with the consulted data of the established ticks, they are not downloaded for all 3481 actions, since the ticks have changed due to mergers or eliminations of companies.

The CSVs contain data for each stock for the historical reading date (daily), open price, high price, low price, close price, dividends generated, stock, and splite.

![Inversion04](https://user-images.githubusercontent.com/124479181/219982499-815bf8d3-3d5d-4907-b560-6ed6cff6c4ef.png)

## 2. Generation of calculations required for the creation of graphs. Archive: [Portfolio_Investment_Calculations.py](https://github.com/FelipePuyolBesa/Investment_Portfolio_Python/blob/main/Portfolio_Investment_Calculations.ipynb)


Obtain a list of the stocks that have been downloaded

    files = [x for x in listdir(PATH) if isfile(join(PATH, x))]
    tickers = [os.path.splitext(x)[0] for x in files]

    tickers.sort()
    len(tickers)


Add a column with the daily returns: closing price of a day divided by the previous minus one.

    def add_daily_return_to_df (df):
        df['daily_return'] = df['Close']/df['Close'].shift(1) - 1
        return df

Add a column with cumulative returns: cumulative daily return for each day

    def add_cum_return_to_df (df):
        df['cum_return'] = (1 + df['daily_return']).cumprod() # cumprod()function is used when we want to compute the cumulative product of array elements over a given axis
        return df


Add Bollinger bands

Bollinger Bands plot 2 lines using a moving average and the standard deviation defines how far apart the lines are. They also are used to define if prices are to high or low. When bands tighten it is believed a sharp price move in some direction. 
Prices tend to bounce off of the bands which provides potential market actions.

A strong trend should be noted if the price moves outside the band. If prices go over the resistance line it is in overbought territory and if it breaks through support it is a sign of an oversold position.

Definition of the function for the Bollinger bands

    def add_bollinger_bands(df):
        # Central band generation (moving average) of the closing price with a 20-day window.
        df['middle_band'] = df['Close'].rolling(window = 20).mean()
        # Generation of the upper band with twice the standard deviation.
        df['upper_band'] = df['middle_band'] + 2*df['Close'].rolling(window = 20).std()
        # Generation of the lower band with two times less the standard deviation.
        df['lower_band'] = df['middle_band'] - 2*df['Close'].rolling(window = 20).std()
        return df


Add the Ichimoku data to the df

The Ichimoku (One Look) is considered an all in one indicator. It provides information on momentum, support and resistance. It is made up of 5 lines. If you are a short term trader you create 1 minute or 6 hour. Long term traders focus on day or weekly data.

Conversion Line (Tenkan-sen) : Represents support, resistance and reversals. Used to measure short term trends.
Baseline (Kijun-sen) : Represents support, resistance and confirms trend changes. Allows you to evaluate the strength of medium term trends. Called the baseline because it lags the price.
Leading Span A (Senkou A) : Used to identify future areas of support and resistance
Leading Span B (Senkou B) : Other line used to identify suture support and resistance
Lagging Span (Chikou) : Shows possible support and resistance. It is used to confirm signals obtained from other lines.
Cloud (Kumo) : Space between Span A and B. Represents the divergence in price evolution.
Formulas

Lagging Span = Price shifted back 26 periods
Base Line = (Highest Value in period + Lowest value in period)/2 (26 Sessions)
Conversion Line = (Highest Value in period + Lowest value in period)/2 (9 Sessions)
Leading Span A = (Conversion Value + Base Value)/2
Leading Span B = (Period high + Period low)/2 (52 Sessions)

    def add_Ichimoku(df):
        # Conversion Line = (Highest Value in period + Lowest value in period)/2 (9 Sessions)
        hi_val = df['High'].rolling(window = 9).max()
        low_val = df['Low'].rolling(window = 9).min() 
        df['Conversion'] = (hi_val + low_val)/2

        # Base Line = (Highest Value in period + Lowest value in period)/2 (26 Sessions)
        hi_val2 = df['High'].rolling(window = 26).max()
        low_val2 = df['Low'].rolling(window = 26).min() 
        df['Baseline'] = (hi_val2 + low_val2)/2

        # Leading Span A = (Conversion Value + Base Value)/2
        df['SpanA'] = ((df['Conversion'] + df['Baseline'])/2)

        # Leading Span B = (Period high + Period low)/2 (52 Sessions)
        hi_val3 = df['High'].rolling(window = 52).max()
        low_val3 = df['Low'].rolling(window = 52).min() 
        df['SpanB'] = ((hi_val3 + low_val3)/2)

        # Lagging Span = Price shifted back 26 periods
        df['Lagging'] = df['Close'].shift(-26)

        return df

Calculation generation test for a CSV file

    try:
        print('Working in:', 'A')
        new_df = get_stock_df_from_csv('A')
        new_df = add_daily_return_to_df(new_df)
        new_df = add_cum_return_to_df(new_df)
        new_df = add_bollinger_bands(new_df)
        new_df = add_Ichimoku(new_df)
        new_df.to_csv(PATH + 'A' + '.csv')
    except Exception as ex:
        print(ex)
    

Perform and add calculations to all ticker or action files

    for x in tickers :
        try:
            print('Working in:', x)
            new_df = get_stock_df_from_csv(x)
            new_df = add_daily_return_to_df(new_df)
            new_df = add_cum_return_to_df(new_df)
            new_df = add_bollinger_bands(new_df)
            new_df = add_Ichimoku(new_df)
            new_df.to_csv(PATH + x + '.csv')
        except Exception as ex:
                print(ex)
### Result:

It is obtained for each stock in each csv file, the values ​​of the daily return, accumulated return from the initial date, the moving average, upper and lower band, baseline, Span A, Span B y lagging.

![Inversion05](https://user-images.githubusercontent.com/124479181/219982502-48d6bab7-4023-4f0b-9a83-a5b16c6663bf.png)

## 3. Generation of functions for the creation of Bollinger Bands and Ichimoku graphs. [Portfolio_Investment_Graphics.py](https://github.com/FelipePuyolBesa/Investment_Portfolio_Python/blob/main/Portfolio_Investment_Graphics.ipynb)

Chart Bollinger bands

    def plot_with_boll_bands(df, ticker):

        fig = go.Figure()

    # Chart the candles with Plotly
        candle = go.Candlestick(x=df.index, open=df['Open'],
        high=df['High'], low=df['Low'],
        close=df['Close'], name="Candlestick")

    # Graph the three calculated lines
        upper_line = go.Scatter(x=df.index, y=df['upper_band'], 
        line=dict(color='rgba(250, 0, 0, 0.75)', 
        width=1), name="Upper Band")

        mid_line = go.Scatter(x=df.index, y=df['middle_band'], 
        line=dict(color='rgba(0, 0, 250, 0.75)', 
        width=0.7), name="Middle Band")

        lower_line = go.Scatter(x=df.index, y=df['lower_band'], 
        line=dict(color='rgba(0, 250, 0, 0.75)', 
        width=1), name="Lower Band")

    # Add all four charts into one
        fig.add_trace(candle)
        fig.add_trace(upper_line)
        fig.add_trace(mid_line)
        fig.add_trace(lower_line)

    # Give titles to the graph and the axes, as well as add a slider
        fig.update_xaxes(title="Date", rangeslider_visible=True)
        fig.update_yaxes(title="Price")

        fig.update_layout(title=ticker + " Bollinger Bands",
        height=1000, width=1800, showlegend=True)
        plot(fig)

Ishimoku graph

    # Cloud color assignment function

    def get_fill_color(label):
        if label >= 1:
            return 'rgba(0,250,0,0.4)'
        else:
            return 'rgba(250,0,0,0.4)'

    # Function for the Ishimoku graph

    def get_Ichimoku(df):

        candle = go.Candlestick(x=df.index, open=df['Open'],
        high=df['High'], low=df["Low"], close=df['Close'], name="Candlestick")

        df1 = df.copy()
        fig = go.Figure()
        df['label'] = np.where(df['SpanA'] > df['SpanB'], 1, 0)
        df['group'] = df['label'].ne(df['label'].shift()).cumsum()

        df = df.groupby('group')

        dfs = []
        for name, data in df:
            dfs.append(data)

        for df in dfs:
            fig.add_traces(go.Scatter(x=df.index, y=df.SpanA,
            line=dict(color='rgba(0,0,0,0)')))

            fig.add_traces(go.Scatter(x=df.index, y=df.SpanB,
            line=dict(color='rgba(0,0,0,0)'),
            fill='tonexty',
            fillcolor=get_fill_color(df['label'].iloc[0])))

        baseline = go.Scatter(x=df1.index, y=df1['Baseline'], 
        line=dict(color='pink', width=2), name="Baseline")

        conversion = go.Scatter(x=df1.index, y=df1['Conversion'], 
        line=dict(color='black', width=1), name="Conversion")

        lagging = go.Scatter(x=df1.index, y=df1['Lagging'], 
        line=dict(color='purple', width=2), name="Lagging")

        span_a = go.Scatter(x=df1.index, y=df1['SpanA'], 
        line=dict(color='green', width=2, dash='dot'), name="Span A")

        span_b = go.Scatter(x=df1.index, y=df1['SpanB'], 
        line=dict(color='red', width=1, dash='dot'), name="Span B")

        fig.add_trace(candle)
        fig.add_trace(baseline)
        fig.add_trace(conversion)
        fig.add_trace(lagging)
        fig.add_trace(span_a)
        fig.add_trace(span_b)

        fig.update_layout(height=1000, width=1800, showlegend=True)

        plot(fig)
        
 ### Result:
 
Bollinger Bands chart for ticker AA.
 
![Inversion06](https://user-images.githubusercontent.com/124479181/219982540-5269fe08-e839-4a46-b2f7-d5d784a21b01.png)


Ishimoku chart for the AA ticker.

![Inversion07](https://user-images.githubusercontent.com/124479181/219982544-c8058647-3b77-488f-bcf1-37145c955d02.png)

 

## 4. Obtain the information of the sectors for the actions that make up the Wilshire 5000. [Portfolio_Investment_Sectors.py](https://github.com/FelipePuyolBesa/Investment_Portfolio_Python/blob/main/Portfolio_Investment_Sectors.ipynb)


This information is in the file 'big_stock_sectors.csv'

    sec_df = pd.read_csv(r'C:\Users\Felipe\Desktop\PortafoliodeInversionPython\CSV/big_stock_sectors.csv')
    
![Inversion08](https://user-images.githubusercontent.com/124479181/219982571-b9cab871-61be-49bd-8c4d-8e63c67e2957.png)


Separation in different DF of the actions by sector

    # Consultation of the sectors
    print(sec_df['Sector'].unique())

    """
    ['Healthcare' 'Materials' 'SPAC' 'Discretionary' 'Real Estate'
     'Industrial' 'Financials' 'Information Technology' 'Industrials'
     'Staples' 'Services' 'Utilities' 'Communication' 'Energy' nan]
    """

    indus_df = sec_df.loc[sec_df['Sector'] == 'Industrial']
    health_df = sec_df.loc[sec_df['Sector'] == 'Healthcare']
    it_df = sec_df.loc[sec_df['Sector'] == 'Information Technology']
    comm_df = sec_df.loc[sec_df['Sector'] == 'Communication']
    staple_df = sec_df.loc[sec_df['Sector'] == 'Staples']
    discretion_df = sec_df.loc[sec_df['Sector'] == 'Discretionary']
    materials_df = sec_df.loc[sec_df['Sector'] == 'Materials']
    spac_df = sec_df.loc[sec_df['Sector'] == 'SPAC']
    real_estate_df = sec_df.loc[sec_df['Sector'] == 'Real Estate']
    financials_df = sec_df.loc[sec_df['Sector'] == 'Financials']
    industrials_df = sec_df.loc[sec_df['Sector'] == 'Industrials']
    services_df = sec_df.loc[sec_df['Sector'] == 'Services']
    utilities_df = sec_df.loc[sec_df['Sector'] == 'Utilities']
    energy_df = sec_df.loc[sec_df['Sector'] == 'Energy']


Creation of a function to calculate the accumulated return for each of the actions

    def get_cum_ret_for_stocks(stock_df):
        tickers = []
        cum_rets = []

        for index, row in stock_df.iterrows():
            df = get_stock_df_from_csv(row['Ticker'])
            if df is None:
                pass
            else:
                tickers.append(row['Ticker'])
                cum = df['cum_return'].iloc[-1]
                cum_rets.append(cum)
        return pd.DataFrame({'Ticker':tickers, 'CUM_RET':cum_rets})

Application of the function to find the accumulated number of shares

    Healthcare = get_cum_ret_for_stocks(health_df)
    Materials = get_cum_ret_for_stocks(materials_df)
    SPAC = get_cum_ret_for_stocks(spac_df)
    Discretionary = get_cum_ret_for_stocks(discretion_df)
    Real_Estate = get_cum_ret_for_stocks(real_estate_df)
    Industrial = get_cum_ret_for_stocks(indus_df)
    Financials = get_cum_ret_for_stocks(financials_df)
    IT = get_cum_ret_for_stocks(it_df)
    Industrials = get_cum_ret_for_stocks(industrials_df)
    Staples = get_cum_ret_for_stocks(staple_df)
    Services = get_cum_ret_for_stocks(services_df)
    Utilities = get_cum_ret_for_stocks(utilities_df)
    Communication = get_cum_ret_for_stocks(comm_df)
    Energy = get_cum_ret_for_stocks(energy_df)

Review by sector of the stocks with the highest cumulative return

    print('Top 10 Industrial')
    print(Industrial.sort_values(by=['CUM_RET'], ascending=False).head(10))
    
![Inversion09](https://user-images.githubusercontent.com/124479181/219982573-ba235e7c-df34-4aff-82a8-a0511cc9d673.png)

Stocks selected by highest cumulative return: PLUG, AMRC, GNRC

Graph some of the actions to decide which one could invest.

    df_ind = get_stock_df_from_csv('AMRC')
    get_Ichimoku(df_ind)
    
![Inversion10](https://user-images.githubusercontent.com/124479181/219982615-f97577c7-96e9-449b-ad90-532f51b9db1a.png)


        print('Top 10 Materials')
        print(Materials.sort_values(by=['CUM_RET'], ascending=False).head(10))
        
![Inversion11](https://user-images.githubusercontent.com/124479181/219982620-7c05cc0d-c2ca-4438-bb8b-079a5411bb9c.png) 

Stocks selected for highest cumulative return: HCC, RFP, CF

Graph some of the actions to decide which one could invest.

        df_mat = get_stock_df_from_csv('HCC')
        get_Ichimoku(df_mat)
        
![Inversion12](https://user-images.githubusercontent.com/124479181/219982644-3bdfee85-a35c-4d36-9ee5-29b41ff552ad.png)


        print('Top 10 Discretionary')
        print(Discretionary.sort_values(by=['CUM_RET'], ascending=False).head(10))

![Inversion13](https://user-images.githubusercontent.com/124479181/219982649-9737ba69-ded3-4888-9dcb-36f48aafe6ea.png)

Stocks selected by highest cumulative return: CELH, BOOT, VERU

Graph some of the actions to decide which one could invest.

        df_Discretionary = get_stock_df_from_csv('CELH')
        get_Ichimoku(df_Discretionary)
        
![Inversion14](https://user-images.githubusercontent.com/124479181/219982676-e765d0e7-21f0-4354-abf9-248aa1aa54e1.png)

In this way, the accumulated returns of the actions by sector are obtained and the Ishimoku graphs are generated, in order to carry out a manual selection of the actions that will make up the portfolio.

## 5. Obtaining the portfolio to invest according to the Markowitz optimization with actions with low correlation. [Obtaining_Investment_Portfolio.py](https://github.com/FelipePuyolBesa/Investment_Portfolio_Python/blob/main/Obtaining_Investment_Portfolio.ipynb)


Get the data from the created CSVs

        def get_stock_df_from_csv(ticker):
            try:
                df = pd.read_csv(PATH + ticker + '.csv', index_col=0)
            except FileNotFoundError:
                print('El archivo no existe')
            else:
                return df
    

Join multiple actions by column name into a single df

        def merge_df_by_column_name(col_name, sdate, edate, *tickers):
            # Will hold data for all dataframes with the same column name
            mult_df = pd.DataFrame()

            for x in tickers:
                df = get_stock_df_from_csv(x)
                mask = (df.index >= sdate) & (df.index <= edate)
                mult_df[x] = df.loc[mask][col_name]

            return mult_df

Markowitz Portfolio Optimization

Harry Markowitz proved that you could make what is called an efficient portfolio. That is a portfolio that optimizes return while also minimizing risk. We don't benefit from analyzing individual securities at the same rate as if we instead considered a portfolio of stocks.

We do this by creating portfolios with stocks that are not correlated. We want to calculate expected returns by analyzing the returns of each stock multiplied by its weight.

w1r1 + w2r2 = rp

The standard deviation of the portfolio is found this way. Sum multiple calculations starting by finding the product of the first securities weight squared times its standard deviation squared. The middle is 2 times the correlation coefficient between the stocks. 
And, finally add those to the weight squared times the standard deviation squared for the second security.

(w1d1 + w2d2)^2 = w1^2*d1^2 + 2w1d1w2d2 + w2^2 * d2^2


Draw the most efficient frontier
Selection of a portfolio with shares previously studied, with the data of the accumulated performance and Ishimoku
For this case, I selected some of the specific sectors.

        port_list = ['PLUG', 'AMRC', 'GNRC',
        'HCC', 'RFP', 'CF',
        'IIPR', 'BRT', 'BRG',
        'CDNA', 'ZYXI', 'ARWR',
        'ATLC', 'KNSL', 'LPLA',
        'ENPH', 'APPS', 'SEDG',
        'RCMT', 'FCN', 'MHH',
        'NEE', 'MSEX', 'EXC',
        'TTGT', 'ROKU', 'IRDM',
        'OAS', 'VTNR', 'EGY']

        num_stocks = len(port_list)
        print(num_stocks)

Generate a df with the closing prices of all selected stocks

        mult_df = merge_df_by_column_name('Close', S_DATE, E_DATE, *port_list)

Generate a chart for stock prices

        fig = px.line(mult_df, x = mult_df.index, y = mult_df.columns)
        fig.update_layout(height=1000, width=1800, showlegend=True)
        fig.update_xaxes(title="Date", rangeslider_visible=True)
        fig.update_yaxes(title="Price")
        plot(fig)

![Inversion15](https://user-images.githubusercontent.com/124479181/219982678-bf3cb187-b917-4bc8-8907-10c7b350f67c.png)


Generate a price transformation and chart

        mult_df_t = np.log10(mult_df)

        fig = px.line(mult_df_t, x = mult_df_t.index, y = mult_df_t.columns)
        fig.update_layout(height=1000, width=1800, showlegend=True)
        fig.update_xaxes(title="Date", rangeslider_visible=True)
        fig.update_yaxes(title="Log10 Price")
        plot(fig)
        
![Inversion16](https://user-images.githubusercontent.com/124479181/219982703-cd15667c-ea0e-4935-958f-6bd23e380fc3.png)


Average returns for one year (252 business days)

        returns = np.log(mult_df / mult_df.shift(1))
        mean_ret = returns.mean()*252
        print(mean_ret)

Calculation of the correlation of actions

        returns.corr()
        
![Inversion17](https://user-images.githubusercontent.com/124479181/219982709-8b8aa395-f5be-4b4d-bda0-c23af32b1159.png)


Stock Correlation Chart
We want a portfolio with low correlation between stocks

        import seaborn as sns
        correlation_matrix = returns.corr(method='spearman')
        fig = sns.heatmap(matriz_correlacion, annot=False)
        # fig.update_layout(height=1000, width=1800, showlegend=True)
        plt.show()
        
![Inversion18](https://user-images.githubusercontent.com/124479181/219982743-fa18befd-43e7-4a19-9a25-32639062ec0e.png)
  

Generation of random weights whose sum is one

        weights = np.random.random(num_stocks)
        weights /= np.sum(weights)  # weights = weights / np.sum(weights)
        print('Weights: ', weights)
        print('Total weight: ', np.sum(weights))

Calculation of the average annual return with the random weights

        print(np.sum(weights * returns.mean()) * 252)


Volatility Calculation

        # Portfolio risk with current pesos

        print(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))

Running a simulation of 10000 portfolios using a function

        p_ret = [] # Returns list
        p_vol = [] # Volatility list
        p_SR = [] # Sharpe Ratio list
        p_wt = [] # Weights per portfolio list


        for x in range(10000):
            # Generate randoms weights
            p_weights = np.random.random(num_stocks)
            p_weights /= np.sum(p_weights)
    
Calculation of the return according to the weights
  
    ret_1 = np.sum(p_weights * returns.mean()) * 252
    p_ret.append(ret_1)
    
Volatility Calculation

    vol_1 = np.sqrt(np.dot(p_weights.T, np.dot(returns.cov() * 252, p_weights)))
    p_vol.append(vol_1)
    
Calculation of the Sharpe ratio

    SR_1 = (ret_1 - risk_free_rate) / vol_1
    p_SR.append(SR_1)
    
Store the weights for each portfolio

    p_wt.append(p_weights)
    
Convert to numpy arrays

        p_ret = np.array(p_ret)
        p_vol = np.array(p_vol)
        p_SR = np.array(p_SR)
        p_wt = np.array(p_wt)

        p_ret, p_vol, p_SR, p_wt


Graph of the simulated portfolios or most efficient frontier

        ports = pd.DataFrame({'Returns': p_ret, 'Volatility': p_vol, })
        ports.plot(x='Volatility', y = 'Returns', kind = 'scatter', figsize = (19,9))
        
![Inversion19](https://user-images.githubusercontent.com/124479181/219982752-7827a472-726f-436f-b438-800ddd2e654b.png)


Sharpe ratio

People want to maximize returns while avoiding as much risk as possible. 
William Sharpe created the Sharpe Ratio to find the portfolio that provides the best return for the lowest amount of risk.

As return increases so does the Sharpe Ratio, but as Standard Deviation increase the Sharpe Ratio decreases.

Returns the index for the highest Sharpe Ratio

        SR_idx = np.argmax(p_SR)

Find the ideal weights for the portfolio in that index

        i = 0
        while i < num_stocks:
            print("Stock : %s : %2.2f" % (port_list[i], (p_wt[SR_idx][i] * 100)))
            i += 1
    
Find the volatility of that portfolio

        print("\nVolatility :", p_vol[SR_idx] * 100)
      
Find the return on that portfolio

        print("Return :", p_ret[SR_idx] * 100)
        
### Result: 

List of the actions that make up the portfolio, with the percentage distribution according to the best result obtained after the execution of 10,000 simulations.

![Inversion20](https://user-images.githubusercontent.com/124479181/219982781-310fc13f-cf95-44d3-896f-3fe70e2a2201.png)

Obtaining the volatility and the expected return during a year of investment.

![Inversion21](https://user-images.githubusercontent.com/124479181/219982786-f192e5a8-9a6b-4d4e-9be6-fd03a86d2a48.png)


You can also take percentages less than one and bring them closer to one, then calculate the portfolio.
In situations in which the percentages are less than one, what can be done is to bring them closer to one or to an action, or directly discard them.
