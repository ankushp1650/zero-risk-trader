import pandas as pd


parameters = ["TIME_SERIES_DAILY"] #, "NEWS_SENTIMENT"]
stock_symbol = ["TCS.BSE", "RELIANCE.BSE", "HDFCBANK.BSE", "BHARTIARTL.BSE", "ICICIBANK.BSE"]


def stock_data(df):
    # ********** table 1 *************
    df.rename(columns={'index': 'Date'}, inplace=True)
    df1 = df[['Date', 'Meta Data']][:5]
    df1["Date"] = df1["Date"].str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
    df1.columns = [col.capitalize() for col in df1.columns]
    table1 = df1

    # ********** table 2 *************
    df2 = df[['Date', 'Time Series (Daily)']][5:]
    df2['Time Series (Daily)'] = df2['Time Series (Daily)'].apply(lambda x: dict(eval(x)))
    df3 = df2['Time Series (Daily)'].apply(pd.Series)
    result = pd.concat([df2, df3], axis=1).drop('Time Series (Daily)', axis=1)
    result.columns = result.columns.str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
    result.columns = [col.capitalize() for col in result.columns]
    table2 = result.reset_index(drop=True)
    table1.index = table1.index + 1
    table2.index = table2.index + 1

    return table1, table2


