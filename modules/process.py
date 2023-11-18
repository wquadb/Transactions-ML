from matplotlib import pyplot as plt
import pandas as pd


def show_correlation(series_1: pd.Series, series_2: pd.Series):
    """
    takes DataFrame and two columns (float/int)
    then shows correlation on x, y surface
    """

    print(f"Correlation between {series_1.name} and {series_2.name}:\n")
    print(series_1.corr(series_2), '\n')

    x = series_1
    y = series_2

    fig, ax = plt.subplots(figsize=(9, 7), dpi=100)

    ax.scatter(x, y, alpha=0.3, s=10)

    plt.xlabel(f"{series_1.name}", fontsize=14, labelpad=15)
    plt.ylabel(f"{series_2.name}", fontsize=14, labelpad=15)
    plt.title("bank_clients", fontsize=14, pad=15)
    plt.show()

    return 0

def show_timeseries(df: pd.DataFrame, item: int = 0):
    """
    takes DataFrame with column 'period' and change dates to the number
    of days from the earliest day creates duplicate for visualising graph
    """
    if item > len(df['item_id'].unique()) - 1:
        raise ValueError("item index out of range")
    else:
        idid = int(df['item_id'].unique()[item])

    if df['date'].dtype != 'datetime64[ns]' and df['date'].dtype != 'int64':
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, format=r'%d.%m.%Y')

    ds = df.loc[df['item_id'] == idid, 'date']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    ds.plot(kind='line', x='date', y="item_price", color='b', ax=axes, rot=0)
    plt.xlabel("date", fontsize=14, labelpad=15)
    plt.ylabel("item_price", fontsize=14, labelpad=15)
    plt.show()

def string_to_TimeAndHour(df: pd.DataFrame, column: str):

    """
    takes DataFrame and column name
    then converts string dates to datetime64[ns] then to int
    and creates column 'hour' with hour from localtime
    """

    days = df[column].str.split(' ').str[0].astype(int)

    times = pd.to_timedelta(df[column].str.split(' ').str[1])

    df[column] = days.apply(pd.Timedelta, unit='d') + times

    df[column] = df[column].dt.total_seconds().astype('int64')

    df['hour'] = times.dt.components['hours'].astype('int64')

    return df
