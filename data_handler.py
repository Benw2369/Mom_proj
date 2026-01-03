import datetime as dt
from datetime import datetime
import pandas as pd
import yfinance as yf
import sqlalchemy
import time
import requests
from io import StringIO
import json
from sqlalchemy import inspect
from pandas.tseries.offsets import BDay


with open('index_info.json', 'r') as f:
    dict_json = json.load(f)



today = datetime.now().date()
start = datetime(2010,1,1)
date_5bd_ago = today - BDay(5)
date_20bd_ago = today - BDay(20)


def render_table(url, table_index=0, col_index=0):
    headers = {'User-Agent': 'Mozilla/5.0'}
    for i in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            return pd.read_html(StringIO(r.text))[table_index].iloc[:, col_index]
        except Exception:
            if i < 2:
                time.sleep(2 ** i)
            else:
                raise


def fetch_index_constituents(ticker="DJI"):
    with open('index_info.json', 'r') as f:
        index_static = json.load(f)
    return render_table(
        index_static[ticker]['url'],
        index_static[ticker]['table_index'],
        index_static[ticker]['col_index']
    ).tolist()


def get_price_data(tickers, start, end):
    data, errors = {}, {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                auto_adjust=True,
                multi_level_index=False,
                start=start,
                end=end,
                progress=False
            ).reset_index()

            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                data[ticker] = df
        except Exception as e:
            errors[ticker] = str(e)

    return data, errors


def create_engine(name):
    return sqlalchemy.create_engine(f"sqlite:///{name}.db")

engines = {k: create_engine(f"{k}_data") for k in dict_json}
dict_tickers = {k: fetch_index_constituents(k) for k in dict_json}

def get_sql_db(engines):
    sql_dbs = {}
    for key, engine in engines.items():
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        sql_dbs[key] = {
            t: pd.read_sql(f'SELECT * FROM "{t}"', engine, parse_dates=['Date'])
            for t in tables
        }
    return sql_dbs

def write_sql_append(sql_dbs, engines):
    for index_key, db in sql_dbs.items():
        engine = engines[index_key]

        for ticker, df in db.items():
            inspector = inspect(engine)

            if ticker in inspector.get_table_names():
                existing = pd.read_sql(
                    f'SELECT Date FROM "{ticker}"',
                    engine,
                    parse_dates=['Date']
                )
                df = df[~df['Date'].isin(existing['Date'])]

            if not df.empty:
                df.to_sql(ticker, engine, if_exists='append', index=False)


# Business-day range utility
def dates_to_bday_ranges(dates):
    if not dates:
        return []
    dates = sorted(pd.to_datetime(dates))
    ranges = []
    start = prev = dates[0]

    for d in dates[1:]:
        if d == prev + BDay(1):
            prev = d
        else:
            ranges.append((start, prev + BDay(1)))
            start = prev = d

    ranges.append((start, prev + BDay(1)))
    return ranges




def update_sql_db():
    for index_key, tickers in dict_tickers.items():

        if index_key not in sql_dbs:
            sql_dbs[index_key] = {}

        for ticker in tickers:

            # ---------- CASE 1: ticker not in DB ----------
            if ticker not in sql_dbs[index_key]:
                data, _ = get_price_data(
                    [ticker],
                    start=start,
                    end=date_5bd_ago + BDay(1)
                )
                if ticker in data:
                    sql_dbs[index_key][ticker] = data[ticker]
                continue

            df_existing = sql_dbs[index_key][ticker]
            last_date = pd.to_datetime(df_existing['Date']).max()

            # ---------- CASE 2: already up to date ----------
            if last_date >= date_5bd_ago:
                continue

            # ---------- CASE 3: incremental forward fill ----------
            query_start = last_date + BDay(1)
            query_end = date_5bd_ago + BDay(1)

            data, _ = get_price_data([ticker], query_start, query_end)

            if ticker in data:
                new_df = data[ticker]
                new_df = new_df[~new_df['Date'].isin(df_existing['Date'])]
                sql_dbs[index_key][ticker] = pd.concat(
                    [df_existing, new_df],
                    ignore_index=True
                )
    write_sql_append(sql_dbs, engines)



sql_dbs = get_sql_db(engines)

