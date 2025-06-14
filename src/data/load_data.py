import yfinance as yf
import pandas as pd
import os

TICKERS = [
    "AIR.PA",    # Airbus
    "SAF.PA",    # Safran
    "HO.PA",     # Thales
    "MTX.DE",    # MTU Aero Engines
    "LDO.MI",    # Leonardo
    "RR.L",      # Rolls-Royce
    "BA.L",      # BAE Systems
    "BA",        # Boeing
    "RTX",       # RTX Corporation
    "LMT",       # Lockheed Martin
    "NOC",       # Northrop Grumman
    "GD",        # General Dynamics
    "TXT",       # Textron
    "HWM",       # Howmet Aerospace
    "SPR",       # Spirit AeroSystems
    "HEI",       # HEICO Corp
    "TDG",       # TransDigm Group
    "BBD-B.TO",  # Bombardier
    "CAE.TO",    # CAE Inc.
    "ERJ",       # Embraer
    "AVAV",      # AeroVironment
]

def adapt_benchmark(perf_df, start_date : str = "2020-01-01"):
    benchmark_df =  compute_benchmark_returns(start_date) 
    benchmark_df = benchmark_df.set_index(pd.to_datetime(benchmark_df['date'])).sort_index()
    dates = pd.to_datetime(perf_df['date'])
    returns = []
    prev = pd.to_datetime(start_date)
    for curr in dates:
        r = benchmark_df.loc[(benchmark_df.index > prev) & (benchmark_df.index <= curr), 'benchmark_return']
        returns.append((1 + r).prod() - 1 if not r.empty else None)
        prev = curr
    return pd.DataFrame({'date': dates, 'benchmark_return': returns})


def compute_benchmark_returns(start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    price_df = yf.download(TICKERS, start=start, end=end, progress=False)["Close"]
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df = price_df["Close"]
    elif "Close" in price_df.columns:
        price_df = price_df[["Close"]]
        price_df.columns = TICKERS if len(TICKERS) == 1 else [TICKERS[0]]
    returns = price_df.pct_change().dropna()
    benchmark_returns = returns.mean(axis=1).reset_index()
    benchmark_returns.columns = ["date", "benchmark_return"]
    return benchmark_returns

def get_all_returns_fundamental(zscore_df: pd.DataFrame) -> pd.DataFrame:
    all_returns = []

    for ticker in TICKERS:
        #print(f"Fetching prices for {ticker}...")
        try:
            dates = zscore_df[zscore_df["ticker"] == ticker]["date"].sort_values().unique()
            if len(dates) < 2:
                continue

            start_date = (pd.to_datetime(dates[0]) - pd.DateOffset(days=5)).strftime("%Y-%m-%d")
            end_date = (pd.to_datetime(dates[-1]) + pd.DateOffset(days=35)).strftime("%Y-%m-%d")

            data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)

            if isinstance(data.columns, pd.MultiIndex):
                data = data["Close"][ticker]
            else:
                data = data["Close"]

            data = data.dropna().sort_index().to_frame(name="price")
            for i in range(len(dates) - 1):
                t0 = pd.to_datetime(dates[i])
                t1 = pd.to_datetime(dates[i + 1])

                p0 = data[data.index <= t0]["price"].iloc[-1] if not data[data.index <= t0].empty else None
                p1 = data[data.index <= t1]["price"].iloc[-1] if not data[data.index <= t1].empty else None

                if p0 is not None and p1 is not None and p0 != 0:
                    ret = (p1 / p0) - 1
                    all_returns.append({"date": t0, "ticker": ticker, "return": ret})

        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue

    df = pd.DataFrame(all_returns)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_all_fundamentals(start_year=2021, end_year=2025):
    dfs = []
    for ticker in TICKERS:
        #print(f"Fetching fundamentals for {ticker}...")
        try:
            df = fetch_ticker_fundamentals_monthly(ticker, start_year, end_year)
            df["ticker"] = ticker
            dfs.append(df)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            continue

    return dfs

def get_val(s, k):
    try:
        val = s.get(k, None)
        return val if pd.notna(val) else None
    except:
        return None

def fetch_ticker_fundamentals_monthly(ticker: str, start_year=2021, end_year=2025):
    t = yf.Ticker(ticker)

    fin = pd.concat([t.quarterly_financials.T, t.financials.T])
    bs = pd.concat([t.quarterly_balance_sheet.T, t.balance_sheet.T]).sort_index()
    bs = bs[~bs.index.duplicated(keep='last')]  
    cf = pd.concat([t.quarterly_cashflow.T, t.cashflow.T]).sort_index()
    cf = cf[~cf.index.duplicated(keep='last')]

    info = t.info
    market_cap = info.get("marketCap")

    monthly = pd.date_range(f"{start_year}-01", f"{end_year}-12", freq="ME")
    fin, bs, cf = fin.sort_index(), bs.sort_index(), cf.sort_index()

    last = {}
    results = []

    for date in monthly:
        row = {"date": date, "ticker": ticker}

        f = fin.loc[fin.index[fin.index <= date].max()] if not fin.index[fin.index <= date].empty else pd.Series()
        b = bs.loc[bs.index[bs.index <= date].max()] if not bs.index[bs.index <= date].empty else pd.Series()
        c = cf.loc[cf.index[cf.index <= date].max()] if not cf.index[cf.index <= date].empty else pd.Series()

        revenue = get_val(f, "Total Revenue")
        ebitda = get_val(f, "EBITDA")
        net = get_val(f, "Net Income")
        debt = get_val(b, "Total Debt")
        equity = get_val(b, "Ordinary Shares Number")
        assets = get_val(b, "Total Assets")
        cl = get_val(b, "Current Liabilities")
        fcf = get_val(c, "Free Cash Flow")
        cash = get_val(b, "Cash And Cash Equivalents")

        if revenue is not None: last['revenue'] = revenue
        if ebitda is not None: last['ebitda'] = ebitda
        if net is not None: last['net'] = net
        if debt is not None: last['debt'] = debt
        if equity is not None: last['equity'] = equity
        if assets is not None: last['assets'] = assets
        if cl is not None: last['cl'] = cl
        if fcf is not None: last['fcf'] = fcf
        if cash is not None: last['cash'] = cash

        revenue = last.get('revenue')
        ebitda = last.get('ebitda')
        net = last.get('net')
        debt = last.get('debt')
        equity = last.get('equity')
        assets = last.get('assets')
        cl = last.get('cl')
        fcf = last.get('fcf')
        cash = last.get('cash')

        ic = (assets - cl) if assets and cl else None
        nopat = 0.75 * net if net else None
        roe = net / equity if net and equity else None
        gearing = debt / equity if debt and equity else None
        ebitda_margin = ebitda / revenue if ebitda and revenue else None
        net_margin = net / revenue if net and revenue else None
        ev = market_cap + (debt or 0) - (cash or 0) if market_cap else None
        ev_ebitda = ev / ebitda if ev and ebitda else None
        pe = market_cap / net if market_cap and net else None

        data = {
            "FCF_Yield": fcf / market_cap if fcf and market_cap else None,
            "ROIC": nopat / ic if nopat and ic else None,
            "Gearing": gearing,
            "ROE": roe,
            "EV_EBITDA": ev_ebitda,
            "P_E": pe,
            "EBITDA_Margin": ebitda_margin,
            "Net_Margin": net_margin
        }

        for k, v in data.items():
            row[k] = v if v is not None and v != last.get(f"last_{k}") else None
            if v is not None:
                last[f"last_{k}"] = v

        results.append(row)

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["date"])

    annual_revenue = t.financials.T[["Total Revenue"]].dropna().sort_index()
    annual_revenue["Total Revenue"] = pd.to_numeric(annual_revenue["Total Revenue"], errors="coerce")
    annual_revenue["Revenue_Growth_YOY"] = annual_revenue["Total Revenue"].pct_change()
    annual_revenue = annual_revenue.reset_index().rename(columns={"index": "date"})
    annual_revenue["date"] = pd.to_datetime(annual_revenue["date"])

    df = df.merge(annual_revenue[["date", "Revenue_Growth_YOY"]], on="date", how="left")

    return df


def fetch_all_fundamentals(input_csv: str, output_dir: str, start_year=2020, end_year=2025):
    os.makedirs(output_dir, exist_ok=True)

    tickers_df = pd.read_csv(input_csv)
    for _, row in tickers_df.iterrows():
        ticker = row["ticker"]
        print(f"Fetching fundamentals for {ticker}...")

        try:
            df = fetch_ticker_fundamentals_monthly(ticker=ticker, start_year=start_year, end_year=end_year)
            df.to_csv(os.path.join(output_dir, f"{ticker}_fundamentals.csv"), index=False)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")


