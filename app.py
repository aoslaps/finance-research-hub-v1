"""Finance Research Hub v1: single-ticker dashboard in Streamlit."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


CACHE_TTL_SECONDS = 300
TIME_RANGE_MAP: dict[str, str] = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "5Y": "5y",
}


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_ticker_info(symbol: str) -> dict[str, Any]:
    """Fetch the ticker info dictionary from yfinance."""
    try:
        info = yf.Ticker(symbol).info
    except Exception as exc:  # pragma: no cover - runtime network path
        raise RuntimeError(
            f"Failed to fetch company info for '{symbol}'. Check the symbol and your network."
        ) from exc

    return info if isinstance(info, dict) else {}


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_price_history(symbol: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV history for a ticker and period from yfinance."""
    try:
        history = yf.Ticker(symbol).history(period=period)
    except Exception as exc:  # pragma: no cover - runtime network path
        raise RuntimeError(
            f"Failed to fetch price history for '{symbol}'. Check the symbol and your network."
        ) from exc

    return history


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_ticker_news(symbol: str) -> list[dict[str, Any]]:
    """Fetch the raw news list from yfinance."""
    try:
        news = yf.Ticker(symbol).news
    except Exception as exc:  # pragma: no cover - runtime network path
        raise RuntimeError(
            f"Failed to fetch news for '{symbol}'. Check the symbol and your network."
        ) from exc

    return news if isinstance(news, list) else []


def rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) using Wilder-style smoothing."""
    # TEACHING: RSI compares average recent gains vs. losses to estimate momentum.
    # Values range from 0 to 100 and are commonly interpreted as:
    # > 70 overbought, < 30 oversold, otherwise neutral.
    # Source convention: J. Welles Wilder's original RSI smoothing (alpha = 1 / period).
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close_prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Compute MACD line and signal line with exponential moving averages."""
    # TEACHING: MACD tracks trend momentum by subtracting a slow EMA from a fast EMA.
    # The signal line is an EMA of MACD itself; crossings are often read as shifts in trend.
    # Source convention: textbook MACD parameters (12, 26, 9).
    ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def format_currency(value: Any) -> str:
    """Format a numeric value as USD text."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


def format_number(value: Any, decimals: int = 2) -> str:
    """Format a generic number with separators."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def format_compact_number(value: Any) -> str:
    """Format a large number into K/M/B/T units."""
    if value is None or pd.isna(value):
        return "N/A"

    number = float(value)
    absolute = abs(number)
    if absolute >= 1_000_000_000_000:
        return f"{number / 1_000_000_000_000:.2f}T"
    if absolute >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{number / 1_000:.2f}K"
    return f"{number:.2f}"


def format_percent(value: Any) -> str:
    """Format a numeric value as a percentage string."""
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def format_news_datetime(raw_publish_time: Any) -> str:
    """Convert provider publish time to America/New_York display text."""
    if raw_publish_time is None:
        return "N/A"

    timestamp = pd.to_datetime(raw_publish_time, unit="s", utc=True, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.to_datetime(raw_publish_time, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "N/A"

    timestamp_ny = timestamp.tz_convert("America/New_York")
    return timestamp_ny.strftime("%Y-%m-%d %H:%M %Z")


def classify_rsi(rsi_value: Any) -> str:
    """Return RSI interpretation label."""
    if rsi_value is None or pd.isna(rsi_value):
        return "N/A"
    if float(rsi_value) > 70:
        return "Overbought"
    if float(rsi_value) < 30:
        return "Oversold"
    return "Neutral"


def classify_macd(macd_value: Any, signal_value: Any) -> str:
    """Return MACD interpretation label."""
    if macd_value is None or signal_value is None:
        return "N/A"
    if pd.isna(macd_value) or pd.isna(signal_value):
        return "N/A"

    macd_float = float(macd_value)
    signal_float = float(signal_value)

    # Neutral rule from spec: within 1% of each other.
    baseline = max(abs(macd_float), abs(signal_float), 1e-9)
    if abs(macd_float - signal_float) <= baseline * 0.01:
        return "Neutral"
    if macd_float > signal_float:
        return "Bullish"
    return "Bearish"


def compute_distance_from_ma_percent(price: Any, moving_average: Any) -> float | None:
    """Compute distance from moving average as a percent of price."""
    if price is None or moving_average is None:
        return None
    if pd.isna(price) or pd.isna(moving_average):
        return None

    price_float = float(price)
    if price_float == 0:
        return None

    # Formula is relative to current price, matching the requested display convention.
    return ((price_float - float(moving_average)) / price_float) * 100


def render_header(
    symbol: str,
    info: dict[str, Any],
    history_for_fallback: pd.DataFrame,
) -> None:
    """Render top-of-page header with symbol, company, price, and day move."""
    company_name = info.get("longName") or info.get("shortName") or "N/A"

    current_price = info.get("regularMarketPrice") or info.get("currentPrice")
    if (current_price is None or pd.isna(current_price)) and not history_for_fallback.empty:
        current_price = history_for_fallback["Close"].iloc[-1]

    previous_close = info.get("regularMarketPreviousClose")
    if (previous_close is None or pd.isna(previous_close)) and len(history_for_fallback) >= 2:
        previous_close = history_for_fallback["Close"].iloc[-2]

    day_change: float | None = None
    day_change_pct: float | None = None
    if (
        current_price is not None
        and previous_close is not None
        and not pd.isna(current_price)
        and not pd.isna(previous_close)
        and float(previous_close) != 0
    ):
        day_change = float(current_price) - float(previous_close)
        day_change_pct = (day_change / float(previous_close)) * 100

    col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
    col1.metric("Ticker", symbol)
    col2.metric("Company", company_name)
    col3.metric("Current Price", format_currency(current_price))

    if day_change is None or day_change_pct is None:
        col4.metric("Day Change", "N/A")
    else:
        # st.metric colors the delta green/red by sign automatically.
        col4.metric(
            "Day Change",
            f"{day_change:+.2f} USD",
            f"{day_change_pct:+.2f}%",
        )


def render_price_chart(history: pd.DataFrame) -> None:
    """Render a candlestick chart with SMA 50 and SMA 200 overlays."""
    hist = history.copy()

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
            )
        ]
    )
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(), name="SMA 50")
    )
    fig.add_trace(
        go.Scatter(x=hist.index, y=hist["Close"].rolling(200).mean(), name="SMA 200")
    )
    fig.update_layout(xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)


def render_key_stats(info: dict[str, Any]) -> None:
    """Render requested key statistics in a two-column metric grid."""
    st.subheader("Key Stats")
    left_col, right_col = st.columns(2)

    with left_col:
        st.metric("Market Cap", format_compact_number(info.get("marketCap")))
        st.metric("P/E Ratio", format_number(info.get("trailingPE")))
        st.metric("Forward P/E", format_number(info.get("forwardPE")))
        st.metric("EPS (TTM)", format_number(info.get("trailingEps")))
        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None and not pd.isna(dividend_yield):
            st.metric("Dividend Yield", format_percent(float(dividend_yield) * 100))
        else:
            st.metric("Dividend Yield", "N/A")

    with right_col:
        st.metric("52-Week High", format_currency(info.get("fiftyTwoWeekHigh")))
        st.metric("52-Week Low", format_currency(info.get("fiftyTwoWeekLow")))
        st.metric("Beta", format_number(info.get("beta")))
        st.metric("Average Volume", format_compact_number(info.get("averageVolume")))


def render_technical_indicators(indicator_history: pd.DataFrame) -> None:
    """Render RSI, MACD stance, and distances from SMA 50/200."""
    st.subheader("Technical Indicators")

    close_prices = indicator_history["Close"]
    sma_50 = close_prices.rolling(50).mean()
    sma_200 = close_prices.rolling(200).mean()

    rsi_series = rsi(close_prices, period=14)
    macd_line, signal_line = macd(close_prices, fast=12, slow=26, signal=9)

    latest_close = close_prices.iloc[-1] if not close_prices.empty else None
    latest_rsi = rsi_series.iloc[-1] if not rsi_series.empty else None
    latest_macd = macd_line.iloc[-1] if not macd_line.empty else None
    latest_signal = signal_line.iloc[-1] if not signal_line.empty else None
    latest_sma_50 = sma_50.iloc[-1] if not sma_50.empty else None
    latest_sma_200 = sma_200.iloc[-1] if not sma_200.empty else None

    distance_50 = compute_distance_from_ma_percent(latest_close, latest_sma_50)
    distance_200 = compute_distance_from_ma_percent(latest_close, latest_sma_200)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RSI (14)", format_number(latest_rsi), classify_rsi(latest_rsi))
    col2.metric("MACD (12, 26, 9)", classify_macd(latest_macd, latest_signal))
    col3.metric("Distance from 50D MA", format_percent(distance_50))
    col4.metric("Distance from 200D MA", format_percent(distance_200))


def render_news(news_items: list[dict[str, Any]]) -> None:
    """Render the last five ticker news items with headline, source, and date."""
    st.subheader("Recent News")

    if not news_items:
        st.info("No recent news items were returned for this ticker.")
        return

    recent_items = news_items[:5]
    for item in recent_items:
        title = item.get("title") or "Untitled"
        link = item.get("link") or ""
        publisher = item.get("publisher") or "Unknown Source"
        published_text = format_news_datetime(item.get("providerPublishTime"))

        headline_col, source_col, date_col = st.columns([6, 2, 2])
        with headline_col:
            if link:
                st.markdown(f"[{title}]({link})")
            else:
                st.write(title)
        with source_col:
            st.write(publisher)
        with date_col:
            st.write(published_text)


def run_app() -> None:
    """Run the Streamlit app layout and interactions."""
    st.set_page_config(page_title="Finance Hub", layout="wide")
    st.title("Finance Research Hub")

    ticker_input = st.text_input("Ticker Symbol", value="AAPL").strip().upper()

    if not ticker_input:
        st.info("Enter a ticker symbol (for example: AAPL, MSFT, SPY).")
        return

    selected_range = st.radio(
        "Time Range",
        options=list(TIME_RANGE_MAP.keys()),
        index=3,
        horizontal=True,
    )
    selected_period = TIME_RANGE_MAP[selected_range]

    try:
        info = fetch_ticker_info(ticker_input)
        price_history = fetch_price_history(ticker_input, selected_period)
        indicator_history = fetch_price_history(ticker_input, "1y")
        news_items = fetch_ticker_news(ticker_input)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    if price_history.empty:
        st.error(
            f"No price data was returned for '{ticker_input}'. Verify the symbol and try again."
        )
        return

    if indicator_history.empty:
        # Fallback keeps technicals functional when the 1Y request returns sparse data.
        indicator_history = price_history

    render_header(ticker_input, info, price_history)
    st.divider()

    render_price_chart(price_history)
    st.divider()

    render_key_stats(info)
    st.divider()

    render_technical_indicators(indicator_history)
    st.divider()

    render_news(news_items)


if __name__ == "__main__":
    run_app()
