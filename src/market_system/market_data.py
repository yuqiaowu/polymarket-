from __future__ import annotations

import csv
import io
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from .config import CBOE_HISTORY_URLS, REFERENCE_SYMBOLS, STOOQ_SYMBOLS, TRADE_SYMBOLS
from .http import FetchError, get_text

try:
    import yfinance as yf
except Exception:  # noqa: BLE001
    yf = None


@dataclass
class Quote:
    symbol: str
    source_symbol: str
    date: Optional[str]
    time: Optional[str]
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    previous_close: Optional[float]
    volume: Optional[float]
    status: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def intraday_return_pct(self) -> Optional[float]:
        if self.open is None or self.close is None or self.open == 0:
            return None
        return (self.close / self.open - 1.0) * 100.0

    @property
    def day_return_pct(self) -> Optional[float]:
        if self.previous_close is None or self.close is None or self.previous_close == 0:
            return self.intraday_return_pct
        return (self.close / self.previous_close - 1.0) * 100.0


@dataclass
class PriceBar:
    date: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _to_float(value: str) -> Optional[float]:
    if value in {"", "N/D", "null", "None"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fetch_stooq_quote(symbol: str, timeout: int = 15) -> Quote:
    source_symbol = STOOQ_SYMBOLS[symbol]
    try:
        text = get_text(
            "https://stooq.com/q/l/",
            params={"s": source_symbol, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
            timeout=timeout,
            ttl_seconds=60,
        )
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            return Quote(symbol, source_symbol, None, None, None, None, None, None, None, None, "ERROR", "empty_csv")
        row = rows[0]
        close = _to_float(row.get("Close", ""))
        status = "OK" if close is not None else "UNAVAILABLE"
        return Quote(
            symbol=symbol,
            source_symbol=source_symbol,
            date=None if row.get("Date") == "N/D" else row.get("Date"),
            time=None if row.get("Time") == "N/D" else row.get("Time"),
            open=_to_float(row.get("Open", "")),
            high=_to_float(row.get("High", "")),
            low=_to_float(row.get("Low", "")),
            close=close,
            previous_close=None,
            volume=_to_float(row.get("Volume", "")),
            status=status,
            error=None if status == "OK" else "no_quote",
        )
    except FetchError as exc:
        return Quote(symbol, source_symbol, None, None, None, None, None, None, None, None, "ERROR", str(exc))


def fetch_cboe_history_quote(symbol: str, timeout: int = 15) -> Quote:
    url = CBOE_HISTORY_URLS[symbol]
    try:
        text = get_text(url, timeout=timeout, ttl_seconds=3600)
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            return Quote(symbol, url, None, None, None, None, None, None, None, None, "ERROR", "empty_csv")
        row = rows[-1]
        return Quote(
            symbol=symbol,
            source_symbol=url,
            date=row.get("DATE"),
            time="CBOE_DAILY",
            open=_to_float(row.get("OPEN", "")),
            high=_to_float(row.get("HIGH", "")),
            low=_to_float(row.get("LOW", "")),
            close=_to_float(row.get("CLOSE", "")),
            previous_close=None,
            volume=None,
            status="OK" if _to_float(row.get("CLOSE", "")) is not None else "UNAVAILABLE",
            error=None,
        )
    except FetchError as exc:
        return Quote(symbol, url, None, None, None, None, None, None, None, None, "ERROR", str(exc))


def fetch_yfinance_vix(timeout: int = 15) -> Quote:
    if yf is None:
        return Quote("VIX", "^VIX", None, None, None, None, None, None, None, None, "ERROR", "yfinance_not_installed")
    try:
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(period="2d", interval="5m", timeout=timeout)
        if hist.empty:
            hist = ticker.history(period="5d", interval="1d", timeout=timeout)
        if hist.empty:
            return Quote("VIX", "^VIX", None, None, None, None, None, None, None, None, "UNAVAILABLE", "empty_yfinance_history")
        row = hist.iloc[-1]
        idx = hist.index[-1]
        date = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else None
        time = idx.strftime("%H:%M:%S %Z") if hasattr(idx, "strftime") else "YFINANCE"
        return Quote(
            symbol="VIX",
            source_symbol="^VIX",
            date=date,
            time=time,
            open=float(row["Open"]) if row.get("Open") is not None else None,
            high=float(row["High"]) if row.get("High") is not None else None,
            low=float(row["Low"]) if row.get("Low") is not None else None,
            close=float(row["Close"]) if row.get("Close") is not None else None,
            previous_close=None,
            volume=float(row["Volume"]) if row.get("Volume") is not None else None,
            status="OK",
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return Quote("VIX", "^VIX", None, None, None, None, None, None, None, None, "ERROR", str(exc))


def fetch_yfinance_equity_quote(symbol: str, timeout: int = 15) -> Quote:
    if yf is None:
        return Quote(symbol, symbol, None, None, None, None, None, None, None, None, "ERROR", "yfinance_not_installed")
    try:
        ticker = yf.Ticker(symbol)
        intraday = ticker.history(period="2d", interval="5m", timeout=timeout)
        daily = ticker.history(period="7d", interval="1d", timeout=timeout)
        if intraday.empty and daily.empty:
            return Quote(symbol, symbol, None, None, None, None, None, None, None, None, "UNAVAILABLE", "empty_yfinance_history")

        if not intraday.empty:
            row = intraday.iloc[-1]
            idx = intraday.index[-1]
            close = float(row["Close"]) if row.get("Close") is not None else None
            high = float(row["High"]) if row.get("High") is not None else None
            low = float(row["Low"]) if row.get("Low") is not None else None
            volume = float(row["Volume"]) if row.get("Volume") is not None else None
            date = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else None
            time = idx.strftime("%H:%M:%S %Z") if hasattr(idx, "strftime") else "YFINANCE"
        else:
            row = daily.iloc[-1]
            idx = daily.index[-1]
            close = float(row["Close"]) if row.get("Close") is not None else None
            high = float(row["High"]) if row.get("High") is not None else None
            low = float(row["Low"]) if row.get("Low") is not None else None
            volume = float(row["Volume"]) if row.get("Volume") is not None else None
            date = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else None
            time = "YFINANCE_DAILY"

        daily_open = None
        previous_close = None
        if not daily.empty:
            today = daily.iloc[-1]
            daily_open = float(today["Open"]) if today.get("Open") is not None else None
            if len(daily) >= 2:
                previous = daily.iloc[-2]
                previous_close = float(previous["Close"]) if previous.get("Close") is not None else None

        return Quote(
            symbol=symbol,
            source_symbol=symbol,
            date=date,
            time=time,
            open=daily_open,
            high=high,
            low=low,
            close=close,
            previous_close=previous_close,
            volume=volume,
            status="OK" if close is not None else "UNAVAILABLE",
            error=None if close is not None else "no_quote",
        )
    except Exception as exc:  # noqa: BLE001
        return Quote(symbol, symbol, None, None, None, None, None, None, None, None, "ERROR", str(exc))


def fetch_yfinance_daily_bars(symbol: str, period: str = "3mo", timeout: int = 15) -> List[PriceBar]:
    if yf is None:
        return []
    try:
        hist = yf.Ticker(symbol).history(period=period, interval="1d", timeout=timeout)
    except Exception:  # noqa: BLE001
        return []
    if hist.empty:
        return []
    bars = []
    for idx, row in hist.iterrows():
        close = row.get("Close")
        if close is None:
            continue
        bars.append(
            PriceBar(
                date=idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                open=float(row["Open"]) if row.get("Open") is not None else None,
                high=float(row["High"]) if row.get("High") is not None else None,
                low=float(row["Low"]) if row.get("Low") is not None else None,
                close=float(close),
                volume=float(row["Volume"]) if row.get("Volume") is not None else None,
            )
        )
    return bars


def fetch_daily_bars(symbols: List[str], period: str = "3mo", timeout: int = 15) -> Dict[str, List[PriceBar]]:
    output: Dict[str, List[PriceBar]] = {}
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(symbols)))) as executor:
        futures = {executor.submit(fetch_yfinance_daily_bars, symbol, period, timeout): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                output[symbol] = future.result()
            except Exception:  # noqa: BLE001
                output[symbol] = []
    return {symbol: output.get(symbol, []) for symbol in symbols}


def fetch_quote(symbol: str, timeout: int = 15) -> Quote:
    if symbol == "VIX":
        intraday = fetch_yfinance_vix(timeout=timeout)
        if intraday.status == "OK" and intraday.close is not None:
            return intraday
        fallback = fetch_cboe_history_quote(symbol, timeout=timeout)
        if fallback.status != "OK":
            return intraday
        fallback.error = f"fallback_after_yfinance_error:{intraday.error}"
        return fallback
    if symbol in CBOE_HISTORY_URLS:
        return fetch_cboe_history_quote(symbol, timeout=timeout)
    if symbol in TRADE_SYMBOLS or symbol in REFERENCE_SYMBOLS:
        quote = fetch_yfinance_equity_quote(symbol, timeout=timeout)
        if quote.status == "OK" and quote.close is not None:
            return quote
    return fetch_stooq_quote(symbol, timeout=timeout)


def fetch_quotes(symbols: List[str], timeout: int = 15) -> Dict[str, Quote]:
    quotes: Dict[str, Quote] = {}
    with ThreadPoolExecutor(max_workers=min(12, max(1, len(symbols)))) as executor:
        futures = {executor.submit(fetch_quote, symbol, timeout): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                quotes[symbol] = future.result()
            except Exception as exc:  # noqa: BLE001
                source_symbol = STOOQ_SYMBOLS.get(symbol) or CBOE_HISTORY_URLS.get(symbol, symbol)
                quotes[symbol] = Quote(symbol, source_symbol, None, None, None, None, None, None, None, None, "ERROR", str(exc))
    return {symbol: quotes[symbol] for symbol in symbols}


def data_status(quotes: Dict[str, Quote]) -> dict:
    ok = [symbol for symbol, quote in quotes.items() if quote.status == "OK"]
    bad = {symbol: quote.error for symbol, quote in quotes.items() if quote.status != "OK"}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok_count": len(ok),
        "bad_count": len(bad),
        "ok_symbols": ok,
        "bad_symbols": bad,
    }
