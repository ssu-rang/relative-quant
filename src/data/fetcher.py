from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import pandas as pd
import yfinance as yf


class DataFetcher:

    STANDARD_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

    @staticmethod
    def _validate_period(start: str, end: str) -> None:
        try:
            start_date = pd.Timestamp(start)
            end_date = pd.Timestamp(end)
        except (TypeError, ValueError) as exc:
            raise ValueError("start와 end는 유효한 날짜여야 합니다.") from exc
        if start_date >= end_date:
            raise ValueError("start는 end보다 이전 날짜여야 합니다.")

    @classmethod
    def _normalize(cls, raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if raw.empty:
            raise ValueError(
                f"'{ticker}' 종목의 데이터를 가져올 수 없습니다. "
                "종목 코드와 날짜 범위를 확인하세요."
            )

        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            ticker_level = next(
                (
                    level
                    for level in range(frame.columns.nlevels)
                    if ticker in frame.columns.get_level_values(level)
                ),
                None,
            )
            if ticker_level is None:
                raise ValueError(f"응답에 '{ticker}' 종목 데이터가 없습니다.")
            frame = frame.xs(ticker, axis=1, level=ticker_level, drop_level=True)

        frame.columns = [str(column).strip().lower() for column in frame.columns]
        missing = set(cls.STANDARD_COLUMNS[1:]) - set(frame.columns)
        if missing:
            raise ValueError(f"'{ticker}' 응답에 다음 컬럼이 없습니다: {sorted(missing)}")

        frame = frame.reset_index()
        frame = frame.rename(columns={frame.columns[0]: "date"})
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        return frame[cls.STANDARD_COLUMNS].sort_values("date").reset_index(drop=True)

    def fetch_yfinance(
        self, ticker: str, start: str, end: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch one ticker. Yahoo's end date is exclusive."""
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker는 비어 있을 수 없습니다.")
        self._validate_period(start, end)

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return self._normalize(raw, ticker)

    def fetch_csv(self, filepath: str | PathLike[str]) -> pd.DataFrame:
        """Read an OHLCV CSV and return rows ordered by date."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")

        frame = pd.read_csv(path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        missing = set(self.STANDARD_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"CSV 파일에 다음 컬럼이 없습니다: {sorted(missing)}")

        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        return frame[self.STANDARD_COLUMNS].sort_values("date").reset_index(drop=True)

    def fetch_multiple(
        self,
        tickers: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        normalized = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
        if not normalized or any(not ticker for ticker in normalized):
            raise ValueError("tickers에는 하나 이상의 유효한 종목 코드가 필요합니다.")
        self._validate_period(start, end)

        raw = yf.download(
            normalized,
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        return {ticker: self._normalize(raw, ticker) for ticker in normalized}
