from dataclasses import dataclass
from datetime import datetime

import seaborn as sns

import src.chartmanager as cm
from src import *
from src.datamanager import AlphaVantageDataManager

"""
The inputs will be a 

"""


@dataclass
class _MomentumCols:
    high: str
    low: str
    open: str
    close: str


@dataclass
class _ResampleCols:
    first: str
    max: str
    min: str
    last: str
    sum: str


def compute_relative_strength_index(
    df: pd.DataFrame, col: ColName, window: int = 14
) -> Dict[datetime, float]:
    """
    Compute the Relative Strength Index (RSI) for a given DataFrame column.

    The RSI is a momentum indicator that measures the magnitude of recent price
    changes to evaluate overbought or oversold conditions in the price of a stock
    or other asset.

    Calculation:
     1. Calculate price changes:
        - Compute the difference between the current price and the previous price for each period.
        - This gives us a series of price changes over time.

     2. Calculate average gain and average loss:
        - Separate the price changes into gains (positive changes) and losses (negative changes).
        - Calculate the average gain over the specified window (default 14 periods).
        - Calculate the average loss over the same window.

     3. Calculate relative strength (RS):
        - RS = Average Gain / Average Loss
        - This ratio compares the strength of upward price movements to downward price movements.

     4. Calculate RSI: 100 - (100 / (1 + RS))
        - Convert the RS value to a scale of 0 to 100.
        - RSI values above 70 typically indicate overbought conditions.
        - RSI values below 30 typically indicate oversold conditions.

     5. Interpretation:
        - RSI oscillates between 0 and 100.
        - Traditional interpretation suggests:
          * RSI > 70: potentially overbought
          * RSI < 30: potentially oversold
        - Crossovers, divergences, and failure swings can also be analyzed for trading signals.

     6. Considerations:
        - The default period is 14, but this can be adjusted based on the asset and trading style.
        - RSI is most effective in ranging markets and may give false signals in strong trends.
        - It's often used in conjunction with other technical indicators for confirmation.

    Args:
        df (pd.DataFrame): Input DataFrame
        col (ColName): Column name to compute RSI for
        window (int): The lookback period to compute RSI, default is 14

    Returns:
        Dict[datetime, float]: A dictionary of datetime keys and RSI values
    """
    delta = df[col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Shift the RSI values to make it forward-biased
    forward_rsi = rsi.shift(-window + 1)

    return forward_rsi.to_dict()


def compute_rolling_average(
    df: pd.DataFrame, col: ColName, window: int, *, forward: bool = False
) -> Dict[datetime, float]:
    """
    Compute the rolling average for a given DataFrame column.

    This function calculates either a backward-looking (traditional) or
    forward-looking rolling average based on the 'forward' parameter.

    Backward-looking averages are typically used for trend identification and
    smoothing historical data. They're useful in technical analysis and
    identifying support/resistance levels.

    Forward-looking averages can be used for forecasting and anticipating future
    price movements. They're particularly useful in predictive models and for
    estimating future trend directions.

    Args:
        df (pd.DataFrame): Input DataFrame
        col (ColName): Column name to compute rolling average for
        window (int): The rolling window size
        forward (bool): If True, compute forward-looking average. Default is False.

    Returns:
        Dict[datetime, float]: A dictionary of datetime keys and rolling average values
    """
    sorted_df = df.sort_index()
    if forward:
        rolling_avg = (
            sorted_df[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(-window + 1)
        )
    else:
        rolling_avg = sorted_df[col].rolling(window=window, min_periods=1).mean()
    return rolling_avg.to_dict()


def compute_volatility(df, col: ColName, window: int) -> Dict[datetime, float]:
    pass


@dataclass
class MovingAverageStrategy:
    df: pd.DataFrame
    short_window: int = 3
    long_window: int = 8
    col: ColName = ColName.CLOSE

    def __post_init__(self):
        self.df["short_ma"] = self.df[self.col].rolling(window=self.short_window).mean()
        self.df["long_ma"] = self.df[self.col].rolling(window=self.long_window).mean()

    @property
    def signal(self):
        self.df["signal"] = 0
        self.df.loc[self.df["short_ma"] > self.df["long_ma"], "signal"] = 1
        self.df.loc[self.df["short_ma"] < self.df["long_ma"], "signal"] = -1
        return self.df["signal"]

    def generate_trades(self):
        self.df["position"] = self.signal.diff()
        return self.df[self.df["position"] != 0].copy()

    def calculate_returns(self):
        self.df["returns"] = self.df[self.col].pct_change()
        self.df["strategy_returns"] = self.df["returns"] * self.df["signal"].shift(1)
        return self.df["strategy_returns"].cumsum()


if __name__ == "__main__":
    df = AlphaVantageDataManager("SGD", "MYR").df
    print(compute_rolling_average(df, ColName.CLOSE, 5))
    print(compute_rolling_average(df, ColName.CLOSE, 5, forward=True))
    pass
