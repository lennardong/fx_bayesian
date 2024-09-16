import configparser
import os
import sqlite3
from contextlib import contextmanager
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.sharedutils import find_project_root

#########################


class AlphaVantageDataManager:
    """
    Manages foreign exchange (FX) data for a given currency pair.
    Handles data fetching and storage and various transformations.
    """

    """
    Initialize the FXDataManager with currency symbols and set up the database.
    
    :param from_symbol: The base currency symbol.
    :param to_symbol: The quote currency symbol.
    """

    def __init__(self, from_symbol: str = "SGD", to_symbol: str = "SGD"):
        self.from_symbol = from_symbol
        self.to_symbol = to_symbol
        self.config = self._load_config()
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except Exception as e:
            print(f"Error creating directory for database: {e}")
            raise

    @property
    def db_path(self):
        relative_path = self.config.get("DATABASE", "path")
        project_root = find_project_root()
        return str(project_root / relative_path)

    @property
    def df(self) -> pd.DataFrame:
        """
        Get the FX data as a DataFrame.
        """
        self._poll_source_and_refresh_db()
        df = self._pull_df_from_db()
        return df

    ##############################

    # PRIVATE METHODS

    def _load_config(self) -> configparser.ConfigParser:
        """Load the configuration from the config file.

        Returns:
            Loaded configuration.
        """
        config = configparser.ConfigParser()
        project_root = find_project_root()
        config.read(project_root / "config.ini")
        print("Loaded config sections:", config.sections())
        return config

    @contextmanager
    def _get_connection(self):
        """Context manager for database connection.

        Yields:
            SQLite connection object.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _poll_source_and_refresh_db(self) -> None:
        """Refresh the database by fetching and inserting FX data."""

        # HELPER FUNCTIONS

        def create_fx_table() -> None:
            """Create the FX data table if it doesn't exist."""
            with self._get_connection() as conn:
                # Check if the table exists and has the correct structure
                cursor = conn.execute("SELECT * FROM fx_data LIMIT 1")

                columns = [description[0] for description in cursor.description]
                if set(columns) != {
                    "date",
                    "from_symbol",
                    "to_symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                }:
                    # If the table doesn't have the correct structure, drop it and recreate
                    conn.execute("DROP TABLE IF EXISTS fx_data")
                    conn.execute(
                        """
                        CREATE TABLE fx_data (
                            date TEXT,
                            from_symbol TEXT,
                            to_symbol TEXT,
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL,
                            PRIMARY KEY (date, from_symbol, to_symbol)
                        )
                        """
                    )

        def is_today_pulled() -> bool:
            """Check if data for today already exists in the database."""
            today = date.today().isoformat()
            with self._get_connection() as conn:
                # Execute a SQL query to check if today's data exists
                cursor = conn.execute(
                    """
                    SELECT 1 FROM fx_data 
                    WHERE date = ? 
                    AND from_symbol = ? 
                    AND to_symbol = ?
                    """,
                    (today, self.from_symbol, self.to_symbol),
                )

                return cursor.fetchone() is not None

        def fetch_fx_data() -> Dict[str, Any]:
            """Fetch FX data from the API.

            Returns:
                JSON response containing FX data.
            """
            # Prepare API request parameters
            parameters = {
                "function": "FX_DAILY",
                "from_symbol": self.from_symbol,
                "to_symbol": self.to_symbol,
                "datatype": "json",
                "apikey": self.config["API"]["key"],
            }
            # Make API request
            response = requests.get(self.config["API"]["url"], params=parameters)
            response.raise_for_status()
            return response.json()

        def insert_fx_data(time_series: Dict[str, Dict[str, str]]) -> None:
            """Insert FX data into the database."""
            with self._get_connection() as conn:
                cursor = conn.executemany(
                    """
                    INSERT OR REPLACE INTO fx_data
                    (date, from_symbol, to_symbol, open, high, low, close)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            date,
                            self.from_symbol,
                            self.to_symbol,
                            values["1. open"],
                            values["2. high"],
                            values["3. low"],
                            values["4. close"],
                        )
                        for date, values in time_series.items()
                    ],
                )
                conn.commit()
                print(f"Inserted {cursor.rowcount} rows into the database.")

        # LOGIC
        create_fx_table()
        # Fetch new data if today's data is not available
        if not is_today_pulled():
            print("Fetching new data...")
            try:
                data = fetch_fx_data()
                time_series = data["Time Series FX (Daily)"]
                insert_fx_data(time_series)
            except requests.RequestException as e:
                print(f"Error fetching data: {e}")
                return pd.DataFrame()
        else:
            print("Data already pulled for today. Generating DF")

    def _pull_df_from_db(
        self,
    ) -> pd.DataFrame:
        """Main method to retrieve and process FX data.

        Returns:
            Processed DataFrame with FX data.
        """

        # HELPER FUNCTIONS

        def _get_fx_dataframe() -> pd.DataFrame:
            """Retrieve FX data from the database as a DataFrame."""
            with self._get_connection() as conn:
                return pd.read_sql_query(
                    """
                    SELECT * FROM fx_data
                    WHERE from_symbol = ? AND to_symbol = ?
                    """,
                    conn,
                    params=(self.from_symbol, self.to_symbol),
                )

        def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            # Convert date to datetime and set as index
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(ascending=False, inplace=True)

            return df

        # LOGIC

        # Retrieve and process the data
        df = _get_fx_dataframe()
        processed_df = _process_dataframe(df)

        # Print summary of processed data
        print(f"Processed DF for {self.from_symbol}/{self.to_symbol}: ")
        print(processed_df.head())
        print(processed_df.tail())

        return processed_df

    ###############################

    # PUBLIC METHODS

    def compute_volatility(self, window: int = 7):
        """Compute the volatility of the FX data.
        Args:
            window: The rolling window size for volatility calculation.
        """
        # Fetch the FX data
        df = self.df
        # Calculate log returns and volatility
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_return"].rolling(window=window).std()
        return df

    def to_weekly(self):
        df = self.df.resample("W").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        df.reset_index(inplace=True)
        return df

    def set_currency_pair(self, from_symbol, to_symbol):
        """Set the currency pair for the FX data manager and  refreshes the DB with  the latest data"""
        self.from_symbol = from_symbol
        self.to_symbol = to_symbol
        self._poll_source_and_refresh_db()


# Usage
if __name__ == "__main__":
    fx_manager = AlphaVantageDataManager("SGD", "MYR")
    df = fx_manager.df
