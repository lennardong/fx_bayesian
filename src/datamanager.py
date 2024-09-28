import configparser
import os
import sqlite3

import seaborn as sns

from src import *
from src._singletons import *

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
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        except Exception as e:
            print(f"Error creating directory for database: {e}")
            raise

    @property
    def db_path(self):
        relative_path = ProjectVar.DB_PATH
        project_root = find_project_root()
        return str(project_root / relative_path)

    @property
    def df(self) -> pd.DataFrame:
        """
        Get the FX data as a DataFrame.
        """
        return self._pull_df_from_db()

    ##############################

    # PRIVATE METHODS

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

    def query_endpoint(self) -> None:
        """Refresh the database by fetching and inserting FX data."""
        # TODO fix refresh to ACTUALLY WORK. currently retrieves data from the API regardless if its already there

        # HELPER FUNCTIONS

        def create_fx_table() -> None:
            """Create the FX data table if it doesn't exist."""
            TABLE_SCHEMA = f"""
                CREATE TABLE IF NOT EXISTS {ProjectVar.TABLE_NAME} (
                    {ColName.DATE} TEXT,
                    {ColName.SYMBOL_FROM} TEXT,
                    {ColName.SYMBOL_TO} TEXT,
                    {ColName.OPEN} FLOAT,
                    {ColName.HIGH} FLOAT,
                    {ColName.LOW} FLOAT,
                    {ColName.CLOSE} FLOAT,
                    PRIMARY KEY ({ColName.DATE}, {ColName.SYMBOL_FROM}, {ColName.SYMBOL_TO})
                )
            """
            with self._get_connection() as conn:
                conn.execute(TABLE_SCHEMA)
                print(f"Table {ProjectVar.TABLE_NAME} created or already exists.")

        def is_today_pulled() -> bool:
            today = datetime.now().date().strftime("%Y-%m-%d")
            print(f"Debug: Checking for data on {today}")
            with self._get_connection() as conn:
                cursor = conn.execute(
                    f"""
                    SELECT 1 FROM {ProjectVar.TABLE_NAME}
                    WHERE {ColName.DATE} = ? 
                    AND {ColName.SYMBOL_FROM} = ? 
                    AND {ColName.SYMBOL_TO}= ?
                    """,
                    (today, self.from_symbol, self.to_symbol),
                )
                result = cursor.fetchone() is not None
                print(f"Debug: Data for {today} exists: {result}")
                return result

        def fetch_fx_data() -> Dict[str, Any]:
            """Fetch FX data from the API and convert dates to datetime objects.

            Returns:
                Dict containing FX data with datetime objects as keys.

            # TODO: amend this to later use Pydantic for dtype vbalidation before passing to DB
            """
            parameters = {
                "function": "FX_DAILY",
                "from_symbol": self.from_symbol,
                "to_symbol": self.to_symbol,
                "datatype": "json",
                "apikey": ProjectVar.KEY,
            }
            response = requests.get(ProjectVar.API_URL, params=parameters)
            response.raise_for_status()
            data = response.json()

            # # Convert string dates to datetime objects
            # time_series = data["Time Series FX (Daily)"]
            # converted_time_series = {
            #     datetime.strptime(date, "%Y-%m-%d"): values
            #     for date, values in time_series.items()
            # }
            # data["Time Series FX (Daily)"] = converted_time_series

            return data

        def insert_fx_data(time_series: Dict[str, Dict[str, str]]) -> None:
            """Insert FX data into the database."""
            with self._get_connection() as conn:
                cursor = conn.executemany(
                    f"""
                    INSERT OR REPLACE INTO {ProjectVar.TABLE_NAME}
                    ({ColName.DATE}, {ColName.SYMBOL_FROM}, {ColName.SYMBOL_TO}, {ColName.OPEN}, {ColName.HIGH}, {ColName.LOW}, {ColName.CLOSE})
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            date,  # Convert datetime to string
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
                    f"""
                    SELECT * FROM {ProjectVar.TABLE_NAME}
                    WHERE from_symbol = ? AND to_symbol = ?
                    """,
                    conn,
                    params=(self.from_symbol, self.to_symbol),
                )

        def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            # Convert date to datetime and set as index
            df[ColName.DATE] = pd.to_datetime(df[ColName.DATE])
            df.set_index(ColName.DATE, inplace=True)
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


# Usage
if __name__ == "__main__":
    fx_manager = AlphaVantageDataManager("SGD", "MYR")
    df = fx_manager.df
