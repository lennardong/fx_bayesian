# CONFIGS


# TABLES
class ColName:
    """
    A class to manage column names used in the project.
    """

    DATE = "date"
    SYMBOL_FROM = "from_symbol"
    SYMBOL_TO = "to_symbol"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"


class ProjectVar:
    API_URL = "https://www.alphavantage.co/query"
    KEY = "0SBRTSPM95BI6E5J"
    DB_PATH = "data/sqlite.db"
    TABLE_NAME = "fx_data"
