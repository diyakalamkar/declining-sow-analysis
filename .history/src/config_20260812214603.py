from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP_FILE = RAW / "Category Code.csv"
PAYMENT_MAP_FILE  = RAW / "Payment Code.csv"
CUSTOMER_FILE     = RAW / "Customer Data.csv"
TXN_FILE          = RAW / "Transaction Data.csv"

DATE_FMT = "%d-%m-%Y"
ABC_PAYMENT_CODE = 3     # ABC Bank Credit Card