from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP_FILE = RAW / "Category Code.csv"
PAYMENT_MAP_FILE  = RAW / "Payment Code.csv"
CUSTOMER_FILE     = RAW / "Customer Data.csv"
TXN_FILE          = RAW / "Transaction Data.csv"

COL_CUSTOMER_ID   = "Customer_ID"
COL_MEMBERSHIP    = "Membership_Type"      # was "Members"
COL_OPEN_DATE     = "Credit_Card_Open_Date"
COL_CLOSE_DATE    = "Credit_Card_Closed_Date"   # was "Credit_Card_Close_Date"
COL_LIMIT         = "Credit_Card_Limit"
COL_APR           = "Credit_Card_APR"

DATE_FMT = "%d-%m-%Y"
ABC_PAYMENT_CODE = 3     # ABC Bank Credit Card