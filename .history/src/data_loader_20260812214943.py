import pandas as pd
from . import config as cfg

def load_raw():
    cat = pd.read_csv(cfg.CATEGORY_MAP_FILE)
    pay = pd.read_csv(cfg.PAYMENT_MAP_FILE)
    cust = pd.read_csv(cfg.CUSTOMER_FILE)
    txn = pd.read_csv(cfg.TXN_FILE)
    return cat, pay, cust, txn