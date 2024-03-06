import pymssql
import pandas as pd


def ms_sql_con():
    sql_name = "voice_ai"
    sql_server = "10.2.4.124"
    sql_login = "ICECORP\\1c_sql"

    with open("sql.pass", "r") as file:
        sql_pass = file.read().replace("\n", "")
        file.close()

    return pymssql.connect(
        server=sql_server,
        user=sql_login,
        password=sql_pass,
        database=sql_name,
        tds_version=r"7.0",
        charset="cp1251",
    )


def read_sql(query):
    return pd.read_sql_query(query, con=ms_sql_con(), parse_dates=None)
