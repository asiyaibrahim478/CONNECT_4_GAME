from connect_db import engine
from sqlalchemy import inspect
try:
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables in database: {tables}")
except Exception as e:
    print(f"Error connecting to database: {e}")
