from typing import List
from pymongo import MongoClient
import pandas as pd
import uuid
import json

class MongoDBToDataFrame:
    def __init__(self, config_file: str):
        self.config = self._read_config(config_file)
        self.client = None

    def _read_config(self, config_file: str) -> dict:
        with open(config_file) as file:
            config = json.load(file)
        return config

    def connect(self):
        self.client = MongoClient(self.config["connection_string"])

    def disconnect(self):
        self.client.close()
        self.client = None

    # connect, convert_to_df, disconnect
    def convert_to_df(self, database_name: str, collection_name: str, id_columns: List[str], fields: List[str] = None,
                      disconnection: bool = True) -> pd.DataFrame:
        if self.client is None:
            self.connect()
        db = self.client[database_name]
        collection = db[collection_name]

        # Если fields передан, создаем словарь проекции
        projection = None
        if fields:
            projection = {field: 1 for field in fields}

        df = pd.DataFrame(list(collection.find({}, projection)))
        df = self._convert_ids_to_uuid(df, id_columns)
        if disconnection:
            self.disconnect()

        return df

    @staticmethod
    def _convert_ids_to_uuid(df: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
        def convert_to_uuid(id_value):
            if isinstance(id_value, uuid.UUID):
                return id_value
            elif isinstance(id_value, bytes) and len(id_value) == 16:  # Добавил проверку на bytes
                return uuid.UUID(bytes=id_value)
            else:
                return None  # или любое другое значение по умолчанию

        for col in id_columns:
            df[col] = df[col].apply(convert_to_uuid)
            df[col] = df[col].astype(str)

        return df

