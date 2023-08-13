import pandas as pd

class DataValidator:
    def __init__(self):
        pass

    def remove_duplicates_by_id(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        initial_length = len(dataframe)
        dataframe.drop_duplicates(subset='Id', keep='first', inplace=True)
        final_length = len(dataframe)

        duplicates_count = initial_length - final_length

        if duplicates_count > 0:
            print(f"Found and removed {duplicates_count} duplicates based on 'Id'.")
        else:
            print("No duplicates found based on 'Id'.")

        return dataframe
