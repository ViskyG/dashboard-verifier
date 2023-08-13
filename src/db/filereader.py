import pandas as pd

class Filereader():

    def read_csv_in_chunks(self, file_path, chunk_size=100000):
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
            #show chunk number
            print("chunk number: ", len(chunks))
        return pd.concat(chunks)

    def get_df_from_csv(self, files: list):
        df_dict = {}
        dir = 'files\\'
        for file in files:
            df_dict[file] = self.read_csv_in_chunks(dir + file)
        return df_dict



class FileReaderTest():

    def read_csv_in_chunks(self, file_path, chunk_size=1000): # Reduced chunk_size for testing
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunks.append(chunk)
            #show chunk number
            print("chunk number: ", len(chunks))
            if len(chunks) * chunk_size >= 10000:  # Stop after reading 10000 lines
                break
        return pd.concat(chunks)

    def get_df_from_csv(self, files: list):
        df_dict = {}
        dir = 'files\\'
        for file in files:
            df_dict[file] = self.read_csv_in_chunks(dir + file)
        return df_dict
