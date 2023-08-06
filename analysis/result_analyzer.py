import pandas as pd

class ResultAnalyzer:
    def __init__(self):
        pass

    def select_objects_by_type(self, objects_df, selected_object_type):
        selected_objects_df = objects_df[objects_df["ObjectType"] == selected_object_type]
        return selected_objects_df

    def get_transformed_values(self, selected_objects_df, transformed_value_df):
        selected_transformed_values_all = transformed_value_df[
            transformed_value_df['ObjectId'].isin(selected_objects_df['Id'])]
        return selected_transformed_values_all

    def check_data_availability(self, selected_transformed_values_all, selected_object_type):
        if selected_transformed_values_all.empty:
            return f"Для выбранного типа объектов '{selected_object_type}' нет данных для соответствующих пользователей.\n"
        else:
            return ''

    def merge_and_prepare_tables(self, selected_transformed_values_all, selected_objects_df):
        result_all_df = pd.merge(selected_transformed_values_all, selected_objects_df[['Id', 'Name']],
                                 left_on='ObjectId', right_on='Id', suffixes=('_Transformed', '_Object'))
        result_all_df.drop(columns='Id', inplace=True, errors='ignore')
        return result_all_df

    def calculate_percentage_of_total_users(self, result_all_df, transformed_value_df):
        total_users = len(result_all_df['UserId'].unique())
        percentage_of_total_users = (total_users / len(transformed_value_df['UserId'].unique())) * 100
        return percentage_of_total_users

    def process_transformed_values(self, result_all_df, threshold_value, objects_df):
        selected_transformed_values = result_all_df[result_all_df['TransformedValue'] > threshold_value]
        grouped_counts = selected_transformed_values.groupby('ObjectId')['UserId'].nunique().reset_index()
        grouped_counts.columns = ['ObjectId', 'Unique_UserId_Count']
        total_users = len(result_all_df['UserId'].unique())
        grouped_counts['Percentage'] = (grouped_counts['Unique_UserId_Count'] / total_users) * 100
        grouped_counts = pd.merge(grouped_counts, objects_df[['Id', 'Name']], left_on='ObjectId', right_on='Id')
        grouped_counts.drop(columns='Id', inplace=True)
        return grouped_counts

    def analyze_results(self, objects_df, transformed_value_df, selected_object_type, threshold_value):
        output_string = ''

        # Шаг 1
        selected_objects_df = self.select_objects_by_type(objects_df, selected_object_type)
        output_string += f"<details><summary>Шаг 1. Данные из таблицы Objects для выбранного типа ObjectType: {selected_object_type}</summary>\n"
        output_string += f"<p>{selected_objects_df.to_html()}</p></details>\n"

        # Шаг 2
        selected_transformed_values_all = self.get_transformed_values(selected_objects_df, transformed_value_df)
        data_availability_message = self.check_data_availability(selected_transformed_values_all, selected_object_type)
        if data_availability_message:
            return None, data_availability_message, output_string

        result_all_df = self.merge_and_prepare_tables(selected_transformed_values_all, selected_objects_df)
        output_string += f'<details><summary>Шаг 2. Все данные из таблицы TransformedValue для выбранного типа ObjectType: {selected_object_type}</summary>'
        output_string += f"<p>{result_all_df.head(100).to_html()}</p></details>\n"

        # Шаг 3
        percentage_of_total_users = self.calculate_percentage_of_total_users(result_all_df, transformed_value_df)
        output_string += f"<details><summary>Шаг 3. Процент пользователей, чьи параметры TransformValue превосходят порог  хотя бы для одного объекта выбранного типа ObjectType:</summary>\n"
        output_string += f"<p>Percentage of Users with TransformedValue > Threshold: {percentage_of_total_users:.2f}%</p></details>\n"

        # Шаг 4
        grouped_counts = self.process_transformed_values(result_all_df, threshold_value, objects_df)
        output_string += f"<details><summary>Шаг 4. Распределение по процентам и число уникальных пользователей для каждого объекта выбранного типа ObjectType: {selected_object_type}, где показатель TransformedValue превысил порог:</summary>\n"
        output_string += f"<p>{grouped_counts.to_html()}</p></details>\n"

        return grouped_counts, percentage_of_total_users, output_string




# class ResultAnalyzer:
#
#
#     def __init__(self, objects_df, transformed_value_df):
#         self.objects_df = objects_df
#         self.transformed_value_df = transformed_value_df
#
#     def select_objects_by_type(self, selected_object_type):
#         selected_objects_df = self.objects_df[self.objects_df["ObjectType"] == selected_object_type]
#         return selected_objects_df
#
#     def get_transformed_values(self, selected_objects_df):
#         selected_transformed_values_all = self.transformed_value_df[
#             self.transformed_value_df['ObjectId'].isin(selected_objects_df['Id'])]
#         return selected_transformed_values_all
#
#     def check_data_availability(self, selected_transformed_values_all, selected_object_type):
#         if selected_transformed_values_all.empty:
#             return f"Для выбранного типа объектов '{selected_object_type}' нет данных для соответствующих пользователей.\n"
#         else:
#             return ''
#
#     def merge_and_prepare_tables(self, selected_transformed_values_all, selected_objects_df):
#         result_all_df = pd.merge(selected_transformed_values_all, selected_objects_df[['Id', 'Name']],
#                                  left_on='ObjectId', right_on='Id', suffixes=('_Transformed', '_Object'))
#         result_all_df.drop(columns='Id', inplace=True, errors='ignore')
#         return result_all_df
#
#     def calculate_percentage_of_total_users(self, result_all_df):
#         total_users = len(result_all_df['UserId'].unique())
#         percentage_of_total_users = (total_users / len(self.transformed_value_df['UserId'].unique())) * 100
#         return percentage_of_total_users
#
#     def process_transformed_values(self, result_all_df, threshold_value):
#         selected_transformed_values = result_all_df[result_all_df['TransformedValue'] > threshold_value]
#         grouped_counts = selected_transformed_values.groupby('ObjectId')['UserId'].nunique().reset_index()
#         grouped_counts.columns = ['ObjectId', 'Unique_UserId_Count']
#         total_users = len(result_all_df['UserId'].unique())
#         grouped_counts['Percentage'] = (grouped_counts['Unique_UserId_Count'] / total_users) * 100
#         grouped_counts = pd.merge(grouped_counts, self.objects_df[['Id', 'Name']], left_on='ObjectId', right_on='Id')
#         grouped_counts.drop(columns='Id', inplace=True)
#         return grouped_counts
#
#     def analyze_results(self, selected_object_type, threshold_value):
#         output_string = ''
#
#         # Шаг 1
#         selected_objects_df = self.select_objects_by_type(selected_object_type)
#         output_string += f"<details><h1><details><summary>Шаг 1. Данные из таблицы Objects для выбранного типа ObjectType: {selected_object_type}</summary></h1>\n"
#         output_string += f"<p>{selected_objects_df.to_html()}</p></details>\n"
#
#         # Шаг 2
#         selected_transformed_values_all = self.get_transformed_values(selected_objects_df)
#         data_availability_message = self.check_data_availability(selected_transformed_values_all, selected_object_type)
#         if data_availability_message:
#             return None, data_availability_message, output_string
#
#         result_all_df = self.merge_and_prepare_tables(selected_transformed_values_all, selected_objects_df)
#         output_string += f'<details><summary>Шаг 2. Все данные из таблицы TransformedValue для выбранного типа ObjectType: {selected_object_type}</summary>'
#         output_string += f"<p>{result_all_df.head(100).to_html()}</p></details>\n"
#
#         # Шаг 3
#         percentage_of_total_users = self.calculate_percentage_of_total_users(result_all_df)
#         output_string += f"<details><h1><summary>Шаг 3. Процент пользователей, чьи параметры TransformValue превосходят порог  хотя бы для одного объекта выбранного типа ObjectType:</summary></h1>\n"
#         output_string += f"<p>Percentage of Users with TransformedValue > Threshold: {percentage_of_total_users:.2f}%</p></details>\n"
#
#         # Шаг 4
#         grouped_counts = self.process_transformed_values(result_all_df, threshold_value)
#         output_string += f"<details>h1><summary>Шаг 4. Распределение по процентам и число уникальных пользователей для каждого объекта выбранного типа ObjectType: {selected_object_type}, где показатель TransformedValue превысил порог:</summary></h1>\n"
#         output_string += f"<p>{grouped_counts.to_html()}</p></details>\n"
#
#         return grouped_counts, percentage_of_total_users, output_string
#
#
