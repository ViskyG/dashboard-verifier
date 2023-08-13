import pandas as pd
from output import html_out_writer as hw
import os
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
    @staticmethod
    def generate_filename(test_name, variant_name):
        test_part = test_name[:35]
        variant_part = variant_name[:35]
        return f"{test_part}_{variant_part}.html"
    @staticmethod
    def analyze_composed_data(composed_data, threshold_value, objects, base_output_dir):
        # Ensure the directory exists
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        # Analyze total results
        output_filename = os.path.join(base_output_dir, 'total_output.html')
        ResultAnalyzer.write_results(output_filename, objects, composed_data['total']['total_results'], threshold_value)
        # Analyze and write results for each test and variant for the total data
        for test, test_data in composed_data['total']['tests'].items():
            for variant, variant_results in test_data['variants'].items():
                filename = ResultAnalyzer.generate_filename(test, variant)
                output_filename = os.path.join(base_output_dir, filename)
                ResultAnalyzer.write_results(output_filename, objects, variant_results, threshold_value)

        # Analyze municipality level
        for municipality, data in composed_data.items():
            if municipality == 'total':  # Skip total results, as they have been processed above
                continue
            municipality_dir = os.path.join(base_output_dir, municipality)
            os.makedirs(municipality_dir, exist_ok=True)

            # Analyze and write results for the municipality
            output_filename = os.path.join(municipality_dir, f'{municipality}_output.html')
            ResultAnalyzer.write_results(output_filename, objects, data['municipality_results'], threshold_value)

            # Analyze and write results for each test and variant at the municipality level
            for test, test_data in data['tests'].items():
                for variant, variant_results in test_data['variants'].items():
                    filename = ResultAnalyzer.generate_filename(test, variant)
                    output_filename = os.path.join(municipality_dir, filename)
                    ResultAnalyzer.write_results(output_filename, objects, variant_results, threshold_value)

            # Analyze schools
            for school, school_data in data['schools'].items():
                school_dir = os.path.join(municipality_dir, school)
                os.makedirs(school_dir, exist_ok=True)

                # Analyze and write results for the school
                output_filename = os.path.join(school_dir, f'{school}_output.html')
                ResultAnalyzer.write_results(output_filename, objects, school_data['school_results'], threshold_value)

                # Analyze and write results for each test and variant at the school level
                for test, test_data in school_data['tests'].items():
                    for variant, variant_results in test_data['variants'].items():
                        filename = ResultAnalyzer.generate_filename(test, variant)
                        output_filename = os.path.join(school_dir, filename)
                        ResultAnalyzer.write_results(output_filename, objects, variant_results, threshold_value)

    @staticmethod
    def write_results(output_filename, objects, results, threshold_value):
        object_types = objects['ObjectType'].unique()
        html_output = hw.HtmlOutWriter(output_filename)
        html_output.clean_html()

        for object_type in object_types:
            print(f"\nAnalyzing object type: {object_type}")
            analyzer = ResultAnalyzer()

            grouped_counts, percentage_of_total_users, output_string = \
                analyzer.analyze_results(objects, results, object_type, threshold_value)

            if grouped_counts is not None:
                html_output.write(f"\n\nPercentage distribution and number of defined users for each object"
                                  f" of the selected type '{object_type}':")
                html_output.write_df_to_html(grouped_counts, "Percentage distribution and number of defined users")
                print('percentage_of_total_users:')
                print(percentage_of_total_users)
                html_output.write("Percentage of total users: " + str(percentage_of_total_users))
                html_output.write(output_string)

    @staticmethod
    def analyze_and_write_results(object_type, threshold_value, objects, result, html_output):
        print(f"\nAnalyzing object type: {object_type}")
        analyzer = ResultAnalyzer()

        grouped_counts, percentage_of_total_users, output_string = \
            analyzer.analyze_results(objects, result, object_type, threshold_value)

        if grouped_counts is not None:
            html_output.write(f"Percentage distribution and number of defined users for each object"
                              f" of the selected type '{object_type}':")
            html_output.write_df_to_html(grouped_counts, "Percentage distribution and number of defined users")
            print('percentage_of_total_users:')
            print(percentage_of_total_users)
            html_output.write("Percentage of total users: " + str(percentage_of_total_users))
            html_output.write(output_string)


