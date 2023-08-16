import pickle

from db import mongo_db_to_data_frame as mongo_db_to_dt, filereader as fr
from analysis import result_analyzer as ra
from output import html_out_writer as hw
from analysis import data_validator as dv
from analysis import data_composer as dc
import os
import pandas as pd
import fastparquet

from src.Test.tester import Tester


def analyze_and_write_results_per_municipality_and_school(object_type, threshold_value, objects, result, base_output_dir):
    municipalities = result['MunicipalityName'].dropna().unique()

    for municipality in municipalities:
        clear_municipality = municipality.replace('"', '').strip()
        municipality_dir = os.path.join(base_output_dir, clear_municipality)
        os.makedirs(municipality_dir, exist_ok=True)

        # Filter results for the current municipality
        municipality_results = result[result['MunicipalityName'] == municipality]

        # Analyze and write results for the municipality
        output_filename = os.path.join(municipality_dir, f'{clear_municipality}_output.html')
        write_results(output_filename, objects, municipality_results, threshold_value)

        schools = municipality_results['SchoolName'].dropna().unique()

        for school in schools:
            clear_school = school.replace('"', '').replace('/', '').strip()
            school_dir = os.path.join(municipality_dir, clear_school)
            os.makedirs(school_dir, exist_ok=True)

            # Filter results for the current school
            school_results = municipality_results[municipality_results['SchoolName'] == school]

            output_filename = os.path.join(school_dir, f'{clear_school}_output.html')

            # Analyze and write results for the school
            write_results(output_filename, objects, school_results, threshold_value)


def write_results(output_filename, objects, results, threshold_value):
    object_types = objects['ObjectType'].unique()
    html_output = hw.HtmlOutWriter(output_filename)
    html_output.clean_html()

    for object_type in object_types:
        print(f"\nAnalyzing object type: {object_type}")
        analyzer = ra.ResultAnalyzer()

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








# The main function is defined to control the execution of the code
def main():
    # # Define the configuration file for MongoDB
    # config_file = "src/mongo_config.json"
    # # Initialize the MongoDBToDataFrame object with the config file
    # db_to_df = mongo_db_to_dt.MongoDBToDataFrame(config_file)
    #
    # # Convert the "Schools" collection in the "Catalog" database to a DataFrame
    # school_df = db_to_df.convert_to_df("Catalog", "Schools", ["_id"])
    #
    # # Convert the "Regions" collection in the "Catalog" database to a DataFrame
    # regions_df = db_to_df.convert_to_df("Catalog", "Regions", ["_id"])
    #
    # # Convert the "Regions" collection in the "Catalog" database to a DataFrame
    # screening_tests_df = db_to_df.convert_to_df("Player", "ScreeningTests", ["_id"])
    #
    #
    #
    # # Convert the "Municipalities" collection in the "Catalog" database to a DataFrame
    # municipalities_df = db_to_df.convert_to_df("Catalog", "Municipalities", ["_id"])
    output_dir = 'files'  # замените на ваш путь к директории


    with open(os.path.join(output_dir, 'Schools.pkl'), 'rb') as file:
        school_df = pickle.load(file)

    with open(os.path.join(output_dir, 'Regions.pkl'), 'rb') as file:
        regions_df = pickle.load(file)

    with open(os.path.join(output_dir, 'ScreeningTests.pkl'), 'rb') as file:
        screening_tests_df = pickle.load(file)

    with open(os.path.join(output_dir, 'Municipalities.pkl'), 'rb') as file:
        municipalities_df = pickle.load(file)


    screening_tests_df = dc.DataComposer.compose_test_variants_dataframe(screening_tests_df)

    # List of CSV file names to be read
    files = [
        '_pskovskaya_result_2.csv', #_yanao_result_1
        '_UserProfiles__202308041858.csv',
        '_SchoolClasses__202308010732.csv',
        'all_sessions.csv',
        '_Objects__202307211518.csv',
        '_PupilUsers__202308010610.csv',
        '_UserAnswers__202308160954.csv',
        '_Aggregates__202308120653.csv',
        '_UserProfiles_Pskov.csv',
        'test_municipalities.csv',
        'test_schools.csv'
    ]

    test_values = [
        {'UserId': '699e018f-2cb4-4116-98ef-280a5b371c00', 'Name': "Спорт"},
        {'UserId': 'c657d17e-d23a-4f46-8199-ed966bb8f310', 'Name': "Красота и мода"},
        {'UserId': '15efcfe0-5842-4ffd-8a9a-2fd098fb3250', 'Name': "Спорт"}
    ]

    Tester.set_test_values(test_values)


    # Initialize the FileReader object
    filereader = fr.Filereader()

    # Read the CSV files into a dictionary of DataFrames
    dataframes = filereader.get_df_from_csv(files)

    # Convert the "Regions" collection in the "Catalog" database to a DataFrame
    sessions_df = dataframes['all_sessions.csv']

    # Get the unique object types from the objects dataframe
    all_object_types = dataframes['_Objects__202307211518.csv']["ObjectType"].unique()

    # Clean the previous html output
    html_output = hw.HtmlOutWriter("../output.html")
    html_output.clean_html()

    # Define the threshold value for the analysis
    threshold_value = 70.0

    validator = dv.DataValidator()
    df_without_duplicates = validator.remove_duplicates_by_id(dataframes['_pskovskaya_result_2.csv'])

    print("tester check. df_without_duplicates")
    Tester.check_values_in_dataframe(df_without_duplicates)

    result_with_tech_info_df = dc.DataComposer.enrich_results_with_technical_info(df_without_duplicates,
                                                                                  dataframes['_UserProfiles_Pskov.csv'])

    result_with_sessions_create_time_df = \
        dc.DataComposer.enrich_results_with_sessions_create_time(result_with_tech_info_df, sessions_df)

    results_with_screening_test_df = \
        dc.DataComposer.enrich_results_with_screening_test_info(result_with_sessions_create_time_df, screening_tests_df)


    results_with_with_edishnal_info_df = dc.DataComposer.enrich_results_with_additional_info(results_with_screening_test_df,
                                                                                             dataframes['_PupilUsers__202308010610.csv'],
                                                                                             dataframes['_SchoolClasses__202308010732.csv'])

    # Analyze each object type and write the results to html
    #for object_type in all_object_types:
    #    ra.ResultAnalyzer.analyze_and_write_results(object_type, threshold_value,
    #                                                dataframes['_Objects__202307211518.csv'],
    #                                                results_with_with_education_df, html_output)

    school_class_df = dataframes['_SchoolClasses__202308010732.csv']
    pupil_user_df = dataframes['_PupilUsers__202308010610.csv']
    objects_df = dataframes['_Objects__202307211518.csv']

    sc_enriched_result = dc.DataComposer.enrich_results_with_school_info(results_with_with_edishnal_info_df, pupil_user_df,
                                                                        school_class_df, school_df)

    profile_df = dataframes['_UserProfiles__202308041858.csv']

    enriched_result = dc.DataComposer.enrich_results_with_municipality_info(sc_enriched_result, profile_df,
                                                                            municipalities_df)

    enrich_results_with_is_deleted = dc.DataComposer.enrich_results_with_is_deleted(enriched_result, profile_df)

    results_with_with_education_df = dc.DataComposer.enrich_results_with_education(enrich_results_with_is_deleted,
                                                                                   dataframes[
                                                                                       '_UserAnswers__202308160954.csv'])
    folder_name = 'Pskov_3'
    enriched_and_filtered_result = dc.DataComposer.filter_test_info(results_with_with_education_df, folder_name,
                                                                    dataframes['test_municipalities.csv'],
                                                                    dataframes['test_schools.csv'])

    print("tester check. enriched_and_filtered_result")
    Tester.check_values_in_dataframe(df_without_duplicates)


    #tests_to_sum = [("Тест на способности для ЯНАО_Стандартный вариант для всех",
    #                 "Тест на интересы для ЯНАО_Стандартный вариант для всех")]
    tests_to_sum = [("Тест на способности для Псковской области_Стандартный вариант для всех",
                     "Тест на интересы для Псковской области_Стандартный вариант для всех")]

    dc.DataComposer.create_excel_from_dataframe(enriched_and_filtered_result, folder_name, tests_to_sum)

    composed_data = dc.DataComposer.compose_by_municipality_and_school(enriched_result)
    ra.ResultAnalyzer.analyze_composed_data(composed_data, threshold_value, objects_df, folder_name)

# Function to analyze a specific object type and write the results to html
def analyze_and_write_results(object_type, threshold_value, objects, result, html_output):
    print(f"\nAnalyzing object type: {object_type}")
    analyzer = ra.ResultAnalyzer()

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

def test():
    # List of CSV file names to be read
    files = [
        '_UserProfiles__202308041858.csv',
        'all_sessions.csv'
    ]

    # Initialize the FileReader object
    filereader = fr.Filereader()

    # Read the CSV files into a dictionary of DataFrames
    dataframes = filereader.get_df_from_csv(files)
    user_profiles_df = dataframes['_UserProfiles__202308041858.csv']

    sessions_df = dataframes['all_sessions.csv']

    # Преобразуем столбец CreatedDate в формат datetime
    #sessions_df["CreatedDate"] = pd.to_datetime(sessions_df["CreatedDate"])

    #print(f"Number of rows in sessions_df: {sessions_df.shape[0]}")

    # Отбираем записи, где год CreatedDate равен 2023
    #filtered_sessions_by_date = sessions_df[sessions_df["CreatedDate"].dt.year == 2023]

    print(f"Number of rows in sessions_df: {sessions_df.shape[0]}")

    # Оставляем только столбцы _id и UserId
    filtered_sessions_by_date = sessions_df[["_id", "UserId"]]

    # Отбираем строки из user_profiles_df, где RegionId равен определенному значению (например, 1)
    filtered_profiles = user_profiles_df[user_profiles_df["RegionId"] == "2beb4790-b106-4de9-bf31-7a87d2dbab5b"]

    # Выводим количество записей в final_filtered_sessions и filtered_profiles
    print(f"Number of rows in final_filtered_sessions: {filtered_sessions_by_date.shape[0]}")
    print(f"Number of rows in filtered_profiles: {filtered_profiles.shape[0]}")

    # Отбираем строки из filtered_sessions_by_date, где UserId соответствует UserId из filtered_profiles
    final_filtered_sessions = filtered_sessions_by_date[
        filtered_sessions_by_date["UserId"].isin(filtered_profiles["UserId"])]

    # Записываем полученный датафрейм в файл
    final_filtered_sessions.to_csv("final_filtered_sessions_pscov.csv", index=False)

def rename():


    selected_ids_df = pd.read_csv("../final_filtered_sessions_pscov.csv")

    # Формирование списка кортежей
    id_tuples = list(zip(selected_ids_df["_id"], selected_ids_df["UserId"]))

    # Запись кортежей в файл
    with open("../selected_ids_pscov.txt", "w") as file:
        for id_tuple in id_tuples:
            file.write(f"('{id_tuple[0]}', '{id_tuple[1]}'),\n")



def uniq_string_in_txt():
    # Открываем файл и читаем его содержимое
    with open('../UniqHrid.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Удаляем возможные пробельные символы (включая перенос строки) с начала и конца каждой строки
    lines = [line.strip() for line in lines]

    # Получаем число уникальных строк
    unique_lines_count = len(set(lines))

    print(f"Число уникальных строк: {unique_lines_count}")


if __name__ == '__main__':
    main()
    #rename()