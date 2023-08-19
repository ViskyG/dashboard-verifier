import pickle

from db import mongo_db_to_data_frame as mongo_db_to_dt, combined_file_reader as fr
from analysis import result_analyzer as ra
from output import html_out_writer as hw
from analysis import data_validator as dv
from analysis import data_composer as dc
from configurations.regional_configuration import RegionalConfiguration
from configurations.configuration import Configuration
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

    # --- Инициализация и загрузка данных ---

    # Установка региональных конфигураций
    regional_config = RegionalConfiguration()
    regional_config.set_region('Псковская область') #Псковская область, ЯНАО

    filereader = fr.CombinedFileReader()
    dataframes_from_files = filereader.load_files(regional_config)

    # Извлечение данных из файлов
    result = dataframes_from_files[regional_config.region_result_file_name]
    user_profiles = dataframes_from_files['_UserProfiles__202308041858.csv']
    enriched_user_profiles = dataframes_from_files[regional_config.enriched_user_profiles]
    school_class = dataframes_from_files['_SchoolClasses__202308010732.csv']
    all_sessions = dataframes_from_files['all_sessions.csv']
    objects = dataframes_from_files['_Objects__202307211518.csv']
    pupil_users = dataframes_from_files['_PupilUsers__202308010610.csv']
    user_answers = dataframes_from_files['_UserAnswers__202308160954.csv']
    aggregates = dataframes_from_files['_Aggregates__202308120653.csv']
    test_municipalities = dataframes_from_files['test_municipalities.csv']
    test_schools = dataframes_from_files['test_schools.csv']
    school_df = dataframes_from_files['Schools.pkl']
    regions_df = dataframes_from_files['Regions.pkl']
    screening_tests_df = dataframes_from_files['ScreeningTests.pkl']
    municipalities_df = dataframes_from_files['Municipalities.pkl']

    # --- Определение настроек ---

    # Основные настройки
    options = {
        'is_test': False,
        'is_choice_tests': True,
        'is_filter_tests': False,
        'is_filter_Objects': False,
        'is_choice_Objects': False,
        'top_n_rows': False,
        'is_technical_information': False,
        'is_last_results': True,
        'is_separate_results': True,
        'delete_is_delete': True,
        'delete_SchoolIsDeleted': True,
        'with_education': True,
        'translate': True,
    }

    # Тесты и объекты
    choice_tests = regional_config.choice_tests
    filter_tests = {}
    filter_objects = {}
    choice_objects = {"FieldDO"}  # SPO_VO FieldDO
    filter_top = 0
    alternate_sum = regional_config.alternate_sum

    format_options = {
        "technical_columns": ['PupilId', 'UserId', 'Birthday', 'FirstName', 'MiddleName', 'LastName', 'PhoneNumber', 'IsDeleted'],
        "remove_columns": ['City'],
        "remove_test_postfixes": [] #"_TransformedValue", "_MinValue", "_MaxValue"
    }

    # Создаем объект конфигурации
    config = Configuration(
        options=options,
        choice_tests=choice_tests,
        filter_tests=filter_tests,
        filter_Objects=filter_objects,
        choice_Objects=choice_objects,
        filter_top=filter_top,
        alternate_sum=alternate_sum,
        regional_config=regional_config,
        format_options=format_options
    )



    # --- Тестовые значения ---

    test_values = [


    ]

    # --- Соединение тестового объекта и результата ---
    Tester.set_test_values(test_values)

    # Очистка предыдущего выходного файла
    html_output = hw.HtmlOutWriter("../output.html")
    html_output.clean_html()

    # Инициализация валидатора
    validator = dv.DataValidator()

    # Удаление дубликатов
    df_without_duplicates = validator.remove_duplicates_by_id(result)
    print(df_without_duplicates.keys())
    Tester.check_values_in_dataframe(df_without_duplicates)

    # Добавление технической информации и дополнительных данных
    result_with_tech_info_df = dc.DataComposer.enrich_results_with_technical_info(
        df_without_duplicates, enriched_user_profiles)
    result_with_sessions_create_time_df = dc.DataComposer.enrich_results_with_sessions_create_time(
        result_with_tech_info_df, all_sessions)

    screening_tests_df = dc.DataComposer.compose_test_variants_dataframe(screening_tests_df)
    results_with_screening_test_df = dc.DataComposer.enrich_results_with_screening_test_info(
        result_with_sessions_create_time_df, screening_tests_df)
    results_with_additional_info_df = dc.DataComposer.enrich_results_with_additional_info(
        results_with_screening_test_df, pupil_users, school_class)

    # Обогащение результатов информацией о школах и муниципалитетах
    sc_enriched_result = dc.DataComposer.enrich_results_with_school_info(results_with_additional_info_df, pupil_users,
                                                                         school_class, school_df)
    enriched_result = dc.DataComposer.enrich_results_with_municipality_info(sc_enriched_result, user_profiles,
                                                                            municipalities_df)
    enrich_results_with_is_deleted = dc.DataComposer.enrich_results_with_is_deleted(enriched_result, user_profiles)

    #Обогащение результатов информацией об образовании
    if options['with_education']:
        results_with_with_education_df = dc.DataComposer.enrich_results_with_education(enrich_results_with_is_deleted, user_answers)
    else:
        results_with_with_education_df = enrich_results_with_is_deleted

    # Фильтрация результатов
    enriched_and_filtered_result = dc.DataComposer.filter_test_info(results_with_with_education_df,
                                                                    regional_config.folder_name, test_municipalities,
                                                                    test_schools)
    Tester.check_values_in_dataframe(enriched_and_filtered_result)
    result = enriched_and_filtered_result.query("UserHrid == 1400995")
    print(result)
    print(enriched_and_filtered_result['UserHrid'].dtype)
    enriched_and_filtered_result['UserHrid'].iloc[0].astype(str)
    # Создание Excel файла
    dc.DataComposer.create_excel_from_dataframe(enriched_and_filtered_result, config)

    # Составление данных по муниципалитетам и школам
    # composed_data = dc.DataComposer.compose_by_municipality_and_school(enriched_result)

    # # Clean the previous html output
    # html_output = hw.HtmlOutWriter("../output.html")
    # html_output.clean_html()
    #
    # # Convert the "Regions" collection in the "Catalog" database to a DataFrame
    # sessions_df = all_sessions
    #
    # # Get the unique object types from the objects dataframe
    # all_object_types = objects["ObjectType"].unique()
    #
    # # Define the threshold value for the analysis
    # threshold_value = 70.0
    #
    # validator = dv.DataValidator()
    #
    # df_without_duplicates = validator.remove_duplicates_by_id(result)
    #
    # print("tester check. df_without_duplicates")
    # Tester.check_values_in_dataframe(df_without_duplicates)
    #
    # result_with_tech_info_df = dc.DataComposer.enrich_results_with_technical_info(df_without_duplicates,
    #                                                                               user_profiles)
    #
    # result_with_sessions_create_time_df = \
    #     dc.DataComposer.enrich_results_with_sessions_create_time(result_with_tech_info_df, sessions_df)
    #
    # results_with_screening_test_df = \
    #     dc.DataComposer.enrich_results_with_screening_test_info(result_with_sessions_create_time_df, screening_tests_df)
    #
    # results_with_with_edishnal_info_df = (
    #     dc.DataComposer.enrich_results_with_additional_info(results_with_screening_test_df,
    #                                                         pupil_users,
    #                                                         school_class))
    #
    # # Analyze each object type and write the results to html
    # #for object_type in all_object_types:
    # #    ra.ResultAnalyzer.analyze_and_write_results(object_type, threshold_value,
    # #                                                dataframes['_Objects__202307211518.csv'],
    # #                                                results_with_with_education_df, html_output)
    #
    # school_class_df = school_class
    # pupil_user_df = pupil_users
    # objects_df = objects
    #
    # sc_enriched_result = dc.DataComposer.enrich_results_with_school_info(results_with_with_edishnal_info_df,
    #                                                                      pupil_user_df,
    #                                                                      school_class_df,
    #                                                                      school_df)
    #
    # profile_df = user_profiles
    #
    # enriched_result = dc.DataComposer.enrich_results_with_municipality_info(sc_enriched_result, profile_df,
    #                                                                         municipalities_df)
    #
    # enrich_results_with_is_deleted = dc.DataComposer.enrich_results_with_is_deleted(enriched_result, profile_df)
    #
    # # results_with_with_education_df = dc.DataComposer.enrich_results_with_education(enrich_results_with_is_deleted,
    # #                                                                                user_answers)
    #
    # enriched_and_filtered_result = dc.DataComposer.filter_test_info(enrich_results_with_is_deleted,
    #                                                                 reg_config.folder_name,
    #                                                                 test_municipalities,
    #                                                                 test_schools)
    #
    # print("tester check. enriched_and_filtered_result")
    # Tester.check_values_in_dataframe(df_without_duplicates)
    #
    # dc.DataComposer.create_excel_from_dataframe(enriched_and_filtered_result, reg_config)
    #
    # composed_data = dc.DataComposer.compose_by_municipality_and_school(enriched_result)
    # ra.ResultAnalyzer.analyze_composed_data(composed_data, threshold_value, objects_df, reg_config.folder_name)

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
    # Define the configuration file for MongoDB
    # config_file = "src/mongo_config.json"
    # # Initialize the MongoDBToDataFrame object with the config file
    # db_to_df = mongo_db_to_dt.MongoDBToDataFrame(config_file)
    #


    # List of CSV file names to be read
    files = [
        '_UserProfiles__202308041858.csv',
        '_Results__202308161724ЗЗ_PSKOB.csv',
        '_Objects__202307211518.csv',
        '_Aggregates__202308162114.csv',
        '_FactorMinMaxes__202308162115.csv',
        'sessions.csv'
    ]

    # Initialize the FileReader object
    filereader = fr.Filereader()

    # Read the CSV files into a dictionary of DataFrames
    dataframes = filereader.get_df_from_csv(files)

    sessions = dataframes['sessions.csv']
    user_profiles_df = dataframes['_UserProfiles__202308041858.csv']
    results_df = dataframes['_Results__202308161724ЗЗ_PSKOB.csv']
    objects_df = dataframes['_Objects__202307211518.csv']
    aggregates_df = dataframes['_Aggregates__202308162114.csv']
    factor_min_maxes_df = dataframes['_FactorMinMaxes__202308162115.csv']

    # Преобразование строковых дат в datetime
    sessions['CreatedDate'] = pd.to_datetime(sessions['CreatedDate'])
    results_df['CreatedDate'] = pd.to_datetime(results_df['CreatedDate'])


    # 1. Отфильтровать sessions по 2023 году
    sessions_2023 = sessions[sessions['CreatedDate'].dt.year == 2023]

    print('sessions_2023')
    # 2. Выбрать пользователей из user_profiles_df
    filtered_users = user_profiles_df[user_profiles_df['RegionId'] == '2beb4790-b106-4de9-bf31-7a87d2dbab5b']

    # 3. Отфильтровать results_df
    screening_test_ids = [
        'd9668ebf-ab47-49dc-bba4-ecdb944060a9',
        'b81b69aa-1f3a-4702-be7f-d1419a962040',
        '22249e14-9150-4ed7-bd77-82d4bd0bf1b5'
    ]

    print('results_filtered')
    results_filtered = results_df[
        (results_df['CreatedDate'].dt.year == 2023) &
        (results_df['ScreeningTestId'].isin(screening_test_ids))
        ]

    print(results_filtered.columns.tolist())

    print('merged_df')
    # 4. Соединения
    merged_df = (
        results_filtered
        .merge(sessions_2023[['_id', 'UserId']], left_on='SessionId', right_on='_id', how='inner')
        .merge(filtered_users['UserId'], on='UserId', how='inner')
        .merge(objects_df[['Id', 'Name', 'ObjectType']], left_on='ObjectId', right_on='Id', how='left', suffixes=('', '_obj'))
        .merge(factor_min_maxes_df[['ObjectId', 'ScreeningTestId', 'VariantId', 'MinValue', 'MaxValue']],
               on=['ObjectId', 'ScreeningTestId', 'VariantId'], how='left')
    )


    print(merged_df.columns.tolist())
    merged_df.drop(columns=['_id', 'Id_obj', 'NormalizedValue', 'CreatedDate'], inplace=True)
    # Убедитесь, что у вас есть все нужные столбцы и они идут в нужном порядке
    merged_df = merged_df[["Id", "UserId", "ObjectId", "TransformedValue", "Value",
                           "MinValue", "MaxValue", "SessionId", "ScreeningTestId",
                           "VariantId", "ObjectType", "Name"]]



    print("CSV")
    # Экспорт в CSV и вывод списка колонок
    merged_df.to_csv('filtered_results_Pskov_v1.csv', index=False)
    print(merged_df.columns.tolist())

    # # Преобразуем столбец CreatedDate в формат datetime
    # #sessions_df["CreatedDate"] = pd.to_datetime(sessions_df["CreatedDate"])
    #
    # #print(f"Number of rows in sessions_df: {sessions_df.shape[0]}")
    #
    # # Отбираем записи, где год CreatedDate равен 2023
    # #filtered_sessions_by_date = sessions_df[sessions_df["CreatedDate"].dt.year == 2023]
    #
    # print(f"Number of rows in sessions_df: {sessions_df.shape[0]}")
    #
    # # Оставляем только столбцы _id и UserId
    # filtered_sessions_by_date = sessions_df[["_id", "UserId"]]
    #
    # # Отбираем строки из user_profiles_df, где RegionId равен определенному значению (например, 1)
    # filtered_profiles = user_profiles_df[user_profiles_df["RegionId"] == "2beb4790-b106-4de9-bf31-7a87d2dbab5b"]
    #
    # # Выводим количество записей в final_filtered_sessions и filtered_profiles
    # print(f"Number of rows in final_filtered_sessions: {filtered_sessions_by_date.shape[0]}")
    # print(f"Number of rows in filtered_profiles: {filtered_profiles.shape[0]}")
    #
    # # Отбираем строки из filtered_sessions_by_date, где UserId соответствует UserId из filtered_profiles
    # final_filtered_sessions = filtered_sessions_by_date[
    #     filtered_sessions_by_date["UserId"].isin(filtered_profiles["UserId"])]
    #
    # # Записываем полученный датафрейм в файл
    # final_filtered_sessions.to_csv("final_filtered_sessions_pscov.csv", index=False)

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