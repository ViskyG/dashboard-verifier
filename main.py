from db import mongo_db_to_data_frame as mongo_db_to_dt, filereader as fr
from analysis import result_analyzer as ra
from output import html_out_writer as hw
import os
import pandas as pd


def analyze_and_write_results_per_municipality_and_school(object_type, threshold_value, objects, result, base_output_dir):
    municipalities = result['MunicipalityName'].dropna().unique()

    for municipality in municipalities:
        clear_municipality = municipality.replace('"', '').strip()
        municipality_dir = os.path.join(base_output_dir, clear_municipality)
        os.makedirs(municipality_dir, exist_ok=True)

        # Filter results for the current municipality
        municipality_results = result[result['MunicipalityName'] == municipality]
        print(municipality_results)
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
            print(school_results)
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


def enrich_results_with_school_info(result, pupil_user, school_class, school_df):
    # Выберем необходимые столбцы для каждой таблицы
    result_selected = result[['Id', 'UserId', 'IsLast', 'ObjectId', 'TransformedValue']]
    pupil_user_selected = pupil_user[['UserId', 'SchoolClassId']]
    school_class_selected = school_class[['Id', 'SchoolId']]
    school_df_selected = school_df[['_id', 'Number']]

    # Переименуем столбцы для избежания конфликтов имен
    school_class_selected = school_class_selected.rename(columns={'Id': 'SchoolClassId'})
    school_df_selected = school_df_selected.rename(columns={'_id': 'SchoolId', 'Number': 'SchoolName'})

    # Объединяем таблицы
    merged_df = pd.merge(result_selected, pupil_user_selected, how='left', on='UserId')
    merged_df = pd.merge(merged_df, school_class_selected, how='left', on='SchoolClassId')
    # Преобразуем SchoolId в строку для корректного соединения
    school_df_selected['SchoolId'] = school_df_selected['SchoolId'].astype(str)
    merged_df = pd.merge(merged_df, school_df_selected, how='left', on='SchoolId')

    return merged_df


def enrich_results_with_municipality_info(result, profiles_df, municipalities_df):
    # Выберем необходимые столбцы для каждой таблицы
    result_selected = result[['Id', 'UserId', 'IsLast', 'ObjectId', 'TransformedValue', 'SchoolId', 'SchoolName']]
    print(profiles_df)
    profiles_df_selected = profiles_df[['UserId', 'MunicipalityId']]
    municipalities_df_selected = municipalities_df[['_id', 'Name']]

    # Переименуем столбцы для избежания конфликтов имен
    profiles_df_selected = profiles_df_selected.rename(columns={'MunicipalityId': 'MunicipalityId'})
    municipalities_df_selected = municipalities_df_selected.rename(columns={'_id': 'MunicipalityId', 'Name': 'MunicipalityName'})

    # Преобразуем MunicipalityId в строку для корректного соединения
    municipalities_df_selected['MunicipalityId'] = municipalities_df_selected['MunicipalityId'].astype(str)

    # Объединяем таблицы
    merged_df = pd.merge(result_selected, profiles_df_selected, how='left', on='UserId')
    merged_df = pd.merge(merged_df, municipalities_df_selected, how='left', on='MunicipalityId')

    return merged_df


# The main function is defined to control the execution of the code
def main():
    # Define the configuration file for MongoDB
    config_file = "mongo_config.json"

    # Initialize the MongoDBToDataFrame object with the config file
    db_to_df = mongo_db_to_dt.MongoDBToDataFrame(config_file)

    # Convert the "Schools" collection in the "Catalog" database to a DataFrame
    school_df = db_to_df.convert_to_df("Catalog", "Schools", ["_id"])

    # Convert the "Regions" collection in the "Catalog" database to a DataFrame
    regions_df = db_to_df.convert_to_df("Catalog", "Regions", ["_id"])

    # Convert the "Municipalities" collection in the "Catalog" database to a DataFrame
    municipalities_df = db_to_df.convert_to_df("Catalog", "Municipalities", ["_id"])

    # List of CSV file names to be read
    files = [
        '_Result_first.csv',
        '_UserProfiles__202308041858.csv',
        '_SchoolClasses__202308010732.csv',
        '_SaasSessions__202307281155.csv',
        '_Objects__202307211518.csv',
        '_PupilUsers__202308010610.csv',
    ]

    # Initialize the FileReader object
    filereader = fr.Filereader()

    # Read the CSV files into a dictionary of DataFrames
    dataframes = filereader.get_df_from_csv(files)

    # Get the unique object types from the objects dataframe
    all_object_types = dataframes['_Objects__202307211518.csv']["ObjectType"].unique()

    # Clean the previous html output
    html_output = hw.HtmlOutWriter("output.html")
    html_output.clean_html()

    # Define the threshold value for the analysis
    threshold_value = 70.0

    # Analyze each object type and write the results to html
    for object_type in all_object_types:
        analyze_and_write_results(object_type, threshold_value, dataframes['_Objects__202307211518.csv'], dataframes['_Result_first.csv'] , html_output)

    result_df = dataframes['_Result_first.csv']
    school_class_df = dataframes['_SchoolClasses__202308010732.csv']
    pupil_user_df = dataframes['_PupilUsers__202308010610.csv']
    objects_df = dataframes['_Objects__202307211518.csv']

    sc_enriched_result = enrich_results_with_school_info(result_df, pupil_user_df, school_class_df, school_df)

    profile_df = dataframes['_UserProfiles__202308041858.csv']
    municipalities_df

    enriched_result = enrich_results_with_municipality_info(sc_enriched_result, profile_df, municipalities_df)

    analyze_and_write_results_per_municipality_and_school(object_type, threshold_value, objects_df, enriched_result, 'Output')


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


# Check if this file is being run directly
if __name__ == '__main__':
    main()