import numpy as np
import pandas as pd
import uuid
from typing import List
import os
import openpyxl
import xlsxwriter
from tqdm import tqdm

class DataComposer:

    def __init__(self):
        pass

    @staticmethod
    def filter_latest_results(df: pd.DataFrame) -> pd.DataFrame:
        """
        Фильтрует DataFrame, оставляя только последние результаты для каждой комбинации UserId, Name_Object, ScreeningTestName и VariantName.

        Параметры:
        - df (pd.DataFrame): Исходный DataFrame с результатами.

        Возвращает:
        - pd.DataFrame: Отфильтрованный DataFrame с последними результатами.
        """

        # Создаем tqdm объект для отслеживания прогресса
        tqdm.pandas(desc="Filtering latest results")

        print(df.columns)

        df['SessionCreatedDate'] = pd.to_datetime(df['SessionCreatedDate'])

        # Группируем данные с помощью tqdm и находим индекс строки с самой поздней датой SessionCreatedDate для каждой комбинации
        idx = df.groupby(['UserId', 'Name', 'ScreeningTestName', 'VariantName'])[
            'SessionCreatedDate'].progress_apply(lambda x: x.idxmax())

        # Используем индексы, чтобы отфильтровать исходный DataFrame
        latest_results = df.loc[idx.values]

        return latest_results

    @staticmethod
    def exclude_tests(df: pd.DataFrame, test_names: list = None) -> pd.DataFrame:
        """
        Удаляет из DataFrame тесты, указанные в списке.

        Параметры:
        - df (pd.DataFrame): Исходный DataFrame для обработки.
        - test_names (list, optional): Список имен тестов для исключения.

        Возвращает:
        - pd.DataFrame: DataFrame, из которого удалены указанные тесты.
        """

        if test_names is None:
            test_names = ["Тест на способности для Псковской области", "Тест на интересы для Псковской области"]

        # Исключаем строки, которые содержат имена тестов из списка test_names
        df_excluded = df[~df['ScreeningTestName'].isin(test_names)]

        return df_excluded

    @staticmethod
    def choice_and_sort(df: pd.DataFrame, test_names: list = None) -> pd.DataFrame:
        """
        Фильтрует и сортирует DataFrame на основе заданных условий.

        Параметры:
        - df (pd.DataFrame): Исходный DataFrame для обработки.
        - test_names (list, optional): Список имен тестов для сортировки.

        Возвращает:
        - pd.DataFrame: Отфильтрованный и отсортированный DataFrame.
        """

        #if True:
        #    test_names = ["Тест на способности для Псковской области", "Тест на интересы для Псковской области"]

        if test_names:
            df_filtered = df[df['ScreeningTestName'].isin(test_names)]

        # Сортируем DataFrame по UserId, ScreeningTestId, VariantId и TransformedValue
        df_sorted = df_filtered.sort_values(by=['UserId', 'ScreeningTestId', 'VariantId', 'TransformedValue'],
                                            ascending=[True, True, True, False])

        return df_sorted


    @staticmethod
    def generate_test_data(df):
        # Шаг 1: Выбор записей по условию
        #mask = df['ScreeningTestName'].isin(["Тест на способности для Псковской области",
        #                                     "Тест на интересы для Псковской области"])
        initial_selection = df.head(60000)

        # Шаг 2: Найти все записи с теми же значениями UserHrid и Name
        unique_users_and_names = initial_selection[['UserHrid', 'Name']].drop_duplicates()

        extended_selection = pd.merge(unique_users_and_names, df, on=['UserHrid', 'Name'], how='left')

        # Шаг 3: Объединение двух выборок
        test_data = pd.concat([initial_selection, extended_selection]).drop_duplicates()

        return test_data

    @staticmethod
    def preprocess_data(df: pd.DataFrame, tests_to_sum=None, isTest=False, technical_information=False) -> pd.DataFrame:
        options = {
        'is_test' : True,
        'is_choice_tests': True,
        'is_filter_tests': False,
        'is_filter_Objects' : False,
        'is_choice_Objects' : False,
        'top_n_rows' : False,
        'is_technical_information' : False,
        'is_last_results' : True,
        'is_separate_relults' : False,
        'delete is_delete' : True,
        'delete SchoolIsDeleted' : True
        }

        pskovskaya_oblast = {"Тест на способности для Псковской области",
                             "Тест на интересы для Псковской области",
                             "Тест для Псковской области"}

        yanao = {"Тест на способности для ЯНАО",
                 "Тест на интересы для ЯНАО",
                 "Тест для ЯНАО"}

        choice_tests = pskovskaya_oblast


        filter_tests = {}
        filter_Objects = {}
        choice_Objects = {"FieldDO"}
        filter_top = 0




        """
        Предобработка данных: фильтрация, сортировка, исключение тестов, генерация тестовых данных.

        Параметры:
        - df (pd.DataFrame): Исходный DataFrame.
        - tests_to_sum (list, optional): Список имен тестов для суммирования.
        - isTest (bool): Флаг генерации тестовых данных.
        - technical_information (bool): Флаг использования технической информации.

        Возвращает:
        - pd.DataFrame: Обработанный DataFrame.
        """

        if options['delete is_delete']:
            print("delete is_delete = True")
            print('before ', df.shape[0])
            df = df[df['IsDeleted'] == False]
            print('after ', df.shape[0])

        if options['delete SchoolIsDeleted']:
            print("delete SchoolIsDeleted = True")
            print('before ', df.shape[0])
            df = df[df['SchoolIsDeleted'] == False]
            print('after ', df.shape[0])

        # Генерация тестовых данных, если требуется
        if options['is_test']:
            print("is_test = True")
            print('before ', df.shape[0])
            df = DataComposer.generate_test_data(df)
            print('after ', df.shape[0])

        if options['is_filter_Objects']:
            print("is_filter_Objects = True")
            print('before ', df.shape[0])
            df = DataComposer.filter_objects(df, filter_Objects)
            print('after ', df.shape[0])

        if options['is_choice_Objects']:
            print("is_choice_Objects = True")
            print('before ', df.shape[0])
            df = DataComposer.choice_objects(df, choice_Objects)
            print('after ', df.shape[0])

        if options['is_choice_tests']:
            print("is_choice_tests = True")
            print('before ', df.shape[0])
            df = DataComposer.choice_and_sort(df, test_names=choice_tests)
            print('after ', df.shape[0])

        if options['is_filter_tests']:
            print("is_filter_tests = True")
            print('before ', df.shape[0])
            df = DataComposer.exclude_tests(df)
            print('after ', df.shape[0])

        if options['is_last_results']:
            print("is_last_results = True")
            print('before ', df.shape[0])
            df = DataComposer.filter_latest_results(df)
            print('after ', df.shape[0])


        return df, options

    @staticmethod
    def save_unique_counts_to_csv(unique_counts: pd.Series, folder_path: str) -> None:
        """
        Сохраняет уникальные счетчики пользователей в CSV-файл.

        Параметры:
        - unique_counts (pd.Series): Количество уникальных пользователей для каждой комбинации.
        - folder_path (str): Путь к папке, в которой будет сохранен файл.
        """

        # Создаем легенду: присваиваем каждому уникальному названию теста номер
        legend = {name: i for i, name in enumerate(unique_counts.index, 1)}



        # Создаем датафрейм для сохранения: номер теста и соответствующее количество уникальных пользователей
        save_df = pd.DataFrame({
            'TestNumber': [legend[name] for name in unique_counts.index],
            'UniqueUsersCount': unique_counts.values.ravel()
        })

        # Сохраняем легенду и датафрейм в CSV
        legend_df = pd.DataFrame(list(legend.items()), columns=['TestNumber', 'TestVariant'])
        legend_df.to_csv(f"{folder_path}/legend.csv", index=False)
        save_df.to_csv(f"{folder_path}/unique_user_counts.csv", index=False)

    # Пример использования:
    # unique_counts, _ = process_dataframe(df)
    # save_unique_counts_to_csv(unique_counts, 'path_to_your_folder')

    import pandas as pd

    def process_dataframe(df: pd.DataFrame) -> (pd.Series, dict):
        # Создаем уникальный идентификатор для комбинации теста и варианта
        df['TestVariant'] = df['ScreeningTestName'] + "_" + df['VariantName']

        # Создаем список комбинаций для каждого пользователя, убираем дубликаты
        user_combinations = df.groupby('UserId')['TestVariant'].unique().apply(lambda x: ','.join(sorted(set(x))))

        # Считаем количество пользователей для каждой уникальной комбинации
        combination_counts = user_combinations.reset_index().groupby('TestVariant').count()


        # Сохраняем подмножество данных для каждой комбинации
        grouped_dfs = {}
        for combo, user_df in user_combinations.reset_index().groupby('TestVariant'):
            merged_df = pd.merge(df, user_df, on='UserId', how='inner')

            grouped_dfs[combo] = merged_df.drop(columns=['TestVariant_x', 'TestVariant_y'])

        return combination_counts, grouped_dfs


    @staticmethod
    def create_excel_from_dataframe(df, output_folder, tests_to_sum=None):
        print("Количество групп")
        print(df.shape[0])

        #equivalent = "Тест для ЯНАО_Стандартный вариант для всех"
        alternate_sum = "Тест для Псковской области_Стандартный вариант для всех"

        df, options = DataComposer.preprocess_data(df, tests_to_sum=tests_to_sum, isTest=False)
        print(options)
        print("Количество групп")
        print(df.shape[0])
        if options['is_separate_relults']:

            unique_user_counts, grouped_dfs = DataComposer.process_dataframe(df)
            DataComposer.save_unique_counts_to_csv(unique_user_counts, output_folder)
            legend = {name: i for i, name in enumerate(grouped_dfs.keys(), 1)}

            for test_variant, group_df in grouped_dfs.items():
                # Используем номер комбинации из легенды для имени файла
                file_name = f"combo_{legend[test_variant]}"
                DataComposer.create_excel_from_prepared_dataframe(group_df, output_folder, file_name, tests_to_sum,
                                                                  alternate_sum=alternate_sum, options=options)

        else:
            DataComposer.create_excel_from_prepared_dataframe(df, output_folder, 'result_out', tests_to_sum,
                                                              alternate_sum=alternate_sum,
                                                              options=options)
    @staticmethod
    def create_excel_from_prepared_dataframe(df, output_folder, file_name, tests_to_sum=None, alternate_sum=None, equivalent=None, options=None):

        print("Количество групп")
        print(df.shape[0])

        if tests_to_sum is None:
            tests_to_sum = []

        unique_tests = df[['ScreeningTestName', 'VariantName']].drop_duplicates()

        columns = ['UserHrid', 'Name', 'ClassName', 'СПО/ВО', 'ObjectType', 'SchoolName', 'MunicipalityName']
        for _, row in unique_tests.iterrows():
            test_name = f"{row['ScreeningTestName']}_{row['VariantName']}"
            columns.extend([test_name + '_TransformedValue', test_name + '_Value', test_name + '_MinValue',
                            test_name + '_MaxValue', test_name + '_Session_Date'])

        result_df = pd.DataFrame(columns=columns)

        def aggregate_tests(group):
            # if technical information is True then we need to add technical information to result_df
            if options['is_technical_information']:
                data = {
                #'UserId', 'City', 'CreatedDate', 'Birthday', 'FirstName', 'MiddleName', 'LastName', 'PhoneNumber'
                    'UserHrid': group['UserHrid'].iloc[0],
                    'PupilId': group['PupilId'].iloc[0],
                    'UserId': group['UserId'].iloc[0],
                    'Name': group['Name'].iloc[0],
                    'ClassName': group['ClassName'].iloc[0],
                    'СПО_ВО': group['СПО_ВО'].iloc[0],
                    'ObjectType': group['ObjectType'].iloc[0],
                    'SchoolName': group['SchoolName'].iloc[0],
                    'MunicipalityName': group['MunicipalityName'].iloc[0],
                    'City': group['City'].iloc[0],
                    'Birthday': group['Birthday'].iloc[0],
                    'CreatedDate': group['CreatedDate'].iloc[0],
                    'FirstName': group['FirstName'].iloc[0],
                    'MiddleName': group['MiddleName'].iloc[0],
                    'LastName': group['LastName'].iloc[0],
                    'PhoneNumber': group['PhoneNumber'].iloc[0],
                    'IsDeleted': group['IsDeleted'].iloc[0]
                }
            else:
                data = {
                    'UserHrid': group['UserHrid'].iloc[0],
                    'PupilId': group['PupilId'].iloc[0],
                    'Name_Object': group['Name'].iloc[0],
                    'ClassName': group['ClassName'].iloc[0],
                    'ObjectType': group['ObjectType'].iloc[0],
                    'SchoolName': group['SchoolName'].iloc[0],
                    'MunicipalityName': group['MunicipalityName'].iloc[0]
                }

            test_names = group['ScreeningTestName'] + '_' + group['VariantName']
            data.update({name + '_TransformedValue': val for name, val in zip(test_names, group['TransformedValue'])})
            data.update({name + '_Value': val for name, val in zip(test_names, group['Value'])})
            data.update({name + '_MinValue': val for name, val in zip(test_names, group['MinValue'])})
            data.update({name + '_MaxValue': val for name, val in zip(test_names, group['MaxValue'])})
            data.update(
                {name + '_Session_Date': date for name, date in zip(test_names, group['SessionCreatedDate'])})

            return pd.Series(data)

        # Группировка по UserHrid и Name_Object
        grouped = df.groupby(['UserHrid', 'Name'])

        counter = 0
        result_data = []
        print("Количество групп")
        print(df.shape[0])
        # Применение функции к каждой группе
        for _, group in grouped:
            result_data.append(aggregate_tests(group))
            counter += 1
            if counter % 10000 == 0:
                print(f"Обработано {counter} групп")

        # Добавление результатов в result_df
        result_df = pd.concat([result_df, pd.DataFrame(result_data)], ignore_index=True)

        # Добавление столбцов для суммированных тестов (если это требуется)
        import numpy as np

        def calculate_values_with_equivalent(result_df, test_pair, attr_dict, equivalent_test_name=None):
            nan_mask_1 = result_df[test_pair[0] + '_Value'].isna()
            nan_mask_2 = result_df[test_pair[1] + '_Value'].isna()
            both_nan_mask = nan_mask_1 & nan_mask_2

            sum_values = result_df[[test + '_Value' for test in test_pair]].sum(axis=1, skipna=True)
            sum_min_values = result_df[[test + '_MinValue' for test in test_pair]].sum(axis=1, skipna=True)
            sum_max_values = result_df[[test + '_MaxValue' for test in test_pair]].sum(axis=1, skipna=True)

            sum_transformed_values = pd.Series(index=result_df.index)

            if equivalent_test_name and equivalent_test_name + '_Value' in result_df.columns:
                eq_date = result_df[equivalent_test_name + '_Session_Date']
                pair_dates = result_df[[test + '_Session_Date' for test in test_pair]]
                latest_test_date = pair_dates.max(axis=1)
                use_equivalent_mask = (eq_date > latest_test_date) & ~eq_date.isna()

                # Добавляем условие, чтобы использовать эквивалентную диагностику, если оба теста равны NaN
                use_equivalent_mask |= both_nan_mask

                # Если эквивалентный тест используется:
                sum_values[use_equivalent_mask] = result_df[equivalent_test_name + '_Value'][use_equivalent_mask]
                sum_min_values[use_equivalent_mask] = result_df[equivalent_test_name + '_MinValue'][use_equivalent_mask]
                sum_max_values[use_equivalent_mask] = result_df[equivalent_test_name + '_MaxValue'][use_equivalent_mask]
                sum_transformed_values[use_equivalent_mask] = result_df[equivalent_test_name + '_TransformedValue'][
                    use_equivalent_mask]

            both_tests_mask = ~nan_mask_1 & ~nan_mask_2
            sum_transformed_values[both_tests_mask] = ((sum_values[both_tests_mask] - sum_min_values[
                both_tests_mask]) * 100) / \
                                                      (sum_max_values[both_tests_mask] - sum_min_values[
                                                          both_tests_mask])

            single_test_mask = nan_mask_1 ^ nan_mask_2
            for i, mask in enumerate([nan_mask_1, nan_mask_2]):
                current_mask = single_test_mask & ~mask
                sum_values[current_mask] = result_df[test_pair[i] + '_Value'][current_mask].astype(str) + "!"
                sum_min_values[current_mask] = result_df[test_pair[i] + '_MinValue'][current_mask].astype(str) + "!"
                sum_max_values[current_mask] = result_df[test_pair[i] + '_MaxValue'][current_mask].astype(str) + "!"

            sum_transformed_values_single = ((sum_values[single_test_mask].str.replace('!', '', regex=False).astype(
                float) -
                                              sum_min_values[single_test_mask].str.replace('!', '', regex=False).astype(
                                                  float)) * 100) / \
                                            (sum_max_values[single_test_mask].str.replace('!', '', regex=False).astype(
                                                float) -
                                             sum_min_values[single_test_mask].str.replace('!', '', regex=False).astype(
                                                 float))
            sum_transformed_values[single_test_mask] = sum_transformed_values_single.astype(str) + "!"

            return sum_values, sum_min_values, sum_max_values, sum_transformed_values
        def calculate_values_with_alternate(result_df, test_pair, attr_dict, alternate_sum=None):
            nan_mask_1 = result_df[test_pair[0] + '_Value'].isna()
            nan_mask_2 = result_df[test_pair[1] + '_Value'].isna()
            both_nan_mask = nan_mask_1 & nan_mask_2

            sum_values = result_df[[test + '_Value' for test in test_pair]].sum(axis=1, skipna=True)
            sum_min_values = result_df[[test + '_MinValue' for test in test_pair]].sum(axis=1, skipna=True)
            sum_max_values = result_df[[test + '_MaxValue' for test in test_pair]].sum(axis=1, skipna=True)

            sum_transformed_values = pd.Series(index=result_df.index)

            if alternate_sum and all([(alternate_sum + key) in result_df.columns for key in attr_dict.keys()]):
                sum_values[both_nan_mask] = result_df[alternate_sum + '_Value'][both_nan_mask]
                sum_min_values[both_nan_mask] = result_df[alternate_sum + '_MinValue'][both_nan_mask]
                sum_max_values[both_nan_mask] = result_df[alternate_sum + '_MaxValue'][both_nan_mask]
                sum_transformed_values[both_nan_mask] = result_df[alternate_sum + '_TransformedValue'][both_nan_mask]

            both_tests_mask = ~nan_mask_1 & ~nan_mask_2
            sum_transformed_values[both_tests_mask] = ((sum_values[both_tests_mask] - sum_min_values[
                both_tests_mask]) * 100) / \
                                                      (sum_max_values[both_tests_mask] - sum_min_values[
                                                          both_tests_mask])

            single_test_mask = nan_mask_1 ^ nan_mask_2
            for i, mask in enumerate([nan_mask_1, nan_mask_2]):
                current_mask = single_test_mask & ~mask
                sum_values[current_mask] = result_df[test_pair[i] + '_Value'][current_mask].astype(str) + "!"
                sum_min_values[current_mask] = result_df[test_pair[i] + '_MinValue'][current_mask].astype(str) + "!"
                sum_max_values[current_mask] = result_df[test_pair[i] + '_MaxValue'][current_mask].astype(str) + "!"

            sum_transformed_values_single = ((sum_values[single_test_mask].str.replace('!', '', regex=False).astype(
                float) -
                                              sum_min_values[single_test_mask].str.replace('!', '', regex=False).astype(
                                                  float)) * 100) / \
                                            (sum_max_values[single_test_mask].str.replace('!', '', regex=False).astype(
                                                float) -
                                             sum_min_values[single_test_mask].str.replace('!', '', regex=False).astype(
                                                 float))
            sum_transformed_values[single_test_mask] = sum_transformed_values_single.astype(str) + "!"

            return sum_values, sum_min_values, sum_max_values, sum_transformed_values

        attr_dict = {
            '_TransformedValue': 'TransformedValue',
            '_Value': 'Value',
            '_MinValue': 'MinValue',
            '_MaxValue': 'MaxValue'
        }

        counter = 0
        for test_pair in tests_to_sum:
            sum_test_name = "Сумма " + " и ".join(test_pair)
            counter += 1
            if counter % 10000 == 0:
                print(f"Обработано {counter} групп")

            tests_columns_exist = all(
                [(test + key) in result_df.columns for test in test_pair for key in attr_dict.keys()])

            if tests_columns_exist:
                if alternate_sum:
                    sum_values, sum_min_values, sum_max_values, sum_transformed_values = calculate_values_with_alternate(
                        result_df, test_pair, attr_dict, alternate_sum)
                if equivalent:
                    sum_values, sum_min_values, sum_max_values, sum_transformed_values = calculate_values_with_equivalent(
                        result_df, test_pair, attr_dict, equivalent)

                result_df[sum_test_name + '_TransformedValue'] = sum_transformed_values
                result_df[sum_test_name + '_Value'] = sum_values
                result_df[sum_test_name + '_MinValue'] = sum_min_values
                result_df[sum_test_name + '_MaxValue'] = sum_max_values


        # if tests_to_sum:
        #     attr_dict = {
        #         '_TransformedValue': 'TransformedValue',
        #         '_Value': 'Value',
        #         '_MinValue': 'MinValue',
        #         '_MaxValue': 'MaxValue'
        #     }
        #
        #     counter = 0
        #
        #     for test_pair in tests_to_sum:
        #         # Создаем имя теста
        #         sum_test_name = "Сумма " + " и ".join(test_pair)
        #
        #         counter += 1
        #         if counter % 10000 == 0:
        #             print(f"Обработано {counter} групп")
        #
        #         # Проверяем наличие всех необходимых столбцов для каждого теста
        #         tests_columns_exist = all(
        #             [(test + key) in result_df.columns for test in test_pair for key in attr_dict.keys()])
        #
        #         # Если все столбцы существуют, применяем логику суммирования
        #         if tests_columns_exist:
        #             # Проверка на NaN значения
        #             nan_mask = result_df[test_pair[0] + '_Value'].isna() | result_df[test_pair[1] + '_Value'].isna()
        #
        #             # Суммирование значений
        #             sum_values = result_df[[test + '_Value' for test in test_pair]].sum(axis=1)
        #             sum_min_values = result_df[[test + '_MinValue' for test in test_pair]].sum(axis=1)
        #             sum_max_values = result_df[[test + '_MaxValue' for test in test_pair]].sum(axis=1)
        #
        #             # Вычисление TransformedValue
        #             sum_transformed_values = ((sum_values - sum_min_values) * 100) / (sum_max_values - sum_min_values)
        #
        #             # Применяем NaN маску
        #             sum_values[nan_mask] = np.nan
        #             sum_min_values[nan_mask] = np.nan
        #             sum_max_values[nan_mask] = np.nan
        #             sum_transformed_values[nan_mask] = np.nan
        #
        #             # Добавление результатов в DataFrame
        #             result_df[sum_test_name + '_TransformedValue'] = sum_transformed_values
        #             result_df[sum_test_name + '_Value'] = sum_values
        #             result_df[sum_test_name + '_MinValue'] = sum_min_values
        #             result_df[sum_test_name + '_MaxValue'] = sum_max_values

        print("Формирование файла Excel...")

        # Если папки не существует, создать её
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Сортировка result_df по UserId и ObjectType
        result_df = result_df.sort_values(by=['UserHrid', 'ObjectType'])

        def top_n_rows(group, n=3):
            column_name = "Сумма Тест на способности для Псковской области_Стандартный вариант для всех и Тест на интересы для Псковской области_Стандартный вариант для всех_TransformedValue"

            # Create a mask to identify '!' but make sure to replace NaN with False
            exclam_mask = group[column_name].astype(str).str.endswith('!').fillna(False)

            # Convert to float, removing '!' and replacing non-convertible strings with NaN
            def to_float(val):
                if isinstance(val, float):
                    return val

                try:
                    return float(val.replace('!', ''))
                except ValueError:
                    return np.nan

            group[column_name] = group[column_name].apply(to_float)

            # Return the top rows with the highest values
            top_rows = group.nlargest(n, column_name)

            # Return '!' where it was originally
            top_rows.loc[exclam_mask, column_name] = top_rows.loc[exclam_mask, column_name].astype(str) + '!'

            return top_rows

        if options['top_n_rows']:
            result_df = result_df.groupby('UserHrid').apply(top_n_rows).reset_index(drop=True)

        # Получите уникальные значения ObjectType
        unique_object_types = result_df['ObjectType'].unique()

        split_size = 1_000_000  # Размер каждого разделенного датафрейма

        with pd.ExcelWriter(f"{output_folder}/, {file_name}.xlsx", engine='xlsxwriter') as writer:
            for obj_type in unique_object_types:
                sub_df = result_df[result_df['ObjectType'] == obj_type]

                num_splits = len(sub_df) // split_size + (1 if len(sub_df) % split_size else 0)

                for split_num in range(num_splits):
                    start_idx = split_num * split_size
                    end_idx = start_idx + split_size
                    split_df = sub_df.iloc[start_idx:end_idx]

                    # Имя листа будет основываться на obj_type и номере раздела
                    split_df.to_excel(writer, sheet_name=f"{obj_type}_{split_num + 1}", index=False)

    @staticmethod
    def decompose_by_tests_and_variants(df):
        tests_data = {}
        print(df.columns)
        tests = df['ScreeningTestName'].dropna().unique()

        for test in tests:
            test_results = df[df['ScreeningTestName'] == test]
            variants_data = {}
            variants = test_results['VariantName'].dropna().unique()

            for variant in variants:
                variant_results = test_results[test_results['VariantName'] == variant]
                variants_data[variant] = variant_results

            tests_data[test] = {
                'test_results': test_results,
                'variants': variants_data
            }

        return tests_data

    @staticmethod
    def compose_by_municipality_and_school(result):
        municipalities = result['MunicipalityName'].dropna().unique()
        composed_data = {}

        for municipality in municipalities:
            clear_municipality = municipality.replace('"', '').strip()
            municipality_results = result[result['MunicipalityName'] == municipality]

            schools_data = {}
            schools = municipality_results['SchoolName'].dropna().unique()

            for school in schools:
                clear_school = school.replace('"', '').replace('/', '').strip()
                school_results = municipality_results[municipality_results['SchoolName'] == school]

                schools_data[clear_school] = {
                    'school_results': school_results,
                    'tests': DataComposer.decompose_by_tests_and_variants(school_results)
                }

            composed_data[clear_municipality] = {
                'municipality_results': municipality_results,
                'schools': schools_data,
                'tests': DataComposer.decompose_by_tests_and_variants(municipality_results)
            }

        composed_data['total'] = {
            'total_results': result,
            'tests': DataComposer.decompose_by_tests_and_variants(result)
        }

        return composed_data

    @staticmethod
    def compose_test_variants_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        # Развертываем колонку Variants
        df_exploded = df.explode('Variants')

        # Извлекаем 'Name' и '_id' из столбца 'Variants'
        df_exploded['VariantName'] = df_exploded['Variants'].apply(lambda x: x['Name'] if pd.notna(x) else None)
        df_exploded['VariantId'] = df_exploded['Variants'].apply(lambda x: x['_id'] if pd.notna(x) else None)

        # Удаляем исходный столбец 'Variants', так как он нам больше не понадобится
        df_exploded = df_exploded.drop('Variants', axis=1)

        # Преобразовываем UUID
        df_exploded = DataComposer._convert_ids_to_uuid(df_exploded, ['VariantId'])

        return df_exploded

    @staticmethod
    def _convert_ids_to_uuid(df: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
        def convert_to_uuid(id_value):
            if isinstance(id_value, uuid.UUID):
                return id_value
            else:
                return uuid.UUID(bytes=id_value)

        for col in id_columns:
            df[col] = df[col].apply(convert_to_uuid)
            df[col] = df[col].astype(str)

        return df

    @staticmethod
    def enrich_results_with_school_info(result, pupil_user, school_class, school_df):
        # Выберем необходимые столбцы для каждой таблицы
        result_selected = result
        pupil_user_selected = pupil_user[['UserId', 'SchoolClassId', 'Id']].rename(columns={'Id': 'PupilId'})
        school_class_selected = school_class[['Id', 'SchoolId']]
        school_df_selected = school_df[['_id', 'Number', 'IsDeleted']].rename(columns={'IsDeleted': 'SchoolIsDeleted'})

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

    @staticmethod
    def enrich_results_with_screening_test_info(result_df, screening_tests_df):
        """
        Обогащает исходный DataFrame информацией о ScreeningTestId.

        :param result_df: DataFrame с результатами.
        :param screening_tests_df: DataFrame с информацией о тестах.
        :return: Обогащенный DataFrame.
        """
        # Выбор необходимых столбцов
        screening_tests_df_selected = screening_tests_df[['_id', 'Name', 'VariantId', 'VariantName']]
        screening_tests_df_selected = screening_tests_df_selected.rename(columns={
            'Name': 'ScreeningTestName',
            'VariantName': 'VariantName'
        })

        # Объединение таблиц
        enriched_df = pd.merge(result_df, screening_tests_df_selected,
                               left_on=['ScreeningTestId', 'VariantId'],
                               right_on=['_id', 'VariantId'],
                               how='left')

        return enriched_df

    @staticmethod
    def enrich_results_with_education(results_df, answers_df):

        # Создаем tqdm объект для отслеживания прогресса
        tqdm.pandas(desc="Add education")

        # Создаём словарь для соответствия AnswerId и образования
        education_map = {
            "1c95cb5b-d1f9-4345-a378-6d46ff8e890b": 1,
            "c2e934c1-e08a-4a40-97ad-2d7313b3ccda": 2,
            "21bbf5c9-0ceb-419a-b7cd-b80b20012941": 3
        }

        tests_with_education_question = ["Тест на интересы для Псковской области",
                                         "Тест на интересы для ЯНАО",
                                         "Тест для Псковской области",
                                         "Тест для ЯНАО"
                                         ]

        # Инициализируем словарь, чтобы отслеживать обработанные сессии
        processed_sessions = {}

        answers_grouped = answers_df.groupby('SessionId')

        # Фильтруем строки с нужными тестами
        mask = results_df['ScreeningTestName'].isin(tests_with_education_question)
        new_rows = []
        for idx, row in tqdm(results_df[mask].iterrows(), total=len(results_df[mask]), desc="Add education"):
            session_id = row['SessionId']
            if session_id not in processed_sessions:
                if session_id in answers_grouped.groups:
                    answer_row = answers_grouped.get_group(session_id)
                    answer_id = answer_row.iloc[0]['AnswerId']
                    if answer_id in education_map:
                        transformed_value = education_map[answer_id]

                        new_row = row.copy()
                        new_row['TransformedValue'] = transformed_value
                        new_row['Value'] = transformed_value  # Если 'Value' также должно быть числом
                        new_row['MinValue'] = 1
                        new_row['MaxValue'] = 3
                        new_row['ObjectType'] = "SPO_VO"
                        new_row['Name'] = "SPO_VO"

                        new_rows.append(new_row)
                        processed_sessions[session_id] = new_row

        # Добавляем обработанные сессии к исходному датафрейму
        results_df = pd.concat([pd.DataFrame(new_rows), results_df], ignore_index=True)


        return results_df
    # @staticmethod
    # def enrich_results_with_education(results_df, answers_df):
    #     # Создаём словарь для соответствия AnswerId и образования
    #     education_map = {
    #         "1c95cb5b-d1f9-4345-a378-6d46ff8e890b": "Высшее",
    #         "c2e934c1-e08a-4a40-97ad-2d7313b3ccda": "Не знаю",
    #         "21bbf5c9-0ceb-419a-b7cd-b80b20012941": "Средне специальное"
    #     }
    #
    #     # Сортируем ответы по дате (по убыванию), чтобы последний ответ был первым
    #     answers_df = answers_df.sort_values(by='CreatedDate', ascending=False)
    #
    #     # Удаление дубликатов, оставляя только последний ответ (после сортировки)
    #     answers_df = answers_df.drop_duplicates(subset=['SessionId', 'QuestionId'], keep='first')
    #
    #     # Создаём новую колонку "СПО/ВО" и заполняем её значениями из словаря, используя merge
    #     merged_df = pd.merge(results_df, answers_df[['SessionId', 'AnswerId']], on='SessionId', how='left')
    #     merged_df['СПО/ВО'] = merged_df['AnswerId'].map(education_map)
    #
    #     # Если вы не хотите сохранять столбец AnswerId в конечном результате, вы можете его удалить
    #     merged_df.drop(columns=['AnswerId'], inplace=True)
    #
    #     return merged_df

    @staticmethod
    def enrich_results_with_municipality_info(result, profiles_df, municipalities_df):
        # Выберем необходимые столбцы для каждой таблицы
        result_selected = result

        profiles_df_selected = profiles_df[['UserId', 'MunicipalityId']]
        municipalities_df_selected = municipalities_df[['_id', 'Name']]

        # Переименуем столбцы для избежания конфликтов имен
        profiles_df_selected = profiles_df_selected.rename(columns={'MunicipalityId': 'MunicipalityId'})
        municipalities_df_selected = municipalities_df_selected.rename(
            columns={'_id': 'MunicipalityId', 'Name': 'MunicipalityName'})

        # Преобразуем MunicipalityId в строку для корректного соединения
        municipalities_df_selected['MunicipalityId'] = municipalities_df_selected['MunicipalityId'].astype(str)

        # Объединяем таблицы
        merged_df = pd.merge(result_selected, profiles_df_selected, how='left', on='UserId')
        merged_df = pd.merge(merged_df, municipalities_df_selected, how='left', on='MunicipalityId')

        return merged_df

    @staticmethod
    def enrich_results_with_additional_info(result, pupil_users, school_classes):

        pupil_users_selected = pupil_users[['UserId', 'SchoolClassId', 'UserHrid']]
        school_classes_selected = school_classes[['Id', 'Number', 'Letter']]

        school_classes_selected = school_classes[['Id', 'Number', 'Letter']].copy()


        # Создаем новый столбец 'ClassName' в school_classes, объединяя 'Number' и 'Letter'
        school_classes_selected['ClassName'] = school_classes_selected['Number'].astype(str) + school_classes_selected['Letter']


        # Слияние с pupil_users_selected
        merged_df = pd.merge(result, pupil_users_selected, how='left', on='UserId')

        # Слияние с school_classes_selected
        merged_df = pd.merge(merged_df, school_classes_selected[['Id', 'ClassName']], how='left',
                             left_on='SchoolClassId', right_on='Id')

        # Удаляем колонку 'SchoolClassId', так как она больше не нужна
        merged_df = merged_df.drop(columns=['SchoolClassId'])

        return merged_df

    @staticmethod
    def enrich_results_with_is_deleted(result, profiles_df):
        """
        Насыщает основной датафрейм данными из профилей пользователей.

        Параметры:
        - result (pd.DataFrame): Исходный датафрейм.
        - profiles_df (pd.DataFrame): Датафрейм профилей пользователей.

        Возвращает:
        - pd.DataFrame: Насыщенный датафрейм.
        """

        # Выберем необходимые столбцы из profiles_df
        profiles_df_selected = profiles_df[['UserId', 'IsDeleted']]

        # Объединяем основной датафрейм и выбранные столбцы из profiles_df по UserId
        merged_df = pd.merge(result, profiles_df_selected, how='left', on='UserId')

        return merged_df
    @staticmethod
    def enrich_results_with_technical_info(result, profiles_df):
        """
        Насыщает основной датафрейм данными из профилей пользователей.

        Параметры:
        - result (pd.DataFrame): Исходный датафрейм.
        - profiles_df (pd.DataFrame): Датафрейм профилей пользователей.

        Возвращает:
        - pd.DataFrame: Насыщенный датафрейм.
        """

        # Выберем необходимые столбцы из profiles_df
        profiles_df_selected = profiles_df[['UserId', 'City', 'CreatedDate', 'Birthday', 'FirstName', 'MiddleName',
                                            'LastName', 'PhoneNumber']]

        # Объединяем основной датафрейм и выбранные столбцы из profiles_df по UserId
        merged_df = pd.merge(result, profiles_df_selected, how='left', on='UserId')

        return merged_df

    @staticmethod
    def enrich_results_with_sessions_create_time(result_with_tech_info_df, sessions_df):
        """
        Насыщает основной датафрейм данными о времени создания сессии.

        Параметры:
        - result_with_tech_info_df (pd.DataFrame): Исходный датафрейм.
        - sessions_df (pd.DataFrame): Датафрейм сессий.

        Возвращает:
        - pd.DataFrame: Насыщенный датафрейм.
        """

        # Выберем необходимые столбцы из sessions_df и переименуем столбец 'Id' в 'sessionId'
        sessions_df_selected = sessions_df[['_id', 'CreatedDate', 'Completed']].\
            rename(columns={'_id': 'SessionId',
                            'CreatedDate': 'SessionCreatedDate',
                            'Completed': 'SessionCompleted'})

        # Объединяем основной датафрейм и выбранные столбцы из sessions_df по SessionId
        merged_df = pd.merge(result_with_tech_info_df, sessions_df_selected, how='left', on='SessionId')

        return merged_df

    @staticmethod
    def filter_test_info(enriched_result, folder_name, tested_school, tested_municipality):
        """
        Фильтрует датафрейм по типу теста.

        Параметры:
        - enriched_result (pd.DataFrame): Насыщенный датафрейм.

        Возвращает:
        - pd.DataFrame: Отфильтрованный датафрейм.
        """

        # Фильтруем enriched_result от тестовых школ и муниципалитетов, которые содержутся в tested_school и tested_municipality
        # Мы должны исключить содержание в tested_school и tested_municipality из датафрейма enriched_result

        tested_school = {'Тестовая школа', 'Demo школа №1', 'nan'}
        tested_municipality = {'Тестовый муниципалитет'}
        print("Enriched result shape:", enriched_result.shape)

        unique_municipalities = enriched_result['MunicipalityName'].unique()
        print("Enriched munic list:", unique_municipalities)

        filtered_df = enriched_result[~enriched_result['SchoolName'].isin(tested_school)]
        filtered_df = filtered_df[~filtered_df['MunicipalityName'].isin(tested_municipality)]

        print("Filtered result shape: ", filtered_df.shape)
        unique_municipalities = filtered_df['MunicipalityName'].unique()
        print("Filtered munic list:", unique_municipalities)

        return filtered_df

    @staticmethod
    def filter_objects(df, filter_Objects):
        """
        Фильтрует датафрейм по объектам.

        Параметры:
        - df (pd.DataFrame): Исходный датафрейм.
        - filter_Objects (list): Список объектов, которые нужно исключить из датафрейма.

        Возвращает:
        - pd.DataFrame: Отфильтрованный датафрейм.
        """

        filtered_df = df[~df['ObjectName'].isin(filter_Objects)]

        return filtered_df

    @staticmethod
    def choice_objects(df, choice_Objects):
        """
        Фильтрует датафрейм по объектам.

        Параметры:
        - df (pd.DataFrame): Исходный датафрейм.
        - choice_Objects (list): Список объектов, которые нужно оставить в датафрейме.

        Возвращает:
        - pd.DataFrame: Отфильтрованный датафрейм.
        """

        filtered_df = df[df['ObjectType'].isin(choice_Objects)]

        return filtered_df
