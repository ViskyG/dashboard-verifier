import pandas as pd


class Tester:
    TEST_VALUES = []

    @staticmethod
    def set_test_values(values):
        """
        Устанавливает тестовые значения.

        :param values: List[Dict] - список словарей с тестовыми значениями.
        :return: None.
        """
        Tester.TEST_VALUES = values

    @staticmethod
    def check_values_in_dataframe(df):
        """
        Проверяет наличие заданных тестовых значений в датафрейме.

        :param df: DataFrame - исходный датафрейм.
        :return: None (выводит результаты в консоль).
        """
        for test_val in Tester.TEST_VALUES:
            mask = True
            for key, value in test_val.items():
                mask &= (df[key] == value)

            if df[mask].shape[0] > 0:
                print(f"Значение {test_val} присутствует в датафрейме.")
            else:
                print(f"Значение {test_val} отсутствует в датафрейме.")

