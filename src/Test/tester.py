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
        precision = 5  # Количество знаков после десятичной точки, до которых вы хотите округлить

        # В функции check_values_in_dataframe
        for test_val in Tester.TEST_VALUES:
            mask = pd.Series([True] * len(df), index=df.index)

            for key, value in test_val.items():
                # Проверка на наличие столбца в датафрейме
                if key not in df.columns:
                    print(f"Поле {key} отсутствует в датафрейме.")
                    mask = pd.Series([False] * len(df), index=df.index)
                    break

                if df[key].dtype == 'float64':  # Если столбец содержит значения float
                    rounded_value = round(float(value), precision)
                    # Если округленное значение является целым числом, конвертируем его в int перед преобразованием в строку
                    value_str = str(int(rounded_value)) if rounded_value.is_integer() else str(rounded_value)
                    mask &= (df[key].round(precision).astype(str) == value_str)
                else:
                    # Приводим значения в датафрейме к строковому типу перед сравнением
                    mask &= (df[key].astype(str) == str(value))

            if mask.any() and df[mask].shape[0] > 0:
                print(f"Значение {test_val} присутствует в датафрейме.")
            else:
                print(f"Значение {test_val} отсутствует в датафрейме.")





