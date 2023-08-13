class HtmlOutWriter:
    def __init__(self, file_name: str):
        """
        Initialize the HtmlOutWriter with a filename to write to.
        :param file_name: The file name to write results to.
        """
        self.file_name = file_name

    def write(self, data: str) -> None:
        """
        Append a string of data to the file.
        :param data: The data to write.
        """
        with open(self.file_name, 'a') as f:
            f.write("\n\n")
            f.write(data)



    def clean_html(self) -> None:
        """
        Clean/initialize the html file by writing an empty string.
        """
        with open(self.file_name, 'w') as f:
            f.write('')

    def write_df_to_html(self, df: pd.DataFrame, title: str) -> None:
        """
        Write a DataFrame to the file in HTML format.
        :param df: The DataFrame to write.
        :param title: The title to give the data.
        """
        with open(self.file_name, 'a') as f:
            f.write(f'<h1>{title}</h1>')
            f.write(df.head(100).to_html())

    def write_df_to_html_with_style(self, dic: dict) -> None:
        """
        Write a DataFrame to the file in HTML format with some style.
        :param dic: The dictionary with DataFrame to write and title to give the data.
        """
        with open(self.file_name, 'a') as f:
            for key, value in dic.items():
                f.write(f'<h1>{key}</h1>')
                f.write(value.head(100).style.render())

    def write_results(self, objects: pd.DataFrame, results: pd.DataFrame, threshold_value: float) -> None:
        """
        Write the analysis results of all unique object types in the objects DataFrame to the file.
        :param objects: The DataFrame with object types.
        :param results: The DataFrame with results to analyze.
        :param threshold_value: The threshold value to use in the analysis.
        """
        object_types = objects['ObjectType'].unique()
        self.clean_html()

        for object_type in object_types:
            print(f"\nAnalyzing object type: {object_type}")
            analyzer = ra.ResultAnalyzer()

            grouped_counts, percentage_of_total_users, output_string = \
                analyzer.analyze_results(objects, results, object_type, threshold_value)

            if grouped_counts is not None:
                self.write(f"\n\nPercentage distribution and number of defined users for each object"
                           f" of the selected type '{object_type}':")
                self.write_df_to_html(grouped_counts, "Percentage distribution and number of defined users")
                print('percentage_of_total_users:')
                print(percentage_of_total_users)
                self.write("Percentage of total users: " + str(percentage_of_total_users))
                self.write(output_string)
