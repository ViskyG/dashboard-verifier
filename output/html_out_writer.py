class HtmlOutWriter:
    def __init__(self, file_name):
        self.file_name = file_name

    def write(self, data):
        with open(self.file_name, 'a') as f:
            f.write("\n\n")  # это добавит отступы в виде двух новых строк перед вашими данными
            f.write(data)

    def clean_html(self):
        with open(self.file_name, 'w') as f:
            f.write('')

    def write_df_to_html(self, df, title):
        with open(self.file_name, 'a') as f:
            f.write(f'<h1>{title}</h1>')
            f.write(df.head(100).to_html())

    def write_df_to_html_with_style(self, dic: dict):
        with open(self.file_name, 'a') as f:
            for key, value in dic.items():
                f.write(f'<h1>{key}</h1>')
                f.write(value.head(100).style.render())

