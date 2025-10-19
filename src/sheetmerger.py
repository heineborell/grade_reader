import subprocess
import tools
import pandas as pd
import xlwt


# get the column ids from original dataframe to replace them with simpler readeable ones
def get_column_ids(orig_columns, column_name):
    id = [i for i, item in enumerate(orig_columns) if item.startswith(column_name)]
    assert len(id) == 1, "sth wrong with renaming the dataframe columns!"
    return id[0]


def push_values(df, values):
    orig_columns = list(df.columns)
    column_name = orig_columns[get_column_ids(orig_columns, "midterm")]
    df.loc[df["Username"] == int(values[0]), column_name] = values[1]
    return df


def merge_grade(df_path, values):
    # Load your LMS datasets
    df = pd.read_csv(df_path)
    df = push_values(df, values)

    subprocess.run(["vd", "-f", "csv", "-"], input=df.to_csv(index=True), text=True)
