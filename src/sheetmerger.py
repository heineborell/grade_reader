import subprocess
import tools
import pandas as pd
import xlwt
from pathlib import Path
from rich import print


# get the column ids from original dataframe to replace them with simpler readeable ones
def get_column_ids(orig_columns, column_name):
    id = [i for i, item in enumerate(orig_columns) if item.startswith(column_name)]
    assert len(id) == 1, "sth wrong with renaming the dataframe columns!"
    return id[0]


def push_values(df, values, section_no):
    orig_columns = list(df.columns)
    column_name = orig_columns[get_column_ids(orig_columns, "midterm")]
    print(df["Username"])
    print(type(int(values[0])))
    if (df["Username"] == int(values[0])).any():
        print("[green]The student number exists![/green]")
        df.loc[df["Username"] == int(values[0]), column_name] = values[1]
        print(df["First Name"][df.loc[df["Username"] == int(values[0])].index[0]])
    else:
        print("[red]The student doesnt number exist![/red]")
        section_no = None

    return (df, column_name, section_no)


def merge_grade(df_path, section_no, values):
    # Load your LMS datasets
    save_path = Path(f"final_{section_no}.csv")
    df = pd.read_csv(df_path)
    pushed = push_values(df, values, section_no)
    df = pushed[0]
    column_name = pushed[1]
    # subprocess.run(["vd", "-f", "csv", "-"], input=df.to_csv(index=True), text=True)

    if pushed[2] is not None:
        if save_path.exists():
            print("path exist")
            final_df = pd.read_csv(save_path)

            # Make both DataFrames indexed by Username for alignment
            final_df.set_index("Username", inplace=True)
            df.set_index("Username", inplace=True)

            # Update only matching rows/columns
            final_df.update(df[[column_name]])

            # Write back to file
            final_df.reset_index(inplace=True)
            final_df.to_csv(save_path, index=False)

        else:
            with open(
                f"final_{section_no}.csv", "w", encoding="utf-8", newline=""
            ) as f:
                print("path doesnt exist")
                df.to_csv(f, index=False)
