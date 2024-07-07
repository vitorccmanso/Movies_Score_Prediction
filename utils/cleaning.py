import os

def cleaning_pipeline(data, save_folder, save_filename):
    """
    Performs a series of data cleaning operations on the input DataFrame 'data' and saves the cleaned data to a CSV file

    Parameters:
    - data (DataFrame): Input DataFrame containing movie data
    - save_folder (str): Folder path where the cleaned CSV file will be saved
    - save_filename (str): Filename for the cleaned CSV file

    Returns:
    - None
    """
    data = data.loc[:, ~data.columns.str.startswith('Unnamed')]
    data.columns = data.columns.str.lower()
    data["genre"] = data["genre"].str.split(", ")
    non_numeric = ~data["released_year"].str.isdigit()
    data.loc[non_numeric, "released_year"] = 1995
    data["runtime"] = data["runtime"].str.replace(" min", "").astype(int)
    data["gross"] = data[data["gross"].notna()]["gross"].astype(str).str.replace(",", "").astype(int)
    data['certificate'].fillna("Unrated", inplace=True)
    data['meta_score'].fillna(data['meta_score'].median(), inplace=True)
    data['gross'].fillna(data["gross"].median(), inplace=True)
    save_path = os.path.join(save_folder, f"{save_filename}.csv")
    processed_df = data.to_csv(save_path, index=False)
    return processed_df