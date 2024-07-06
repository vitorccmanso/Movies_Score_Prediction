import os
import ast

def certificate_grouping(certificate_name):
    """
    Group movie certificates into broad categories based on age suitability

    Parameters:
    - certificate_name (str): The certificate name or code to categorize

    Returns:
    - str: A string representing the age group category:
      - "All ages" for certificates like U, G, A, Passed, Approved
      - "Watch with parents" for certificates like PG, TV-PG, U/A, GP, UA
      - "13-15 year old" for certificates like PG-13, TV-14
      - "16-17 year old" for certificate 16
      - "Adults" for certificates like R, TV-MA
      - "Unrated" for any other certificates not listed
    """
    if certificate_name in ["U", "G", "Passed", "Approved"]:
        return "All ages"
    elif certificate_name in ["PG", "TV-PG", "U/A", "GP", "UA"]:
        return "Watch with parents"
    elif certificate_name in ["PG-13", "TV-14", "16"]:
        return "13-16 year old"
    elif certificate_name in ["R", "TV-MA", "A"]:
        return "Adults"
    else:
        return "Unrated"

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
    data['certificate'] = data['certificate'].apply(certificate_grouping)
    data['meta_score'].fillna(data['meta_score'].median(), inplace=True)
    data['gross'].fillna(data["gross"].median(), inplace=True)
    save_path = os.path.join(save_folder, f"{save_filename}.csv")
    processed_df = data.to_csv(save_path, index=False)

    return processed_df