import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import skew, mode
import pickle

def saving_dataset(data, save_folder, save_filename):
    """
    Saves the dataset, and creates the specified folder if it doesn't exist

    Parameters:
    - data: DataFrame containing the original dataset
    - save_folder: Folder path where the datasets will be saved
    - save_filename: Base filename for the saved datasets
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    original_save_path = os.path.join(save_folder, f"{save_filename}.csv")
    data.to_csv(original_save_path, index=False)

class Plots:
    """
    A class for plotting data distributions and correlations

    Attributes:
    - data (DataFrame): The dataset to be visualized

    Methods:
    - __init__: Initialize the Plots object with a DataFrame
    - plot_transformed_distributions: Plot distributions of a numeric column and its transformed versions
    - plot_corr: Plot a heatmap of the correlation matrix for numerical columns in the DataFrame
    """
    def __init__(self, data):
        """
        Initialize the Plots object with a DataFrame

        Parameters:
        - data: DataFrame containing the data
        """
        self.data = data

    def plot_transformed_distributions(self, col):
        """
        Plot distributions of a numeric column and its transformed versions

        Parameters:
        - col (str): Name of the column to plot
        """
        data = self.data[col]
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))
        ax = ax.ravel()

        # Plot original distribution
        sns.histplot(data, kde=True, ax=ax[0])
        ax[0].set_title(f"Original Distribution\n Skewness:{round(skew(data), 2)}")
        self.plot_legends(ax, 0, data)

        # Apply log transformation
        log_transformed = np.log1p(data)
        sns.histplot(log_transformed, kde=True, ax=ax[1])
        ax[1].set_title(f"Log Transformation\n Skewness: {round(skew(log_transformed), 2)}")
        self.plot_legends(ax, 1, log_transformed)

        # Apply cubic root transformation
        sqrt_transformed = np.cbrt(data)
        sns.histplot(sqrt_transformed, kde=True, ax=ax[2])
        ax[2].set_title(f"Cubic Transformation\n Skewness: {round(skew(sqrt_transformed), 2)}")
        self.plot_legends(ax, 2, sqrt_transformed)

        skewness_values = [skew(data), skew(log_transformed), skew(sqrt_transformed)]
        closest_to_zero_idx = np.argmin(np.abs(skewness_values))

        # Plot boxplot of the transformation with the lowest skewness
        if closest_to_zero_idx == 0:
            sns.boxplot(x=data, ax=ax[3])
        elif closest_to_zero_idx == 1:
            sns.boxplot(x=log_transformed, ax=ax[3])
        else:
            sns.boxplot(x=sqrt_transformed, ax=ax[3])
        ax[3].set_title(f"Boxplot: Best Distribution")
        plt.tight_layout()
        plt.show()

    def plot_legends(self, ax, pos, data):
        """
        Plot legends on a given axis for a specific plot position

        Parameters:
        - ax: Axis object to plot on
        - pos (int): Position in the subplot grid
        - data: Data for which legends are plotted
        """
        ax[pos].axvline(data.mean(), color='r', linestyle='--', label='Mean: {:.2f}'.format(data.mean()))
        ax[pos].axvline(mode(data)[0], color='g', linestyle='--', label='Mode: {:.2f}'.format(mode(data)[0]))
        ax[pos].axvline(data.median(), color='b', linestyle='--', label='Median: {:.2f}'.format(data.median()))
        ax[pos].legend()

    def plot_corr(self, method, figsize):
        """
        Plots a heatmap of the correlation matrix for numerical columns in the DataFrame

        Parameters:
        - method: Correlation method to use ('pearson', 'kendall', or 'spearman')
        """
        plt.figure(figsize=figsize)
        sns.heatmap(self.data.select_dtypes(include="number").corr(method=method), annot=True, fmt=".2f", cmap="RdYlGn")
        plt.show()

class ColumnMapping:
    """
    A class for creating and managing categorical column mappings in a DataFrame

    Attributes:
    - data (pd.DataFrame): The DataFrame containing the data to process
    - column_mappings (dict): Dictionary to store mappings of categorical columns

    Methods:
    - __init__: Initialize the ColumnMapping object with a DataFrame
    - save_mapping: Save the column mappings to a file
    - certificate_grouping: Group movie certificates into broad categories based on age suitability
    - create_column_mapping: Create mappings for categorical columns and apply them to the DataFrame
    """
    def __init__(self, data):
        """
        Initialize the ColumnMapping object with a DataFrame

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data to process
        """
        self.data = data
        self.column_mappings = {}

    def save_mapping(self, mapping, save_folder, save_filename):
        """
        Save the column mappings to a pkl file

        Parameters:
        - mapping (dict): The column mapping to save
        - save_folder (str): Folder path where the collumn mapping will be saved
        - save_filename (str): Filename for the pkl file
        """
        file_path = os.path.join(save_folder, f"{save_filename}.pkl")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(mapping, f)

    def certificate_grouping(self, certificate_name, mapping_type=1):
        """
        Group movie certificates into broad categories based on age suitability

        Parameters:
        - certificate_name (str): The certificate name or code to categorize
        - mapping_type (int): Type of mapping to use:
            - 1: Categorical mapping for one-hot encoding
            - 2: Label mapping to use as numerical feature

        Returns:
        - str: A string representing the age group category or an integer based on the mapping_type:
        - Original mapping:
            - "all_ages" for certificates like U, G, A, Passed, Approved
            - "watch_with_parents" for certificates like PG, TV-PG, U/A, GP, UA
            - "teens" for certificates like PG-13, TV-14, 16
            - "adults" for certificates like R, TV-MA
            - "unrated" for any other certificates not listed
        - Modified mapping (integer):
            - 0 for "all_ages"
            - 1 for "watch_with_parents"
            - 2 for "teens"
            - 3 for "adults"
            - -1 for "unrated"
        """
        if mapping_type == 1:
            if certificate_name in ["u", "g", "passed", "approved"]:
                return "all_ages"
            elif certificate_name in ["pg", "tv-pg", "u/a", "gp", "ua"]:
                return "watch_with_parents"
            elif certificate_name in ["pg-13", "tv-14", "16"]:
                return "teens"
            elif certificate_name in ["r", "tv-ma", "a"]:
                return "adults"
            else:
                return "unrated"
        elif mapping_type == 2:
            if certificate_name in ["u", "g", "passed", "approved"]:
                return 0
            elif certificate_name in ["pg", "tv-pg", "u/a", "gp", "ua"]:
                return 1
            elif certificate_name in ["pg-13", "tv-14", "16"]:
                return 2
            elif certificate_name in ["r", "tv-ma", "a"]:
                return 3
            else:
                return -1

    def create_column_mapping(self, cat_cols, mapping_type=1):
        """
        Create mappings for categorical columns and apply them to the DataFrame

        Parameters:
        - cat_cols (list): List of categorical columns to map
        - mapping_type (int): Type of mapping to use:
            - 1: Categorical mapping for one-hot encoding
            - 2: Label mapping to use as numerical feature

        Returns:
        - DataFrame: The transformed DataFrame
        - dict: Dictionary of column mappings
        """
        for col in cat_cols:
            col_counts = self.data[col].value_counts()
            if mapping_type == 1:
                col_category_mapping = {
                    value: self.certificate_grouping(value, mapping_type=1) if col == "certificate" 
                    else (
                        "multiple_movies" if count > 5 
                        else "few_movies" if count >= 2 
                        else "one_movie"
                    )
                    for value, count in col_counts.items()
                }
                default_value = "one_movie"
            elif mapping_type == 2:
                col_category_mapping = {
                    value: self.certificate_grouping(value, mapping_type=2) if col == "certificate"
                    else (
                        3 if count > 5
                        else 2 if count >= 2
                        else 1
                    )
                    for value, count in col_counts.items()
                }
                default_value = 1
            self.data[col] = self.data[col].apply(lambda x: col_category_mapping.get(x, default_value))
            self.column_mappings[col] = col_category_mapping
        return self.data, self.column_mappings

