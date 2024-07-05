import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    """
    A class for visualizing data

    Attributes:
    - data (DataFrame): The dataset to be visualized

    Methods:
    - __init__: Initialize the Visualization object
    - create_subplots: Create subplots for visualization
    - remove_unused_axes: Remove unused axes from subplots
    - plot_columns: Plot columns of the dataset using various plot types
    - numerical_univariate_analysis: Perform univariate analysis for numerical features
    - categorical_univariate_analysis: Perform univariate analysis for categorical features
    - num_features_vs_target: Plot numerical features against the target variable
    - cat_features_vs_target: Plot categorical features against the target variable using box plots
    - facegrid_hist_target: Plot scatterplots of numerical features against the target variable using FacetGrid
    - plot_scatter_numericals_target: Plot scatter plots for numerical features with the target variable as hue
    """
    def __init__(self, data):
        """
        Initialize the Visualization object

        Parameters:
        - data (DataFrame): The dataset to be visualized and analyzed
        """
        self.data = data

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - figsize (tuple, optional): Figure size. Default is (18, 12)
        
        Returns:
        - fig: The figure object
        - ax (array of Axes): Array of axes objects
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def remove_unused_axes(self, fig, ax, num_plots):
        """
        Remove unused axes from the subplots grid

        Parameters:
        - fig (Figure): The figure object
        - ax (array of Axes): Array of axes objects
        - num_plots (int): Number of plots to be displayed
        """
        total_axes = len(ax)
        for j in range(num_plots, total_axes):
            fig.delaxes(ax[j])

    def plot_columns(self, cols, plot_func, ax, title_prefix="", target=None):
        """
        Plot columns from the dataset using the specified plotting function

        Parameters:
        - cols (list): List of column names to be plotted
        - plot_func (function): Plotting function (e.g., sns.boxplot, sns.histplot)
        - ax (array of Axes): Array of axes objects
        - title_prefix (str, optional): Prefix to be added to each plot title. Default is ""
        - target (str, optional): Target variable for plotting. Default is None
        """
        for i, col in enumerate(cols):
            if plot_func == sns.boxplot and target is not None:
                value_counts = self.data[col].value_counts()
                if len(value_counts) > 10:
                    top_values = value_counts.head(10)
                    plot_func(x=self.data[col], y=self.data[target], order=top_values.index, ax=ax[i])
                    ax[i].axhline(self.data[target].mean(), color="orange", linestyle="--")
                    ax[i].tick_params(axis="x", rotation=45)
                else:
                    plot_func(x=self.data[col], y=self.data[target], ax=ax[i])
                    ax[i].axhline(self.data[target].mean(), color="orange", linestyle="--")
            elif plot_func == sns.histplot:
                plot_func(self.data[col], ax=ax[i], kde=True)
            elif plot_func == sns.scatterplot and target is not None:
                plot_func(x=self.data[col], y=self.data[target], ax=ax[i])
            elif plot_func == sns.countplot:
                value_counts = self.data[col].value_counts()
                top_values = value_counts.head(10)
                if len(value_counts) > 10:
                    plot_func(x=self.data[col], order=top_values.index, ax=ax[i])
                    ax[i].tick_params(axis="x", rotation=45)
                else:
                    plot_func(x=self.data[col], order=top_values.index, ax=ax[i])
            else:
                plot_func(x=self.data[col], ax=ax[i])
            ax[i].set_title(f"{title_prefix}{col.capitalize()}")

    def numerical_univariate_analysis(self, rows, columns):
        """
        Perform univariate analysis on numerical columns and plot distributions

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data.select_dtypes(include="number")
        self.plot_columns(cols, sns.histplot, ax, title_prefix="Distribution of ")
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def categorical_univariate_analysis(self, cols, rows, columns):
        """
        Performs univariate analysis on categorical columns and plot count plots

        Parameters:
        - rows (int): Number of rows for subplots grid
        - cols (list): Columns to be used
        - columns (int): Number of columns for subplots grid
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data[cols]
        self.plot_columns(cols, sns.countplot, ax)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def num_features_vs_target(self, rows, columns, target):
        """
        Plots numerical features against the target variable

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - target (str): Name of the target variable
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(18, 12))
        cols = self.data.drop(columns=target).select_dtypes(include="number")
        self.plot_columns(cols, sns.scatterplot, ax, "IMDB Rating x ", target=target)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def cat_features_vs_target(self, rows, columns, target, cols, figsize):
        """
        Plot categorical features against the target variable using box plots

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - target (str): Name of the target variable
        - cols (list): List of column names for plotting
        - figsize (tuple): Figure size
        """
        fig, ax = self.create_subplots(rows, columns, figsize=figsize)
        cols = self.data[cols]
        self.plot_columns(cols, sns.boxplot, ax, "IMDB Rating x ", target=target)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()