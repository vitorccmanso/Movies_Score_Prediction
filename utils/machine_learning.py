import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
import pickle

mlflow_tracking_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
uri = os.environ.get("uri")

class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    """
    A custom transformer for multi-label binarization of a specified column

    This class uses the MultiLabelBinarizer from scikit-learn to transform a multi-label column
    into a binary matrix, where each label is represented as a separate feature

    Attributes:
    - mlb (MultiLabelBinarizer): The MultiLabelBinarizer instance used for transformation
    - classes_ (array-like): The list of all unique classes observed in the data
    - column (str): The name of the column to be transformed

    Methods:
    - fit: Fits the MultiLabelBinarizer to the specified column in the data
    - transform: Transforms the specified column in the data using the fitted MultiLabelBinarizer
    """
    def __init__(self, column):
        """
        Initializes the MultiLabelEncoder instance

        Parameters:
        - mlb (MultiLabelBinarizer): The MultiLabelBinarizer instance used for transformation
        - classes_ (array-like): The list of all unique classes observed in the data
        - column (str): The name of the column to be transformed
        """
        self.mlb = MultiLabelBinarizer()
        self.classes_ = None
        self.column = column

    def fit(self, df, y=None):
        """
        Fits the MultiLabelBinarizer to the specified column in the DataFrame

        Parameters:
        - df (DataFrame): The input DataFrame containing the column to be transformed
        - y: Ignored, exists for compatibility with scikit-learn's TransformerMixin

        Returns:
        - self: The fitted MultiLabelEncoder instance
        """
        self.mlb.fit(df[self.column])
        # Store fitted classes
        self.classes_ = self.mlb.classes_
        return self

    def transform(self, df):
        """
        Transforms the specified column in the DataFrame into a binary matrix

        Parameters:
        - df (DataFrame): The input DataFrame containing the column to be transformed

        Returns:
        - array-like: A binary matrix where each column represents a unique class from the specified column
        """
        data = df[self.column]
        return self.mlb.transform(data).reshape(df.shape[0], -1)

class DataPreprocess:
    """
    A class for preprocessing data including feature engineering, transformation, and splitting into train-test sets

    Methods:
    - __init__: Initializes the DataPreprocess object
    - save_preprocessor: Saves the preprocessor object to a file
    - get_feature_names: Retrieves the feature names after preprocessing
    - preprocessor: Creates and returns a preprocessor pipeline for data preprocessing
    - preprocess_data: Preprocesses the input data, including feature engineering, transformation, and splitting into train-test sets
    """
    def __init__(self):
        pass

    def save_preprocessor(self, preprocessor, path):
        """
        Saves the preprocessor object to a file

        Parameters:
        - preprocessor: The preprocessor object to be saved
        - path: The file path where the preprocessor will be saved
        """
        if not os.path.exists("../artifacts"):
            os.makedirs("../artifacts")
        with open(path, "wb") as f:
            pickle.dump(preprocessor, f)
    
    def get_feature_names(self, preprocessor, no_transformation_cols, log_cols, cbrt_cols, one_hot_cols, multi_label_cols, onehot):
        """
        Retrieves the feature names after preprocessing

        Parameters:
        - preprocessor: The preprocessor object
        - coordinate_cols: List of column names representing coordinate features
        - log_cols: List of column names for which log transformation is applied
        - cbrt_cols: List of column names for which cubic root transformation is applied
        - one_hot_cols: List of column names for which onehot encoding is applied
        - multi_label_cols: List of column names for which multilabel encoding is applied
        - onehot: Boolean indicating whether get the onehot encoded feature names

        Returns:
        - feature_names: List of feature names after preprocessing
        """
        numeric_features = no_transformation_cols + log_cols + cbrt_cols
        one_hot_features = []
        if onehot:
            one_hot_features = list(preprocessor.named_transformers_["cat_onehot"]["onehot"].get_feature_names_out(one_hot_cols))
        multilabel_features = []
        for col in multi_label_cols:
            multilabel_features.extend([f"{col}_" + str(cls) for cls in preprocessor.named_transformers_[f"cat_multi_label_{col}"].classes_])
        feature_names = numeric_features + one_hot_features + multilabel_features
        return feature_names

    def preprocessor(self, no_transformation_cols, log_cols, cbrt_cols, one_hot_cols, multi_label_cols, onehot):
        """
        Creates and returns a preprocessor pipeline for data preprocessing

        Parameters:
        - coordinate_cols: List of column names representing coordinate features
        - log_cols: List of column names for which log transformation is applied
        - cbrt_cols: List of column names for which cubic root transformation is applied
        - one_hot_cols: List of column names for which onehot encoding is applied
        - multi_label_cols: List of column names for which multilabel encoding is applied
        - onehot: Boolean indicating whether to apply one-hot encoding

        Returns:
        - preprocessor: Preprocessor pipeline for data preprocessing
        """
        # Define transformers for numeric columns
        log_transformer = Pipeline(steps=[
            ("log_transformation", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", RobustScaler())
        ])
        cubic_transformer = Pipeline(steps=[
            ("sqrt_transformation", FunctionTransformer(np.cbrt, validate=True)),
            ("scaler", RobustScaler())
        ])

        #Define transformer for categorical columns
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder())
        ])

        # Combine transformers for numeric and categorical columns
        transformers=[
            ("num_only_scale", RobustScaler(), no_transformation_cols),
            ("num_log", log_transformer, log_cols),
            ("num_sqrt", cubic_transformer, cbrt_cols)
        ]

        if onehot:
            transformers.append(("cat_onehot", categorical_transformer, one_hot_cols))
        for col in multi_label_cols:
            transformers.append((f"cat_multi_label_{col}", MultiLabelEncoder(column=col), [col]))
        preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)
        return preprocessor


    def preprocess_data(self, data, test_size, target_name, onehot=True):
        """
        Preprocesses the input data, including feature engineering, transformation, and splitting into train-test sets

        Parameters:
        - data: Input DataFrame containing the raw data
        - test_size: The proportion of the dataset to include in the test split
        - target_name: Name of the target variable
        - onehot: Boolean indicating whether the dataframe is mapped with the onehot encoding values

        Returns:
        - X_train: Features of the training set
        - X_test: Features of the testing set
        - y_train: Target labels of the training set
        - y_test: Target labels of the testing set
        """
        data_process = data.drop(columns=[target_name])

        # Specify columns needing log transformation, square root transformation
        log_cols = ["runtime", "no_of_votes"]
        cbrt_cols = ["gross", "gross_per_vote"]
        no_transformation_cols = ["released_year", "meta_score"]
        if onehot == False:
            no_transformation_cols = no_transformation_cols + ["certificate", "director", "star1", "star2", "star3", "star4"]
        one_hot_cols = data_process.drop(columns=["genre"]).select_dtypes("object").columns
        multi_label_cols = ["genre"]

        # Build preprocessor
        preprocessor = self.preprocessor(no_transformation_cols, log_cols, cbrt_cols, one_hot_cols, multi_label_cols, onehot=onehot)

        # Fit and transform data
        data_preprocessed = preprocessor.fit_transform(data_process)

        feature_names = self.get_feature_names(preprocessor, no_transformation_cols, log_cols, cbrt_cols, one_hot_cols, multi_label_cols, onehot=onehot)
        data_preprocessed = pd.DataFrame(data_preprocessed, columns=feature_names)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, data[target_name], test_size=test_size, shuffle=True, random_state=42)

        # Save preprocessor if not already saved
        path = "../artifacts/preprocessor_label.pkl"
        if onehot:
            path = "../artifacts/preprocessor_onehot.pkl"
        if not os.path.exists(path):
            self.save_preprocessor(preprocessor, path)

        return X_train, X_test, y_train, y_test

class ModelTraining:
    """
    A class for training machine learning models, evaluating their performance and save the best one

    Methods:
    - __init__: Initializes the ModelTraining object
    - save_model: Saves the model to a pkl file
    - initiate_model_trainer: Initiates the model training process
    - evaluate_models: Evaluates multiple models using random search cross-validation and logs the results with MLflow
    """
    def __init__(self):
        pass

    def save_model(self, model_name, save_folder, save_filename):
        """
        Save the model to a pkl file

        Parameters:
        - model_name (dict): The model to save
        - save_folder (str): Folder path where the model will be saved
        - save_filename (str): Filename for the pkl file
        """
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)

        # Get the latest version of the registered model
        latest_version = client.get_latest_versions(model_name)[-1]

        # Construct the logged model path
        run_id = latest_version.run_id
        artifact_path = latest_version.source.split('/')[-1]
        logged_model = f'runs:/{run_id}/{artifact_path}'

        # Load the model from MLflow and saves it to a pkl file
        loaded_model = mlflow.sklearn.load_model(logged_model)
        file_path = os.path.join(save_folder, f"{save_filename}.pkl")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(loaded_model, f)

    def initiate_model_trainer(self, train_test, experiment_name):
        """
        Initiates the model training process

        Parameters:
        - train_test: A tuple containing the train-test split data in the format (X_train, y_train, X_test, y_test)
        - experiment_name: Name of the MLflow experiment where the results will be logged
        
        Returns:
        - dict: A dictionary containing the evaluation report for each model
        """
        mlflow.set_tracking_uri(uri)
        X_train, y_train, X_test, y_test = train_test
        
        models = {
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
        }
        
        params = {
            "Ridge": {
                "alpha":[0.1, 0.5, 1, 10, 100], 
                "max_iter":[1000, 3000, 5000], 
                "tol": [0.0001, 0.001, 0.01, 0.1],
                "random_state": [42]
            },
            "Lasso":{
                "alpha":[0.1, 0.5, 1, 10, 100], 
                "max_iter":[1000, 3000, 5000], 
                "tol": [0.0001, 0.001, 0.01, 0.1],
                "random_state": [42]
            },
            "Random Forest":{
                "criterion":["squared_error", "absolute_error", "poisson"],
                "max_features":["sqrt","log2"],
                "n_estimators": [5, 10, 25, 50, 100],
                "max_depth": [5, 10, 20, 30],
                "random_state": [42]
            },
            "Gradient Boosting":{
                "loss":["squared_error", "absolute_error", "quantile"],
                "max_features":["sqrt","log2"],
                "n_estimators": [5, 10, 25, 50, 100],
                "max_depth": [5, 10, 20, 30],
                "learning_rate": [0.001, 0.01, 0.1],
                "random_state": [42]
            },
        }
        
        model_report = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                           models=models, params=params, experiment_name=experiment_name)
        
        return model_report

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params, experiment_name):
        """
        Evaluates multiple models using random search cross-validation and logs the results with MLflow

        Parameters:
        - X_train: Features of the training data
        - y_train: Target labels of the training data
        - X_test: Features of the testing data
        - y_test: Target labels of the testing data
        - models: A dictionary containing the models to be evaluated
        - params: A dictionary containing the hyperparameter grids for each model
        - experiment_name: Name of the MLflow experiment where the results will be logged
        
        Returns:
        - dict: A dictionary containing the evaluation report for each model
        """
        mlflow.set_experiment(experiment_name)
        report = {}
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                param = params[model_name]
                gs = GridSearchCV(model, param, cv=5, scoring=["neg_mean_absolute_error", "r2"], refit="neg_mean_absolute_error")
                search_result = gs.fit(X_train, y_train)
                model = search_result.best_estimator_
                y_pred = model.predict(X_test)

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = root_mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Log metrics to MLflow
                mlflow.log_params(search_result.best_params_)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.sklearn.log_model(model, model_name, registered_model_name=f"{model_name} - {experiment_name}")
                
                # Store the model for visualization
                report[model_name] = {"model": model, "y_pred": y_pred, "mae": mae, "rmse": rmse, "r2": r2}        
        return report


class MetricsVisualizations:
    """
    A class for visualizing model evaluation metrics and results

    Attributes:
    - models: A dictionary containing the trained models

    Methods:
    - __init__: Initializes the MetricsVisualizations object with a dictionary of models
    - plot_pred_x_real: Plots predicted vs real values for each model
    - plot_feature_importance: Plots feature importance for each model
    - plot_residuals: Plots residuals, residual autocorrelation, and residual distribution for each model
    """
    def __init__(self, models):
        """
        Initializes the MetricsVisualizations object with a dictionary of models

        Parameters:
        - models: A dictionary containing the trained models
        """
        self.models = models

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows: Number of rows for subplots grid
        - columns: Number of columns for subplots grid
        - figsize: Figure size. Default is (18, 12)
        
        Returns:
        - fig: The figure object
        - ax: Array of axes objects
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def plot_pred_x_real(self, y_test, rows, columns):
        """
        Plots predicted vs real values for each model

        Parameters:
        - y_test: True labels of the test data
        - rows: Number of rows for subplots
        - columns: Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Get predicted values and make a copy of y_test
            y_pred = pd.Series(model_data["y_pred"])
            y_test_copy = y_test.copy()

            # Reset indices for easy plotting
            y_pred.reset_index(drop=True, inplace=True)
            y_test_copy.reset_index(drop=True, inplace=True)

            # Create DataFrame for plotting
            df_plot = pd.DataFrame({"Predicted Values": y_pred.values, "Real Values": y_test_copy.values})

            # Plot scatter plot and regression line
            sns.scatterplot(data=df_plot, x="Predicted Values", y="Real Values", ax=ax[i])
            sns.regplot(data=df_plot, x="Predicted Values", y="Real Values", ax=ax[i], scatter=False, color="red", line_kws={"linewidth": 2})
            ax[i].set_title(f"Predicted x Real Values: {model_name}")
            ax[i].set_xlabel("Predicted Values")
            ax[i].set_ylabel("Real Values")

        fig.tight_layout()
        plt.show()

    def plot_feature_importance(self, y_test, X_test, metric, rows, columns):
        """
        Plots feature importance for each model

        Parameters:
        - y_test: True labels of the test data
        - X_test: Features of the test data
        - metric: Metric used for evaluating feature importance
        - rows: Number of rows for subplots
        - columns: Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Calculate and sort permutation importances
            result = permutation_importance(model_data["model"], X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring=metric)
            sorted_importances_idx = result["importances_mean"].argsort()[::-1]

            # Select top 5 features
            top_features_idx = sorted_importances_idx[:5]
            top_features = X_test.columns[top_features_idx]
            importances = pd.DataFrame(result.importances[top_features_idx].T, columns=top_features)

            # Plot boxplot of feature importances
            box = importances.plot.box(vert=False, whis=10, ax=ax[i])
            box.set_title(f"Top 5 Feature Importance - {model_name}")
            box.axvline(x=0, color="k", linestyle="--")
            box.set_xlabel(f"Increase in MAE")
            box.figure.tight_layout()

        fig.tight_layout()
        plt.show()

    def plot_residuals(self, y_test, rows, columns):
        """
        Plots residuals, residual autocorrelation, and residual distribution for each model

        Parameters:
        - y_test: True labels of the test data
        - rows: Number of rows for subplots
        - columns: Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Calculate residuals
            y_pred = model_data["y_pred"]
            residuals = y_test - y_pred

            # Plot residual plot
            sns.scatterplot(x=y_pred, y=residuals, ax=ax[i * columns])
            ax[i * columns].set_xlabel("Predicted Values")
            ax[i * columns].set_ylabel("Residuals")
            ax[i * columns].axhline(y=0, color="r", linestyle="--")
            ax[i * columns].set_title(f"Residual Plot - {model_name}")

            # Plot residual autocorrelation
            plot_acf(residuals, lags=40, ax=ax[i * columns + 1])
            ax[i * columns + 1].set_title(f"Residual Autocorrelation - {model_name}")
            ax[i * columns + 1].set_xlabel("Lags")
            ax[i * columns + 1].set_ylabel("Autocorrelation")

            # Plot residual distribution
            stats.probplot(residuals, dist="norm", plot=ax[i * columns + 2])
            ax[i * columns + 2].set_title(f"Residual Distribution - {model_name}")

        fig.tight_layout()
        plt.show()