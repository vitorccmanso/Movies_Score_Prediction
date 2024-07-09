import pandas as pd
import pickle
import ast

class PredictPipeline:
    """
    A class for predicting movie ratings using a pre-trained model and preprocessing pipeline

    Methods:
    - __init__: Initializes the PredictPipeline object by loading the preprocessor and model from .pkl files
    - process_dataset: Processes the input dataset, ensuring it contains the required columns and reordering them if necessary
    - values_to_lowercase: Converts the values of object type columns to lowercase
    - categorize_columns: Categorizes specified columns in the DataFrame based on a predefined mapping. If a value is not found in the mapping, it is replaced with 'New'
    - get_feature_names: Retrieves the feature names after preprocessing
    - preprocess_data: Preprocesses the input data, including feature engineering and transformation
    - predict: Predicts movie ratings based on the input data
    """
    def __init__(self):
        """
        Initializes the PredictPipeline object by loading the mapping, preprocessor and model from .pkl files
        """
        # Load mapping
        mapping_path = "app/artifacts/column_mappings_onehot.pkl"
        with open(mapping_path, "rb") as f:
            self.mapping = pickle.load(f)

        # Load preprocessor
        preprocessor_path = "app/artifacts/preprocessor_onehot.pkl"
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        # Load model
        model_path = "app/artifacts/model.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def process_dataset(self, input_data):
        """
        Processes the input dataset, ensuring it contains the required columns and reordering them if necessary

        Parameters:
        - input_data (DataFrame): The input dataset to be processed

        Returns:
        - pandas.DataFrame: The processed dataset
        """
        columns = ["released_year", "certificate", "runtime",
                    "genre", "meta_score", "director", "star1",
                    "star2", "star3", "star4", "no_of_votes", "gross"]
        input_data.columns = input_data.columns.str.lower().str.replace(r"\[.*\]", "", regex=True).str.rstrip().str.replace(" ", "_")

        # Check if the uploaded dataset contains all required columns
        if not set(columns).issubset(input_data.columns):
            raise ValueError("Dataset must contain all the columns listed above")

        filtered_data = input_data[columns]
        reordered_data = filtered_data.reindex(columns=columns)
        return reordered_data

    def values_to_lowercase(self, df, manual):
        """
        Converts the values of object type columns to lowercase

        Parameters:
        - df (DataFrame): The input DataFrame
        - manual (bool): A boolean indicating whether it's a manual prediction or not

        Returns:
        - DataFrame: The transformed DataFrame
        """
        for col in df.select_dtypes("object"):
            if col != "genre":
                df[col] = df[col].str.lower()
            else:
                if manual:
                    df[col] = df[col].apply(lambda x: [genre.lower() for genre in x])
                else:
                    df[col] = df[col].str.lower()
                    df[col] = df[col].apply(ast.literal_eval)
        return df

    def categorize_columns(self, df):
        """
        Categorizes specified columns in the DataFrame based on a predefined mapping. If a value is not found 
        in the mapping, it is replaced with 'New'

        Parameters:
        - df (DataFrame): The input DataFrame containing the columns to be categorized

        Returns:
        - DataFrame: The DataFrame with categorized columns
        """
        for col in self.mapping.keys():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: self.mapping[col].get(x, "one_movie"))
        return df

    def get_feature_names(self, one_hot_cols):
        """
        Retrieves the feature names after preprocessing

        Parameters:
        - one_hot_cols: Onehot encoded columns in the dataset

        Returns:
        - list: List of feature names
        """
        log_cols = ["runtime", "no_of_votes"]
        cbrt_cols = ["gross", "gross_per_vote"]
        no_transformation_cols = ["released_year", "meta_score"]
        numeric_features = no_transformation_cols + log_cols + cbrt_cols

        # Get the categorical features names
        one_hot_features = list(self.preprocessor.named_transformers_['cat_onehot']['onehot'].get_feature_names_out(one_hot_cols))
        multilabel_features = ["genre_" + str(cls) for cls in self.preprocessor.named_transformers_["cat_multi_label_genre"].classes_]

        feature_names = numeric_features + one_hot_features + multilabel_features
        return feature_names

    def preprocess_data(self, data, manual):
        """
        Preprocesses the input data, including feature engineering and transformation

        Parameters:
        - data: The input data to be preprocessed
        - manual: A boolean indicating whether it's a manual prediction or not

        Returns:
        - pandas.DataFrame: The preprocessed data
        """
        # Create the gross_per_vote column and select the onehot encoded columns
        data["gross_per_vote"] = data["gross"] / data["no_of_votes"]
        one_hot_cols = data.drop(columns=["genre"]).select_dtypes("object").columns

        # Standardize the dataset to have lowercase values and apply the mapping to categorical columns
        data = self.values_to_lowercase(data, manual=manual)
        data = self.categorize_columns(data)

        # Transform the data with the preprocessor
        data = self.preprocessor.transform(data)
        feature_names = self.get_feature_names(one_hot_cols)

        # Create a DataFrame with the transformed data and feature names
        new_data = pd.DataFrame(data, columns=feature_names)
        return new_data

    def predict(self, data, manual=False):
        """
        Predicts movies ratings based on the input data

        Parameters:
        - data: The input data for prediction
        - manual: A boolean indicating whether it's a manual prediction or not. Default is False

        Returns:
        - float or list: The predicted movie rating(s)
        """
        prediction = self.model.predict(self.preprocess_data(data, manual=manual))
        if manual:
            return round(prediction[0], 1)

        # If predicting from a dataset
        predictions = [round(pred, 1) for pred in prediction]
        return predictions

class CustomData:
    """ 
    A class representing custom datasets

    Attributes:
    - released_year: Year of release
    - certificate: Certificate earned by that movie regarding age rating
    - runtime: Runtime of the movie
    - genre: Genre of the movie
    - meta_score: Score earned by the movie
    - director: Director of the movie
    - star1: Star #1
    - star2: Star #2
    - star3: Star #3
    - star4: Star #4
    - no_of_votes: Total number of votes
    - gross: How much money the movie made
    """
    def __init__(self, released_year: int,
                    certificate: str,
                    runtime: int,
                    genre: str,
                    meta_score: float,
                    director: str,
                    star1: str,
                    star2: str,
                    star3: str,
                    star4: str,
                    no_of_votes: int,
                    gross: int):
        """
        Initializes the CustomData object with the provided attributes
        """
        self.released_year = released_year
        self.certificate = certificate
        self.runtime = runtime
        self.genre = genre
        self.meta_score = meta_score
        self.director = director
        self.star1 = star1
        self.star2 = star2
        self.star3 = star3
        self.star4 = star4
        self.no_of_votes = no_of_votes
        self.gross = gross

    def get_data_as_dataframe(self):
        """
        Converts the CustomData object into a pandas DataFrame

        Returns:
        - pd.DataFrame: The CustomData object as a DataFrame
        """
        custom_data_input_dict = {
            "released_year": [self.released_year],
            "certificate": [self.certificate],
            "runtime": [self.runtime],
            "genre": [[self.genre]],
            "meta_score": [self.meta_score],
            "director": [self.director],
            "star1": [self.star1],
            "star2": [self.star2],
            "star3": [self.star3],
            "star4": [self.star4],
            "no_of_votes": [self.no_of_votes],
            "gross": [self.gross]
        }
        return pd.DataFrame(custom_data_input_dict)