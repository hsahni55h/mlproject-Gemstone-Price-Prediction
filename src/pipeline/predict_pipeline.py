import sys
import pandas as pd
from src.exception import Custom_Exception
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    """
    Class for handling the prediction pipeline process.
    """
    def __init__(self):
        pass

    def predict(self, features):
        """
        Predict the target using the loaded model and preprocessor.

        Args:
        features (pd.DataFrame): DataFrame containing the input features for prediction.

        Returns:
        np.ndarray: Predicted values.

        Raises:
        CustomException: If any exception occurs during prediction.
        """
        try:
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'

            # Load the preprocessor object
            preprocessor = load_object(file_path=preprocessor_path)
            # Load the trained model
            model = load_object(file_path=model_path)

            # Preprocess the input features
            data_scaled = preprocessor.transform(features)
            # Predict using the preprocessed features
            pred = model.predict(data_scaled)

            return pred
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise Custom_Exception(e, sys)

class CustomData:
    """
    Class for handling custom input data.
    """
    def __init__(self,
                 carat: float,
                 depth: float,
                 table: float,
                 x: float,
                 y: float,
                 z: float,
                 cut: str,
                 color: str,
                 clarity: str):
        """
        Initializes the CustomData instance with input features.

        Args:
        carat (float): Carat weight of the diamond.
        depth (float): Total depth percentage.
        table (float): Width of the top of the diamond relative to the widest point.
        x (float): Length of the diamond.
        y (float): Width of the diamond.
        z (float): Depth of the diamond.
        cut (str): Cut quality of the diamond.
        color (str): Color grade of the diamond.
        clarity (str): Clarity grade of the diamond.
        """
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        """
        Convert the custom input data to a DataFrame.

        Returns:
        pd.DataFrame: DataFrame containing the input features.

        Raises:
        CustomException: If any exception occurs during the DataFrame creation.
        """
        try:
            custom_data_input_dict = {
                'carat': [self.carat],
                'depth': [self.depth],
                'table': [self.table],
                'x': [self.x],
                'y': [self.y],
                'z': [self.z],
                'cut': [self.cut],
                'color': [self.color],
                'clarity': [self.clarity]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Exception occurred in prediction pipeline')
            raise Custom_Exception(e, sys)
