import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Spliting training and test data")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "KNeighbors": KNeighborsRegressor()
            }

            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                         models=models)

            logging.info("Model training completed")
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("Best model score is less than 0.6")
                raise CustomException("Best model score is less than 0.6", sys)


            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)

            logging.info(f"Best model r2 score: {r2}")
            return r2


        except Exception as e:
            logging.info("Exception occured in the initiate_model_trainer method")
            raise CustomException(e, sys)
