import pandas as pd
import mlflow
import mlflow.pyfunc
from app.log_utils import append_prediction_log

class LoggedModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.base_model = mlflow.sklearn.load_model(context.artifacts["base_model"])

    def predict(self, context, model_input: pd.DataFrame):
        y_pred = self.base_model.predict(model_input)
        append_prediction_log(model_input, y_pred, req_meta=None)
        return y_pred