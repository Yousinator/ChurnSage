from .data_preparer import DataPreparer


class ChurnPredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        data_preparer = DataPreparer(input_data)
        prepared_data = data_preparer.prepare_input_data()

        prediction = self.model.predict(prepared_data)
        return prediction
