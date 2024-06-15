from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle


class DataPreparer:
    def __init__(self, input_data):
        """
        Initialize the DataPreparer object with a pandas DataFrame.

        Parameters:
        data (pandas.DataFrame): The DataFrame to be prepared.
        """
        self.input_data = input_data
        self.data_columns = input_data.columns

    def prepare_input_data(self):
        self.input_data = pd.DataFrame(self.input_data)
        # self.label_encode_columns()
        self.map_encode_columns()
        self.scale_data()

        return self.input_data

    def label_encode_columns(self):

        le_pos = self.load_from_pickle(r"variables/le_pos.pkl")
        self.input_data["POS_MSISDN"] = le_pos.transform(self.input_data["POS_MSISDN"])

        le_distributer = self.load_from_pickle(r"variables/le_distributer.pkl")
        self.input_data["Distributer"] = le_distributer.transform(
            self.input_data["Distributer"]
        )

        le_plan = self.load_from_pickle(r"variables/le_plan.pkl")
        self.input_data["Plan Name"] = le_plan.transform(self.input_data["Plan Name"])

        le_reason = self.load_from_pickle(r"variables/le_reason.pkl")
        self.input_data["Status Reason"] = le_reason.fit_transform(
            self.input_data["Status Reason"]
        )

    def map_encode_columns(self):
        self.input_data["Tenure Category"] = self.input_data["Tenure Category"].map(
            {"Short-term": 0, "Medium-term": 1, "Long-term": 2}
        )

        self.input_data["Status"] = self.input_data["Status"].map(
            {"Hard Suspended": 0, "Soft Suspended": 1, "Deactive": 2, "Active": 3}
        )

        self.input_data["Segment1"] = self.input_data["Segment1"].map(
            {"Prepaid": 0, "Postpaid": 1}
        )

        self.input_data["Segment2"] = self.input_data["Segment2"].map(
            {"Residential": 0, "Corporate": 1, "PRO": 2}
        )

    def scale_data(self):

        scaler = self.load_from_pickle(r"variables/scaler.pkl")
        self.input_data = scaler.transform(self.input_data.to_numpy())

        self.input_data = pd.DataFrame(self.input_data, columns=self.data_columns)

    def load_from_pickle(self, path):
        with open(path, "rb") as file:
            variable = pickle.load(file)

        return variable
