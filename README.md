# ChurnSage: Advanced Customer Churn Prediction

ChurnSage is an AI-powered tool designed to predict customer churn with high accuracy. Leveraging a real dataset, the project involves extensive data cleaning, preprocessing, and model development using Python and [PrismaML](https://github.com/Yousinator/PrismaML). The best performing model is deployed through an interactive Gradio interface, making it accessible and user-friendly.

## Project Structure

```bash
ChurnSage
│
├── churn_predictor
│   ├── __init__.py
│   ├── churn_predictor.py
│   └── data_preparer.py
│
├── data
│   ├── actiavation_Aug_Oct2023.csv
│   ├── clean_churn_data.csv
│   └── processed_data.csv
│
├── flagged
│
├── models
│   └── knn_model.pkl
│
├── notebooks
│   ├── data_cleaning.ipynb
│   ├── model_preprocessing.ipynb
│   └── modeling.ipynb
│
├── variables
│   ├── le_distributer.pkl
│   ├── le_plan.pkl
│   ├── le_pos.pkl
│   ├── le_reason.pkl
│   └── scaler.pkl
│
├── app.py
├── LICENSE
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Notebooks Summary

### Data Cleaning (`notebooks/data_cleaning.ipynb`)

The `data_cleaning.ipynb` notebook involves the following steps:

#### EDA (Exploratory Data Analysis)

- Utilized the `PrismaML DatasetInformation` class to generate a comprehensive summary of the dataset, including basic statistics and data types.
- Used `PrismaML DatasetInformation` class to analyze categorical variables.
- Generated visualizations with the `PrismaML Plotting` class to better understand the distribution and relationships of categorical data.
- Analyzed numerical variables using the `PrismaML DatasetInformation` class.
- Created plots using the `PrismaML Plotting` class to explore distributions and identify potential outliers.

#### Data Cleaning

1. **Data Suitability**: Assessed the suitability of the data types with the data in the columns, identifying any issues that needed addressing.

2. **Dropping Duplicates**: Removed duplicate rows to ensure the dataset's integrity.

3. **Changing Column Names**: Standardized column names for consistency and clarity.

4. **Column Value Cleaning**: Cleaned and standardized column values, for example, changing `tenure` values to a consistent format:

```python
tenure = {"Short": "Short-term", "Medium": "Medium-term", "Long": "Long-term"}
```

5. **Filling Missing Data**: Addressed missing data using multiple strategies:

- **Group By Mean/Median/Mode:** Imputed missing values based on grouped statistics.
- **Using Data from Other Columns:** Leveraged information from other columns for imputation.
- **Using Machine Learning:** Applied machine learning techniques to predict and fill missing values.

6. **Removing Unneeded Columns for Modeling**: Dropped columns that were deemed unnecessary for the modeling process to streamline the dataset.

### Model Preprocessing (`notebooks/model_preprocessing.ipynb`)

The `model_preprocessing.ipynb` notebook involves the following steps:

#### Encoding

- **Sklearn Label Encoder:**

  - Utilized `sklearn`'s `LabelEncoder` to convert categorical variables into numerical format.
  - Applied label encoding to columns with categorical data to facilitate model training.

- **Manual Label Encoding:**
  - Performed manual label encoding for specific columns that required custom encoding logic.
  - Mapped categorical values to numerical codes for consistency.

#### Scaling

- **MinMaxScaler:**
  - Applied `MinMaxScaler` from `sklearn` to scale numerical features.
  - Transformed data to a range of [0, 1] to normalize the feature values and improve model performance.

### Modeling (`notebooks/modeling.ipynb`)

#### Select KBest

The section involves using the KBest SkLearn algorithm with KNN, RandomForest, ans SVM for building the models

##### Selecting the Features

1. **PrismaML.MachineLearning.select_best_features()**: This method selects the best features from the dataset based on their importance. It helps in reducing the dimensionality of the dataset by keeping only the most relevant features for the model.
2. **PrismaML.MachineLearning.plot_accuracy_vs_features()**: This method plots the model's accuracy against the number of selected features. It helps in visualizing the impact of different numbers of features on the model's performance.

##### Building the Model

1. **PrismaML.MachineLearning.evaluate_model()**: This method evaluates the model's performance using the selected features. It involves training the model, making predictions, and calculating performance metrics such as accuracy, precision, recall, and F1-score.

##### Comparing the Models

1. **PrismaML.Plotting.plot_algorithm_comparison()**: This method plots a comparison of the different models (KNN, Random Forest, SVM) based on their performance metrics. It helps in visualizing which model performs the best and under what conditions.

#### Without Feature Selection

In this section, we evaluate the models without performing feature selection. This helps in comparing the performance of models with and without feature selection.

##### Building the Model

- **PrismaML.MachineLearning.evaluate_model()**: This method evaluates the KNN model's performance using all available features without any feature selection.

##### Comparing the Models

1. **PrismaML.Plotting.plot_algorithm_comparison()**: This method plots a comparison of the different models (KNN, Random Forest, SVM) based on their performance metrics. It helps in visualizing which model performs the best and under what conditions.

## Gradio Interface Deployment

### Overview

This section describes the setup and deployment of a Gradio interface for predicting customer churn using a trained KNN model. The interface collects user inputs and provides a prediction on whether a customer is likely to churn.

### Files

1. `app.py`
2. `churn_predictor.py`
3. `DataPreparer.py`

### app.py

This script sets up a Gradio interface to interact with the churn prediction model.

1. **Model Loading**: The script loads a pre-trained KNN model from a pickle file.
2. **Prediction Function**: model_prediction function takes in several inputs such as tenure, tenure_category, segment1, segment2, status, loyalty_points, and data_usage_tier, and processes them into a DataFrame.
3. **Gradio Interface**: The interface collects user inputs and maps them to the prediction function, then displays the prediction result.

#### DataPreparer.py

This script prepares the input data for the model by encoding categorical variables and scaling numerical variables.

1. **Initialization**: The class is initialized with a pandas DataFrame containing the input data.
2. **Data Preparation**: The prepare_input_data method encodes categorical variables using map_encode_columns and scales numerical data using scale_data.
3. **Label Encoding**: Encodes categorical columns by mapping string values to numerical values.
4. **Scaling**: Scales the input data using a MinMaxScaler loaded from a pickle file.
5. **Utility Function**: The load_from_pickle method is used to load pickled objects.

## Usage

To run the project, follow these steps:

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/ChurnSage.git
cd ChurnSage
```

2. **Install dependencies**:

```bash
poetry install
```

3. Run the Jupyter notebooks in the notebooks directory to reproduce the data cleaning, preprocessing, and modeling steps.

4. **Launch the Gradio interface**:

```bash
python app.py
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
