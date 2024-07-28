
# Part 1: Machine Learning Skills

This repository contains the first part of the Machine Learning skills assessment for Konecta. It includes various scripts and functions for traditional ML tasks such as data preprocessing, model training, and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methods](#methods)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates basic Machine Learning skills, focusing on data preprocessing, model training, and inference using Python and popular ML libraries like pandas and scikit-learn.

## Installation

To use this project, clone the repository and install the necessary dependencies:

\`\`\`bash
git clone https://github.com/DANN9907/Konecta.git
cd Konecta/Part1%20ML%20skills
pip install -r requirements.txt
\`\`\`

## Usage

1. **Open CSV Files**: Load your training and inference data.
   \`\`\`python
   ml = ML_skills()
   data, inference = ml.open_csv('path_to_training_data.csv', 'path_to_inference_data.csv')
   \`\`\`

2. **Data Treatment**: Preprocess the data using various methods.
   \`\`\`python
   data = ml.data_treatment(data, 'encoding')
   data = ml.data_treatment(data, 'drop_c', ['id', 'CustomerId', 'Surname', 'Geography'])
   \`\`\`

3. **Train Model**: Train the model using the preprocessed data.
   \`\`\`python
   X_train, X_val, y_train, y_val = ml.train_test_split(data)
   ml.train_model(X_train, y_train)
   \`\`\`

4. **Inference**: Make predictions on new data and add predictions as a new column.
   \`\`\`python
   inference_with_predictions = ml.predict_and_add_column(inference, 'Predictions')
   print(inference_with_predictions.head())
   \`\`\`

## Project Structure

\`\`\`
.
├── data/                    # Folder containing the data files
├── scripts/                 # Folder containing the Python scripts
│   ├── part1_traditional_ML_skills.py  # Main script with ML skills
│   └── ...                  # Other scripts
├── requirements.txt         # Python packages required
├── README.md                # This README file
└── ...                      # Other project files
\`\`\`

## Methods

### `ML_skills`

- `open_csv(path_df: str, path_inference: str) -> tuple[pd.DataFrame, pd.DataFrame]`
  - Opens CSV files for training and inference data.

- `data_treatment(data: pd.DataFrame, method: str, drop_c: list = None) -> pd.DataFrame`
  - Cleans and preprocesses the data based on the specified method.

- `train_test_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]`
  - Splits the data into training and testing sets.

- `train_model(X_train: pd.DataFrame, y_train: pd.Series)`
  - Trains a Random Forest classifier.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
