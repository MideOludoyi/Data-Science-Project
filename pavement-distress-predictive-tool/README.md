# Pavement Distress Predictive Tool

This repository contains a predictive tool designed to evaluate and forecast pavement distress using machine learning techniques. The tool processes raw data, performs feature engineering, trains a machine learning model, and evaluates its performance. 

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#data)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Pavement Distress Predictive Tool is aimed at providing an accurate prediction of Pavement Condition Index (PCI) based on various distress features such as area, length, severity, and distress density. This tool leverages a machine learning model to assist in pavement maintenance and planning.

## Dataset Summary

#### Summary
The City of Bloomington contracted with a firm to conduct a citywide survey of street pavement conditions. The data and associated infrastructure metrics was collected between January and March 2018 through field inspections utilizing the use of Light Detection and Ranging (LIDAR) technology. LIDAR is much more time efficient than visual data collection, allows for real-time data collection and provides more uniform and accurate reporting. The street pavement data that was collected updated the Pavement Condition Index (PCI) ratings for the City’s entire 234 miles of street network. Street Department staff uses PCI ratings to prepare targeted improvements during the development of the City's annual paving schedule. The City is also required to annually collect and report PCI rating data for its entire street network to the Indiana Department of Transportation. Visualizations: * Pavement Segment Viewer Documentation from this survey effort is listed below.

Source: https://data.bloomington.in.gov/dataset/2018-pavement-condition-report
Last updated at https://data.bloomington.in.gov/dataset : 2019-10-24

## Features

- **Data Loading and Cleaning**: Handles raw data input, cleansing, and preprocessing.
- **Feature Engineering**: Extracts and transforms features necessary for model training.
- **Model Training**: Trains a machine learning model to predict PCI.
- **Model Evaluation**: Provides metrics to evaluate the performance of the predictive model.
- **Visualization**: Generates plots to visualize the performance of the model.

## Requirements

To run the predictive tool, you'll need the following:

- Python 3.8 or later
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
## Installations
1. Clone the Repository:
```bash
git clone https://github.com/MideOludoyi/Data-Science-Project.git
```
2. Navigate to the Project Directory:
```bash
cd pavement-distress-predictive-tool
```
3. Install Dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Prepare Data:
- Ensure your data file pavement.csv is in the data directory.
2. Run the Main Script:
```bash
python main.py
```
The script performs the following tasks:
- Loads and cleans the data.
- Performs feature extraction and transformation.
- Scales the features and saves the processed data.
- Splits the data into training and testing sets.
- Trains a machine learning model and saves it.
- Evaluates the model's performance and generates plots.

3. Output:
- The cleaned data will be saved in output/cleaned_data.csv.
- The engineered features will be saved in output/engineered_features.csv.
- The trained model will be saved in output/pavement_distress_model.pkl.
- Evaluation metrics will be saved in output/evaluation_metrics.txt.
- Plots will be saved in the output/plots directory

## Project Structure
```bash
pavement-distress-predictive-tool/
│
├── data/
│   └── pavement.csv                # Raw data file
│
├── output/
│   ├── cleaned_data.csv            # Cleaned data
│   ├── engineered_features.csv     # Features after engineering
│   ├── pavement_distress_model.pkl # Trained model
│   ├── evaluation_metrics.txt      # Model evaluation metrics
│   └── plots/                      # Directory for plots
│
├── src/
│   ├── data_preprocessing.py       # Module for data loading and cleaning
│   ├── feature_engineering.py      # Module for feature extraction and transformation
│   ├── model_training.py           # Module for training and evaluating model
│   ├── model_evaluation.py         # Module for additional evaluation
│   └── utils.py                    # Utility functions
│
├── main.py                         # Main script to run the entire process
├── requirements.txt                # Required dependencies
└── README.md                       # This README file
```

## Contributing

Contributions are welcome! Please follow these steps:

- Fork the repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes.
- Commit your changes (git commit -m 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Create a new Pull Request.

## License

This project is licensed under the MIT License