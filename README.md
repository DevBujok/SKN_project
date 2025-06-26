# Heart Disease Risk Prediction System

## Overview

This project aims to predict the risk of heart disease based on patient health data using machine learning techniques. It is being developed as part of academic work at the **Silesian University of Technology**.

The system:
- takes features such as **age**, **BMI**, **blood pressure**, **cholesterol levels**, and other health indicators,
- analyzes their influence on heart disease occurrence,
- identifies the most relevant predictors using feature selection and correlation analysis,
- compares the performance of multiple classification algorithms.

The project also includes a web-based frontend built with **Django**, allowing users to:
- input patient data through a form,
- receive predictions along with confidence scores,
- view key factors contributing to the model's decision.

## Repository

GitHub: [DevBujok/SKN_project](https://github.com/DevBujok/SKN_project)

## Technologies Used

- **Python** – core logic and model development
- **Pandas** – data preprocessing and manipulation
- **NumPy** – numerical operations
- **Scikit-learn** – machine learning models
- **Matplotlib / Seaborn** – data visualization
- **Django** – web application for interactive prediction interface
- **Joblib** – model serialization

## Functionality

- Preprocessing: categorical encoding, correlation filtering
- Training: logistic regression, ensemble models, and others
- Evaluation: accuracy, F1-score, confusion matrix
- Model versioning: saving/loading models with metadata (JSON)
- Web interface: simple form to test the trained models

## Folder Structure (simplified)

SKN_project/
├── trained_models/ # .pkl files and model metadata (JSON)
├── main.py # main execution script
├── app/ # Django frontend (planned)
├── notebooks/ # data analysis and experiments
└── README.md

csharp
Kopiuj
Edytuj

## Status

**In progress** – core model selection is complete, integration with Django frontend ongoing.

## Author

**Jakub Bujok**  
Silesian University of Technology  
[GitHub Profile](https://github.com/DevBujok)

## License

MIT License
