# Heart Disease Risk Prediction System

## Overview

This project will aim to predict the risk of heart disease based on patient health data using machine learning techniques. It will be developed as part of academic work at the **Silesian University of Technology**.

The system will:
- take features such as **age**, **BMI**, **blood pressure**, **cholesterol levels**, and other health indicators,
- analyze their influence on heart disease occurrence,
- identify the most relevant predictors using feature selection and correlation analysis,
- compare the performance of multiple classification algorithms.

The project will also include a web-based frontend built with **Django**, allowing users to:
- input patient data through a form,
- receive predictions along with confidence scores,
- view key factors contributing to the model's decision.

## Repository

GitHub: [DevBujok/SKN_project](https://github.com/DevBujok/SKN_project)

## Technologies Planned

- **Python** – core logic and model development
- **Pandas** – data preprocessing and manipulation
- **NumPy** – numerical operations
- **Scikit-learn** – machine learning models
- **Matplotlib / Seaborn** – data visualization
- **Django** – web application for interactive prediction interface
- **Joblib** – model serialization

## Planned Functionality

- Preprocessing: categorical encoding, correlation filtering
- Training: logistic regression, ensemble models, and others
- Evaluation: accuracy, F1-score, confusion matrix
- Model versioning: saving/loading models with metadata (JSON)
- Web interface: simple form to test the trained models

## Folder Structure (simplified)
<pre>
SKN_project/
├── trained_models/ # .pkl files and model metadata (JSON)
├── main.py # main execution script
├── app/ # Django frontend (planned)
├── notebooks/ # data analysis and experiments
└── README.md
</pre>


## Status

**In early development**

## Author

**Paweł Bujok**  
Silesian University of Technology  
[GitHub Profile](https://github.com/DevBujok)

## License

MIT License
