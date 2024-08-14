---
title: HeartR
emoji: 🏢
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Here is a possible README file for your repository:

# HeartDiseasePredictor

A Machine Learning Tool for Early Intervention

## Overview

Heart disease is a leading cause of death worldwide, and accurate prediction of heart disease remains a significant challenge. This project aims to develop a machine learning model capable of predicting heart disease using a comprehensive dataset of key indicators.

## Dataset

The dataset used in this project is the "Personal Key Indicators of Heart Disease" dataset from Kaggle. The dataset contains over 400,000 adult profiles, capturing the diverse health status of individuals across various demographics and risk factors.

## Methodology

The project follows the following methodology:

1. Exploratory Data Analysis (EDA): Summarize main characteristics of the dataset using various visualization techniques and statistical methods.
2. Data Preprocessing: Apply PCA to the training features, normalize categorical labels, and shuffle the dataset to increase randomness and reduce bias.
3. Model Evaluation: Use classification report from Scikit-learn to evaluate the performance of the trained model.
4. Hyperparameter Tuning: Use BayesSearchCV and RandomizedSearchCV to fine-tune the hyperparameters of the model.
5. Prediction: Predict on new, unseen patient data from a random sample from the test set, and identify high-risk individuals for heart disease.

## Models

The project uses the following models:

1. DecisionTreeClassifier
2. RandomForestClassifier
3. ExtraTreesClassifier
4. GradientBoostingClassifier
5. HistGradientBoostingClassifier
6. XGBClassifier
7. LGBMClassifier
8. CatBoostClassifier
9. SVC
10. LogisticRegression
11. MLPClassifier
12. AdaBoostClassifier
13. GaussianNB

## Results

The project achieves a test accuracy of 94.66% using the MLPClassifier model. The best hyperparameters for the RandomForestClassifier and XGBClassifier models are also identified using BayesSearchCV and RandomizedSearchCV.

## Requirements

To run the project, you will need to install the following libraries:

* pandas
* numpy
* scikit-learn
* tensorflow
* xgboost
* lightgbm
* catboost
* scikit-optimize

You can install these libraries using pip:

```
pip install pandas numpy scikit-learn tensorflow xgboost lightgbm catboost scikit-optimize
```

## Usage

To run the project, simply execute the Python script:

```
python heart_disease_predictor.py
```

This will train the models, evaluate their performance, and predict on new, unseen patient data.

## License

The project is licensed under the MIT License.

## Acknowledgments

The project uses the "Personal Key Indicators of Heart Disease" dataset from Kaggle, which is licensed under the CC0-1.0 license.

I hope this helps! Let me know if you have any further requests.

Here's an updated version of the README with a more conversational tone, additional content, and visual appeal:

# HeartDiseasePredictor 💖

A Machine Learning Tool for Early Intervention 🚑

**Welcome! 👋**

Heart disease is a leading cause of death worldwide, and accurate prediction of heart disease remains a significant challenge. This project aims to develop a machine learning model capable of predicting heart disease using a comprehensive dataset of key indicators.

**What's in this project? 🤔**

* A dataset of over 400,000 adult profiles, capturing the diverse health status of individuals across various demographics and risk factors
* A range of machine learning models, including Decision Trees, Random Forests, Gradient Boosting, and more
* Hyperparameter tuning using BayesSearchCV and RandomizedSearchCV
* Model evaluation using classification reports and accuracy scores
* Prediction on new, unseen patient data from a random sample from the test set

**Getting started 🚀**

To run the project, you'll need to install the following libraries:

* pandas
* numpy
* scikit-learn
* tensorflow
* xgboost
* lightgbm
* catboost
* scikit-optimize

You can install these libraries using pip:

```
pip install pandas numpy scikit-learn tensorflow xgboost lightgbm catboost scikit-optimize
```

**How it works 🔍**

1. **Data Preprocessing**: We apply PCA to the training features, normalize categorical labels, and shuffle the dataset to increase randomness and reduce bias.
2. **Model Evaluation**: We use classification reports from Scikit-learn to evaluate the performance of the trained model.
3. **Hyperparameter Tuning**: We use BayesSearchCV and RandomizedSearchCV to fine-tune the hyperparameters of the model.
4. **Prediction**: We predict on new, unseen patient data from a random sample from the test set, and identify high-risk individuals for heart disease.

**Models 🤖**

We've implemented the following models:

1. DecisionTreeClassifier
2. RandomForestClassifier
3. ExtraTreesClassifier
4. GradientBoostingClassifier
5. HistGradientBoostingClassifier
6. XGBClassifier
7. LGBMClassifier
8. CatBoostClassifier
9. SVC
10. LogisticRegression
11. MLPClassifier
12. AdaBoostClassifier
13. GaussianNB

**Results 📊**

Our best-performing model achieves a test accuracy of 94.66% using the MLPClassifier model. We've also identified the best hyperparameters for the RandomForestClassifier and XGBClassifier models using BayesSearchCV and RandomizedSearchCV.

**License 📜**

This project is licensed under the MIT License.

**Acknowledgments 🙏**

We'd like to thank the creators of the "Personal Key Indicators of Heart Disease" dataset from Kaggle, which is licensed under the CC0-1.0 license.

**Contributing 🤝**

We welcome contributions to this project! If you'd like to contribute, please fork the repository and submit a pull request.

**Stay in touch 📱**

Follow us on Twitter for updates on this project and other machine learning endeavors: [Your Twitter handle]

I hope this updated version meets your requirements! Let me know if you have any further requests. 😊
