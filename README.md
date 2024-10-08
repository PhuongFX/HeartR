# Heart Disease Predictor 💖

A Machine Learning Tool for Early Intervention 

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://github.com/PhuongFX/ButterFlySpace/blob/main/LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-gr.svg)](https://scikit-learn.org/stable/)
[![Pandas](https://img.shields.io/badge/pandas-1.3.5-red.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.21.4-purple.svg)](https://numpy.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-green.svg)](https://www.tensorflow.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-0.11.2-pink.svg)](https://seaborn.pydata.org/)
[![Dataset](https://img.shields.io/badge/Dataset-📊-red.svg)](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-green.svg)](https://github.com/PhuongFX/HeartR)



## 🫀`About`
> Heart disease is a leading cause of death worldwide, and accurate prediction of heart disease remains a significant challenge.
> > This project aims to develop a machine learning model capable of predicting heart disease using a comprehensive dataset of key indicators.


## `What's in this project?` 🫶

* A dataset of over 400,000 adult profiles, capturing the diverse health status of individuals across various demographics and risk factors
* A range of machine learning models, including Decision Trees, Random Forests, Gradient Boosting, and more
* Hyperparameter tuning using BayesSearchCV and RandomizedSearchCV
* Model evaluation using classification reports and accuracy scores
* Prediction on new, unseen patient data from a random sample from the test set


## `Dataset` 📊

* **Dataset URL:** [💖 Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease)
* **License:** CC0-1.0
* **Number of samples:** 400,000
* **Number of factors:** 40

| Category | Number of Images |
| --- | --- |
| Training | 12594 |
| Validation | 500 |
| Testing | 500 |

## `Methodology` 🔍

> ### Requirements

* Python 3.x
* Xgboost
* Keras
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Plotly

> ### Data Preprocessing 🔀

* Data Scaling: Appling PCA to the training features, normalize categorical labels, and shuffle the dataset to increase randomness and reduce bias. 🔀

> ### Models 🤖

The following models are implemented and compared:

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



> ### Model Performance 📊

The model achieves a test accuracy of 94.66% using the MLPClassifier model, which is a great result considering the complexity of the dataset! 🎉 
I have also identified the best hyperparameters for the RandomForestClassifier and XGBClassifier models using BayesSearchCV and RandomizedSearchCV.

* Training accuracy: 0.9996
* Validation accuracy: 0.9420
* Test accuracy: 0.9600

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| **Actual Positive** | 232 | 12 |
| **Actual Negative** | 15 | 213 |

> ### Hyperparameter Tuning 🔧

* GridSearchCV 
* RandomizedSearchCV 


## `Acknowledgments` 🙏

* Kaggle dataset: 💖 Indicators of Heart Disease (2022 UPDATE)
* Scikit-learn and Xgboost libraries for model training
* Matplotlib and Seaborn libraries for data visualization

## `🙅‍♂️Disclaimer`

> This project is licensed under AGPL-3.0 License and is for personal use only and should not be used for commercial purposes.
The pre-trained model and may not always produce accurate results.


## `Get Involved!` 😌
This project demonstrates the potential of machine learning for heart disease prediction. 
The model achieves high accuracy and can be used as a starting point for further research and development in this field. 


I hope you found this project informative and engaging! 😊  
If you're interested in collaborating and contributing to the project, please let me know! I'd love to hear from you.
* [Follow me on GitHub](https://github.com/PhuongFX)
* [Follow me on Hugging Face](https://huggingface.co/PhuongFX)

## `Getting Started` 🚀

To get started with this project, you'll need to:

* Install the required libraries, including pandas, numpy, scikit-learn, xgboost, catboost, lightgbm `pip install pandas numpy scikit-learn tensorflow xgboost lightgbm catboost` 📦
* Download the dataset from Kaggle 📈
* Run the code to train and evaluate the model 🤖

Enjoy working with the content! 😊
