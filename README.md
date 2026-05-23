# Data Science Internship - Predictive & Classification Models

This repository contains an end-to-end suite of Machine Learning and Data Science projects developed during my Data Science Internship. The tasks span regression pipelines, multiclass classification frameworks, and natural language processing wrappers, emphasizing robust model training, evaluations, modular code design, and model serialization for deployment.

---

## 📂 Project Repository Structure

```text
├── 01_Spam_Classifier/
│   ├── spam.csv                        # Raw text messages SMS dataset
│   ├── spam_detector_model.pkl         # Trained Multinomial Naive Bayes / Logistic model
│   ├── tfidf_vectorizer.pkl            # Saved state TF-IDF text vectorizer 
│   └── app.py                          # Predictor framework script
│
├── 02_Sales_Forecasting/
│   ├── Advertising.csv                 # Marketing spend and media data
│   ├── sales_rf_model.pkl              # Optimal trained Random Forest Regressor 
│   ├── sales_lr_model.pkl              # Benchmark Linear Regression model
│   └── test_sales_model.py             # Scenario-driven business testing pipeline
│
├── 03_Car_Price_Estimator/
│   ├── car data.csv                    # Used vehicle parameters and historical prices
│   ├── car_price_model.pkl             # Trained multi-variable Linear Regression model
│   └── test_car_model.py               # Boundary testing validation script
│
├── 04_Iris_Species_Classification/
│   ├── Iris.csv                        # Morphological flower dimension dataset
│   ├── iris_classification_model.pkl   # Serialized optimal Multiclass Classifier
│   └── test_iris_model.py              # Operational inference and probability tester
│
└── README.md                           # Main repository documentation
