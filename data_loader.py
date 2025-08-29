import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN


data = pd.read_csv('customer_churn.csv')
data.drop(columns=['customerID'], inplace=True)
categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())

X, y = data.drop(columns=['Churn']), data['Churn']
y = y.map({'No': 0, 'Yes': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train[numerical_cols])
X_test_scaled = ss.transform(X_test[numerical_cols])

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = ohe.fit_transform(X_train[categorical_cols])
X_test_cat = ohe.transform(X_test[categorical_cols])

X_train_final = pd.concat([
    pd.DataFrame(X_train_scaled, columns=numerical_cols, index=X_train.index),
    pd.DataFrame(X_train_cat, columns=ohe.get_feature_names_out(categorical_cols), index=X_train.index)
], axis=1)
y_train = pd.DataFrame(y_train, columns=['Churn'])

sme = SMOTEENN(random_state=42)

X_test_final = pd.concat([
    pd.DataFrame(X_test_scaled, columns=numerical_cols, index=X_test.index),
    pd.DataFrame(X_test_cat, columns=ohe.get_feature_names_out(categorical_cols), index=X_test.index)
], axis=1)
y_test = pd.DataFrame(y_test, columns=['Churn'])

X_proc = pd.concat([X_train_final, X_test_final])
y_proc = pd.concat([y_train, y_test])
X_res, y_res = sme.fit_resample(X_proc, y_proc)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, stratify=y_res, random_state=42)

train_final = pd.concat([X_train, y_train], axis=1)
test_final = pd.concat([X_test, y_test], axis=1)
train_final.to_csv('./data/train_data.csv')
test_final.to_csv('./data/test_data.csv')