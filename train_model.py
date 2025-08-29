import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import optuna
import time
import warnings
import pickle
import os


os.makedirs('model_weights', exist_ok=True)
train_path = './data/train_data.csv'
test_path = './data/test_data.csv'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
data = pd.concat([train_data, test_data])
X, y = data.drop(columns=['Churn']), data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y)

def linear_model_objective(trial):
    '''Оптимизационная функция для линейных моделей'''

    model_type = trial.suggest_categorical('model_type', ['logistic_regression', 'svc'])
    C = trial.suggest_float('C', 0.1, 10)
    max_iter = trial.suggest_int('max_iter', 100, 500)
    if model_type == 'svc':
        model = LinearSVC(C=C, max_iter=max_iter)
    else:
        model = LogisticRegression(C=C, max_iter=max_iter)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return 1 - scores.mean()

def ensemble_model_objective(trial):
    '''Оптимизационная функция для ансамблевых моделей'''

    model_type = trial.suggest_categorical('model_type', ['random_forest', 'gradient_boosting'])
    max_depth = trial.suggest_int('max_depth', 2, 32)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    else:
        model = GradientBoostingClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return 1 - scores.mean()

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params, time_limit):
    """Функция для обучения и логирования одной модели"""
    
    mlflow.log_params(params)
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mlflow.log_metrics({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'training_time': train_time,
        'optuna_time': time_limit
    })
    mlflow.sklearn.log_model(model, artifact_path=f"model_{model_name}")
    mlflow.set_tag("model_type", model_name)
    mlflow.log_param('train_size', len(X_train))
    mlflow.log_param('test_size', len(X_test))
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    with open(f'model_weights/model_{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model, accuracy, f1

mlflow.set_experiment('bank churn')
start_time_linear = time.time()
study1 = optuna.create_study(direction='minimize')  # подбираем гиперпараметры для линейных моделей
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    study1.optimize(linear_model_objective, n_trials=20)
linear_time = time.time() - start_time_linear

lr_trials = [t for t in study1.trials if t.params.get('model_type') == 'logistic_regression']  # trials для логистической регрессии
svc_trials = [t for t in study1.trials if t.params.get('model_type') == 'svc']  # trials для метода опорных векторов

start_time_ensemble = time.time()
study2 = optuna.create_study(direction='minimize')  # подбираем гиперпараметры для ансамблевых моделей
study2.optimize(ensemble_model_objective, n_trials=10)
ensemble_time = time.time() - start_time_ensemble

rf_trials = [t for t in study2.trials if t.params.get('model_type') == 'random_forest']
gb_trials = [t for t in study2.trials if t.params.get('model_type') == 'gradient_boosting']

model_configs = [
    (lr_trials, LogisticRegression, "LogisticRegression", linear_time),
    (svc_trials, LinearSVC, "LinearSVC", linear_time),
    (rf_trials, RandomForestClassifier, "RandomForestClassifier", ensemble_time),
    (gb_trials, GradientBoostingClassifier, "GradientBoostingClassifier", ensemble_time)
]


for trials, model_class, model_name, time_limit in model_configs:
    if trials:
        best_trial = min(trials, key=lambda t: t.value)  
        best_params = best_trial.params.copy()  # лучшие параметры для модели
        best_params.pop('model_type', None)
        try:
            model = model_class(**best_params)
        except TypeError:
            model = model_class(**{**best_params, 'random_state': 42})
        with mlflow.start_run(run_name=model_name):
            train_and_log_model(  # логируем каждую модель
                model, model_name,
                X_train, X_test, y_train, y_test, 
                best_params, time_limit
            )