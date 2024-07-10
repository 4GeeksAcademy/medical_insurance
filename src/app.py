from utils import db_connect
engine = db_connect()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Paso 1: Carga del conjunto de datos
url = 'https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv'
data = pd.read_csv(url)

# Comprender el dataset
print(data.head())
print(data.info())
print(data.describe())

# Paso 2: Análisis Exploratorio de Datos (EDA)

# Distribución de las variables numéricas
data.hist(bins=30, figsize=(10, 7), color='skyblue')
plt.tight_layout()
plt.show()

# Análisis de correlación solo con variables numéricas
numeric_data = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
plt.title('Matriz de Correlación')
plt.show()

# Boxplot para las variables categóricas
plt.figure(figsize=(8, 6))
sns.boxplot(x='sex', y='charges', hue='sex', data=data, palette='Set2', dodge=False)
plt.title('Distribución de Charges por Sexo')
plt.legend([], [], frameon=False)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', hue='smoker', data=data, palette='Set3', dodge=False)
plt.title('Distribución de Charges por Smoker')
plt.legend([], [], frameon=False)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='region', y='charges', hue='region', data=data, palette='Set1', dodge=False)
plt.title('Distribución de Charges por Región')
plt.legend([], [], frameon=False)
plt.show()

# Paso 3: Construcción del modelo de regresión lineal

# Separar variables independientes y dependientes
X = data.drop('charges', axis=1)
y = data['charges']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento de variables categóricas
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

# Transformador para las características categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numeric_features)
    ])

# Construir el pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir con el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R2 Score: {r2}')

# Paso 4: Optimización del modelo

# Optimización con Ridge
ridge_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)

print(f'Ridge MSE: {ridge_mse}')
print(f'Ridge R2 Score: {ridge_r2}')

# Optimización con Lasso
lasso_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso())
])

lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

print(f'Lasso MSE: {lasso_mse}')
print(f'Lasso R2 Score: {lasso_r2}')

