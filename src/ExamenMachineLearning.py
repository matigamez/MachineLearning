import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los archivos CSV
file_red = './data/winequality-red.csv' 
file_white = './data/winequality-white.csv' 

# Leer los datasets
red_wine = pd.read_csv(file_red, sep=';')  
white_wine = pd.read_csv(file_white, sep=';')  

# 2. Agregar una nueva columna "tipo" a cada dataset
red_wine['tipo'] = 'red'  # Asignar 'red' para identificar vino tinto
white_wine['tipo'] = 'white'  # Asignar 'white' para identificar vino blanco

# 3. Combinar ambos datasets
wine_combined = pd.concat([red_wine, white_wine], ignore_index=True)  # Combinar en un único dataset

# Paso 1: Identificación y eliminación de valores duplicados
wine_combined = wine_combined.drop_duplicates()

# Paso 2: Manejo de valores no numéricos en columnas numéricas
numeric_columns = wine_combined.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    wine_combined[col] = pd.to_numeric(wine_combined[col], errors='coerce')  # Convertir a numérico, reemplazando strings con NaN

# Paso 3: Manejo de valores faltantes en columnas numéricas
imputer_numeric = SimpleImputer(strategy='median')  # Usar la mediana para imputar valores
wine_combined[numeric_columns] = imputer_numeric.fit_transform(wine_combined[numeric_columns])

# Paso 4: Verificación de columnas categóricas
categorical_columns = wine_combined.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    if col != 'tipo':  
        encoder = LabelEncoder()
        wine_combined[col] = encoder.fit_transform(wine_combined[col].astype(str))  # Convertir valores categóricos a numéricos
    else:
        # Asegurar que 'tipo' esté bien categorizado
        wine_combined['tipo'] = wine_combined['tipo'].astype('category')

# Paso 5: Verificar y ajustar tipos de datos finales
print(wine_combined.dtypes)

# Paso 6: Preparación de los datos para los modelos
X = wine_combined.drop(['quality', 'tipo'], axis=1)  # Excluir 'quality' y 'tipo' de las características
y = wine_combined['quality']  # 'quality' como variable objetivo

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características numéricas (solo las numéricas)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos

## 1. Regresión Logística
log_reg_model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)
log_reg_model.fit(X_train_scaled, y_train)

# Predicciones
y_pred_log_reg = log_reg_model.predict(X_test_scaled)

# Evaluación del modelo de regresión logística
print("\nEvaluación del modelo de Regresión Logística:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg, average='weighted', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_log_reg, average='weighted', zero_division=0):.4f}")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_log_reg))

## 2. K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Predicciones
y_pred_knn = knn_model.predict(X_test_scaled)

# Evaluación del modelo KNN
print("\nEvaluación del modelo K-Nearest Neighbors:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_knn, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_knn, average='weighted', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_knn, average='weighted', zero_division=0):.4f}")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_knn))

# Comparación de rendimiento
if accuracy_score(y_test, y_pred_log_reg) > accuracy_score(y_test, y_pred_knn):
    print("El modelo de Regresión Logística es más adecuado debido a su mayor Accuracy.")
    print(f"Accuracy de Regresión Logística: {accuracy_score(y_test, y_pred_log_reg):.4f}")
    print(f"Accuracy de KNN: {accuracy_score(y_test, y_pred_knn):.4f}")
else:
    print("El modelo de KNN es más adecuado debido a su mayor F1-Score.")
    print(f"F1-Score de Regresión Logística: {f1_score(y_test, y_pred_log_reg, average='weighted'):.4f}")
    print(f"F1-Score de KNN: {f1_score(y_test, y_pred_knn, average='weighted'):.4f}")

# Visualización de la matriz de confusión
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Matriz de Confusión - Regresión Logística')
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Matriz de Confusión - K-Nearest Neighbors')
plt.tight_layout()
plt.show()

# Exploración de Datos

# Visualizaciones univariadas y multivariadas

# a. Histogramas para las variables numéricas
numeric_columns = wine_combined.select_dtypes(include=['float64']).columns
wine_combined[numeric_columns].hist(bins=20, figsize=(14, 12), color='skyblue', edgecolor='black')
plt.suptitle('Histogramas de las columnas numéricas', fontsize=16)
plt.tight_layout()
plt.show()

# Gráfico de barras para la variable categórica 'tipo' (red/white wine)
sns.countplot(data=wine_combined, x='tipo', hue='tipo', palette='viridis', legend=False)
plt.title('Distribución de los tipos de vino')
plt.show()


# c. Diagrama de dispersión entre dos variables numéricas (por ejemplo, 'alcohol' y 'quality')
sns.scatterplot(data=wine_combined, x='alcohol', y='quality', hue='tipo', palette='coolwarm', alpha=0.7)
plt.title('Relación entre Alcohol y Calidad')
plt.show()

# d. Mapa de calor de las correlaciones entre variables numéricas
correlation_matrix = wine_combined[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Mapa de Calor de Correlaciones')
plt.show()

# Estadísticas descriptivas

# Medidas de tendencia central y dispersión
stats = wine_combined[numeric_columns].describe().T  # Estadísticas básicas
stats['range'] = stats['max'] - stats['min']  # Rango: max - min
stats['mode'] = wine_combined[numeric_columns].mode().iloc[0]  # Moda (solo la primera moda)
print("\nEstadísticas Descriptivas:")
print(stats)
