from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Cargar los datos desde el archivo CSV
data = pd.read_csv('House_Rent_Dataset.csv')

# Seleccionar las columnas de interés
selected_columns = ['BHK', 'Rent', 'Size', 'Bathroom']
data_selected = data[selected_columns]

# Dividir los datos en características (X) y etiquetas (y)
X = data_selected.drop('Rent', axis=1)  # características
y = data_selected['Rent']  # etiquetas

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Escalar los datos para normalizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear el modelo de regresión MLP con verbose=True
model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=5000, verbose=True)

# Entrenar el modelo
model.fit(X_train_scaled, y_train)

# Evaluar el modelo en el conjunto de prueba
score = model.score(X_test_scaled, y_test)
print(f'R^2 Score on Test Set: {score}')
