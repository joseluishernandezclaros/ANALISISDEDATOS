import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score

# Cargar los datos desde el archivo CSV
df = pd.read_csv('House_Rent_Dataset.csv')

# Crear nueva columna 'Puede Arrendar' basada en alguna lógica
# Por ejemplo, aquí asumimos que puedes arrendar si el precio de renta es menor que tu presupuesto
presupuesto = float(input("Ingrese su presupuesto: "))
df['Puede Arrendar'] = (df['Rent'] <= presupuesto).astype(int)

# Codificar variables categóricas usando one-hot encoding
df = pd.get_dummies(df, columns=[
                    'Area Type', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact'])

# Seleccionar columnas relevantes
features = df[['BHK', 'Size', 'Bathroom', 'Area Type_Super Area', 'Area Type_Carpet Area', 'Furnishing Status_Unfurnished', 'Furnishing Status_Semi-Furnished',
               'Furnishing Status_Furnished', 'Tenant Preferred_Bachelors/Family', 'Tenant Preferred_Bachelors', 'Tenant Preferred_Family', 'Point of Contact_Contact Owner', 'Point of Contact_Contact Agent']
              ]

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Seleccionar la columna 'Puede Arrendar' como objetivo
targets = df['Puede Arrendar']

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, targets, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
# Usar activación sigmoide para clasificación binaria
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_data=(X_test, y_test))

# Ahora puedes ingresar datos por teclado y predecir si puedes arrendar o no
nuevos_datos = pd.DataFrame(columns=[
    'BHK', 'Size', 'Bathroom', 'Area Type_Super Area', 'Area Type_Carpet Area', 'Furnishing Status_Unfurnished',
    'Furnishing Status_Semi-Furnished', 'Furnishing Status_Furnished', 'Tenant Preferred_Bachelors/Family',
    'Tenant Preferred_Bachelors', 'Tenant Preferred_Family', 'Point of Contact_Contact Owner', 'Point of Contact_Contact Agent'
])

# Solicitar al usuario que ingrese los valores por teclado
nuevos_datos.loc[0] = [float(input("Ingrese cantidad de habitaciones: ")), float(input("Ingrese tamaño de vivienda: ")),
                       float(input("Ingrese cantidad de baños: ")), 1, 0, 1, 0, 0, 1, 0, 0, 1, 1]

# Normalizar los nuevos datos
nuevos_datos_scaled = scaler.transform(nuevos_datos)

# Realizar la predicción de si puedes arrendar o no
prediccion = model.predict(nuevos_datos_scaled)
if prediccion[0, 0] >= 0.5:
    print("Puedes arrendar la casa.")
else:
    print("No puedes arrendar la casa.")

# Calcular la precisión y la sensibilidad en el conjunto de prueba
y_pred_test = model.predict(X_test)
y_pred_test_binary = (y_pred_test >= 0.5).astype(int)

accuracy_test = accuracy_score(y_test, y_pred_test_binary)
recall_test = recall_score(y_test, y_pred_test_binary)

print(f"Precisión en el conjunto de prueba: {accuracy_test}")
print(f"Sensibilidad en el conjunto de prueba: {recall_test}")

input("Presiona Enter para salir...")
