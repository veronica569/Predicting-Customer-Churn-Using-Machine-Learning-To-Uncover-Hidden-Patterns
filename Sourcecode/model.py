import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Example fake data
data = pd.DataFrame({
    'gender': ['Male', 'Female']*50,
    'SeniorCitizen': [0, 1]*50,
    'tenure': list(range(1, 101)),
    'MonthlyCharges': [70.5]*100,
    'TotalCharges': [2000.0]*100,
    'Churn': [0, 1]*50
})

# Encoding
encoders = {'gender': LabelEncoder()}
data['gender'] = encoders['gender'].fit_transform(data['gender'])

# Scaling
scaler = StandardScaler()
data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(data[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Train/test split
X = data.drop('Churn', axis=1)
y = data['Churn']
model = RandomForestClassifier()
model.fit(X, y)

# Save model, encoder, scaler
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoders, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("All files saved.")
