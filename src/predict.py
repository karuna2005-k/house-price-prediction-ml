import joblib
import numpy as np

# load the trained model
model = joblib.load("house_price_model.pkl")

print("Model loaded successfully")

# sample house data (38 features required by the model)
sample_house = np.array([
    [1, 528456, 20, 70, 8450, 7, 5, 2003, 2003, 856, 854, 0, 1710,
     1, 0, 2, 3, 2, 548, 2003, 0, 61, 0, 0, 0, 2, 2008,
     1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# predict price
prediction = model.predict(sample_house)

print("Predicted House Price:", prediction[0])