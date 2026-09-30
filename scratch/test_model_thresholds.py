import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pickle
import pandas as pd
import numpy as np

# Load trained model
with open("models/lgbm_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]

print("Loaded model.")
# Let's inspect what features the model uses
print("Features:", model.n_features_in_)
