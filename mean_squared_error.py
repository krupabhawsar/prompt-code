import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Load the data
file_path = r"C:\Users\krupa\OneDrive\Documents\PSU\spring_24\Research\Data_Scoring\new_project\data_processing\data_compiled .csv"
data = pd.read_csv(file_path)

# Step 2: Prepare the data
standard_values = data['standard']
predictions_gpt3_5 = data['gpt 3.5']
predictions_gpt4o = data['gpt 4']
predictions_gpt4o_mini = data['gpt 4-mini']

# Step 3: Calculate Mean Squared Errors
mse_gpt3_5 = mean_squared_error(standard_values, predictions_gpt3_5)
mse_gpt4o = mean_squared_error(standard_values, predictions_gpt4o)
mse_gpt4o_mini = mean_squared_error(standard_values, predictions_gpt4o_mini)

print("Mean Squared Error for GPT-3.5:", mse_gpt3_5)
print("Mean Squared Error for GPT-4o:", mse_gpt4o)
print("Mean Squared Error for GPT-4o Mini:", mse_gpt4o_mini)

# Step 4: Plot the results
plt.figure(figsize=(10, 5))
plt.plot(standard_values, label='Standard', color='black', linestyle='-')
plt.plot(predictions_gpt3_5, label='GPT-3.5', color='blue', linestyle='-')
plt.plot(predictions_gpt4o, label='GPT-4o', color='green', linestyle='-')
plt.plot(predictions_gpt4o_mini, label='GPT-4o Mini', color='gray', linestyle='-')
plt.title('Comparison of Model Predictions with Standard')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()
