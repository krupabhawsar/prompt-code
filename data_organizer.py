import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = r"C:\Users\krupa\OneDrive\Documents\PSU\spring_24\Research\Data_Scoring\new_project\rounded_data.csv"
df = pd.read_csv(file_path)

# Function to save filtered data based on originality score
def save_filtered_data(df, score, folder_path):
    filtered_df = df[df['originality_rescaled_factor'] == score]
    file_path = f"{folder_path}\\{score}_rated_data.csv"
    filtered_df.to_csv(file_path, index=False)
    print(f"{score}_rated_data.csv successfully saved")

# Save data for each originality score
scores = [1, 2, 3, 4, 5]
folder_path = r"C:\Users\krupa\OneDrive\Documents\PSU\spring_24\Research\Data_Scoring\new_project\\"
for score in scores:
    save_filtered_data(df, score, folder_path)

# Split the dataset into train, test, and dev datasets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
test_df, dev_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Paths for the datasets
train_path = folder_path + "train_data.csv"
test_path = folder_path + "test_data.csv"
dev_path = folder_path + "dev_data.csv"

# Save the datasets
train_df.to_csv(train_path, index=False)
print("train_data.csv successfully saved")

test_df.to_csv(test_path, index=False)
print("test_data.csv successfully saved")

dev_df.to_csv(dev_path, index=False)
print("dev_data.csv successfully saved")
