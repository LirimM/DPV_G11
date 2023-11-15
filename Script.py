import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "Crime_Data_from_2020_to_Present_20231113.csv"
df = pd.read_csv(file_path)

# Check for null values and visualize
null_counts = df.isnull().sum()
duplicate_count = df.duplicated().sum()

print(f"Number of null rows: '{null_counts}' | Number of duplicate rows: {duplicate_count}")

# Visualizing the amount of null & duplicate values for each column in the dataset
plt.figure(figsize = (12, 6))
plt.bar(null_counts.index, null_counts.values, color='lightcoral', label='Null Values')
plt.axhline(y = duplicate_count, color='orange', linestyle='--', label='Duplicate Rows')
plt.title('Null Values and Duplicate Rows Count in Each Column')
plt.xlabel('Columns')
plt.ylabel('Count')
plt.xticks(rotation = 45, ha = 'right')
plt.legend()
plt.show()
