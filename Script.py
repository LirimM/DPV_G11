import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "Crime_Data_from_2020_to_Present_20231113.csv"
df = pd.read_csv(file_path)

# Check for null values and visualize
null_counts = df.isnull().sum()
duplicate_count = df.duplicated().sum()

print(f"Number of null rows \n'{null_counts}' \nNumber of duplicate rows: {duplicate_count} \n")

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

# Print the percentage and number of null values for specified columns
visualization_columns = ['Vict Sex', 'Vict Descent', 'DATE OCC', 'AREA NAME', 'Vict Age']

for col in visualization_columns:
    null_percentage = (df[col].isnull().sum() / len(df)) * 100
    null_count = df[col].isnull().sum()
    print(f"Column: '{col}' | Percentage of rows with null values: {null_percentage:.2f}% | Number of null values: {null_count}")

# Fill null values in 'Vict Sex' and 'Vict Descent' with 'X' for unknown
df['Vict Sex'].fillna('X', inplace=True)
df['Vict Descent'].fillna('X', inplace=True)

# Count the number of rows with 'Vict Age' as 0
rows_age= len(df[df['Vict Age'] == 0])

# Rows with 'Vict Age' as 0
print(f"\nNumber of rows with 'Vict Age' as 0: {rows_age}")

# Calculate average age excluding rows where 'Vict Age' is 0
average_age = df[df['Vict Age'] != 0]['Vict Age'].mean()


# Fill null values and rows with 'Vict Age' as 0 with the calculated average
df['Vict Age'].replace(0, pd.NA, inplace = True)
df['Vict Age'].fillna(average_age, inplace = True)

# Print the average age and the number of rows affected
print(f"\nAverage Age: {average_age:.2f}")


# Convert 'DATE OCC' to datetime format
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Select only the columns in visualization_columns
df = df[visualization_columns]

# Binarization of 'Sex' column
sex_mapping = {'M': 1, 'F': 2, 'X': 0}
df['Vict Sex'] = df['Vict Sex'].map(sex_mapping)

# Save the modified dataset as a new CSV file
new_file_name = "Preprocessed_Data.csv"
df.to_csv(new_file_name, index = False)

print(f"\nFile saved as '{new_file_name}'")
