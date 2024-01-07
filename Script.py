import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy import stats
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

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

# Sample a subset of your data for faster visualization and testing
sample_size = 10000
df_sample = df.sample(sample_size, random_state=42)

# Visualize class distribution before SMOTE for the sample
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df_sample['Vict Sex'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution Before SMOTE (Sample)')
plt.xlabel('Vict Sex')
plt.ylabel('Count')

# Prepare data for modeling
X_sample = pd.get_dummies(df_sample.drop(['Vict Sex'], axis=1))
y_sample = df_sample['Vict Sex']

# Handle NaN values using simple imputation
imputer = SimpleImputer(strategy='mean')  # You can use other strategies as needed
X_sample_imputed = pd.DataFrame(imputer.fit_transform(X_sample.drop(['DATE OCC'], axis=1)), columns=X_sample.columns[:-1])

# Concatenate the 'DATE OCC' column back to the imputed DataFrame
X_sample_imputed['DATE OCC'] = X_sample['DATE OCC']

# Determine the appropriate number of neighbors
n_neighbors = min(5, X_sample_imputed.shape[0] - 1)  # Set an appropriate maximum value
if n_neighbors >= X_sample_imputed.shape[0]:
    n_neighbors = X_sample_imputed.shape[0] - 1

# Split the sample dataset into training and testing sets
X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(
    X_sample_imputed, y_sample, test_size=0.2, random_state=42
)

# Exclude 'DATE OCC' column before SMOTE
X_train_sample_no_date = X_train_sample.drop(['DATE OCC'], axis=1)

# Determine the appropriate number of neighbors
n_neighbors = min(5, X_sample_imputed.shape[0] - 1)  # Set an appropriate maximum value
if n_neighbors >= X_sample_imputed.shape[0]:
    n_neighbors = X_sample_imputed.shape[0] - 1

# Check if n_neighbors is less than the number of samples
if n_neighbors >= X_train_sample_no_date.shape[0]:
    n_neighbors = X_train_sample_no_date.shape[0] - 1

# Ensure that n_neighbors is greater than 1 (SMOTE requires at least 2 neighbors)
n_neighbors = max(2, n_neighbors)

# Apply SMOTE to balance the sample dataset
smote = SMOTE(random_state=42, k_neighbors=min(n_neighbors, X_train_sample_no_date.shape[0] - 1))
X_train_resampled_sample_no_date, y_train_resampled_sample = smote.fit_resample(
    X_train_sample_no_date, y_train_sample
)

# Reset the index of X_train_sample before concatenation
X_train_sample_reset_index = X_train_sample.reset_index(drop=True)

# Concatenate 'DATE OCC' column back to the resampled features
X_train_resampled_sample = pd.concat([X_train_resampled_sample_no_date, X_train_sample_reset_index['DATE OCC']], axis=1)


# Visualize class distribution after SMOTE for the sample
plt.subplot(1, 2, 2)
pd.Series(y_train_resampled_sample).value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Class Distribution After SMOTE (Sample)')
plt.xlabel('Vict Descent')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Count the number of rows with 'Vict Age' as -1
rows_age = len(df[df['Vict Age'] == -1])

# Rows with 'Vict Age' as 0
print(f"\nNumber of rows with 'Vict Age' as -1: {rows_age}")

# Remove rows where 'Vict Age' is -1
df = df[df['Vict Age'] != -1]

# Z-score for victim age
numerical_columns = ['Vict Age']
z_scores = stats.zscore(df[numerical_columns])

# Z-score threshold so we can identify outliers (e.g 3 standard deviations)
z_threshold = 3

# Identifying outliers based on the previously set z-score
outliers = (abs(z_scores) > z_threshold).any(axis = 1)

# Print the number of outliers and their indices
print(f"\nNumber of outliers: {outliers.sum()}")
print("Indices of outliers:")
print(df.index[outliers])

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

# Distribution of victim's age visualized, before removing outliers
axes[0, 0].hist(df['Vict Age'], bins = 20, color = 'skyblue', edgecolor = 'black')
axes[0, 0].set_title('Distribution of Vict Age Before Removing Outliers')
axes[0, 0].set_xlabel('Vict Age')
axes[0, 0].set_ylabel('Count')

# Visualize the distribution of 'Vict Descent' before removing outliers
sns.countplot(data=df, x='Vict Descent', color='skyblue', ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Vict Descent Before Removing Outliers')
axes[0, 1].set_xlabel('Vict Descent')
axes[0, 1].set_ylabel('Count')

# Remove outliers
df_no_outliers = df[~outliers]

# Remove 'Vict Descent' categories representing less than 3% of the total numbers
total_counts = df_no_outliers['Vict Descent'].value_counts()
threshold_percentage = 3
valid_categories = total_counts[total_counts / len(df_no_outliers) >= threshold_percentage / 100].index
df_no_outliers_filtered = df_no_outliers[df_no_outliers['Vict Descent'].isin(valid_categories)]

# Visualize the distribution of 'Vict Age' after removing outliers
axes[1, 0].hist(df_no_outliers_filtered['Vict Age'], bins=20, color='salmon', edgecolor='black')
axes[1, 0].set_title('Distribution of Vict Age After Removing Outliers')
axes[1, 0].set_xlabel('Vict Age')
axes[1, 0].set_ylabel('Count')

# Visualize the distribution of 'Vict Descent' after removing outliers and filtering
sns.countplot(data=df_no_outliers_filtered, x='Vict Descent', color='salmon', ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Vict Descent After Removing Outliers and Filtering')
axes[1, 1].set_xlabel('Vict Descent')
axes[1, 1].set_ylabel('Count')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

df = df_no_outliers_filtered

# Remove rows based on 'DATE OCC' range because we only show stats for 2020 to 2022
initial_rows = len(df)
df = df[(df['DATE OCC'].dt.year >= 2020) & (df['DATE OCC'].dt.year <= 2022)]

# Count and print the number and percentage of rows removed based on 'DATE OCC' range
removed_rows = initial_rows - len(df)
removed_percentage = (removed_rows / initial_rows) * 100

print(f"\nRows removed based on 'DATE OCC' range (2020-2022): {removed_rows} \nPercentage of removed rows: {removed_percentage:.2f}%")

# Map numeric codes to descriptions
descent_mapping = {
    'B': 'Black',
    'H': 'Hispanic/Latin/Mexican',
    'O': 'Other',
    'W': 'White',
    'X': 'Unknown',
}

# Create a new column 'Vict Descent Desc' based on the mapping
df['Vict Descent Desc'] = df['Vict Descent'].map(descent_mapping)

# Encode 'Vict Descent' column using Label Encoding
label_encoder = LabelEncoder()
df['Vict Descent'] = label_encoder.fit_transform(df['Vict Descent'])

# Data Exploration Summary Statistics
print("Data Exploration Summary Statistics:")
print(df.info())
print(df.describe(include='all'))

# Multivariate Summary Statistics
print("\nMultivariate Summary Statistics:")
correlation_matrix = df.corr()
covariance_matrix = df.cov()

# Customized Pairplot for numerical columns
numerical_columns = df.select_dtypes(include='number').columns
plt.figure(figsize=(15, 15))
pairplot = sns.pairplot(df.sample(10000), vars=numerical_columns, diag_kind='kde', markers='o', hue='Vict Descent Desc', palette='viridis')

# Customize plot titles
pairplot.fig.suptitle('Pairplot for Numerical Columns', y=1.02)

# Rotate x-axis labels for better readability
pairplot.set(xticklabels=[])

# Add a legend
pairplot.add_legend()

# Show the plot
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Covariance Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Covariance Matrix')
plt.show()

# Identify numerical columns for skewness analysis
numerical_columns = df.select_dtypes(include = 'number').columns

# Skewness for each numerical column
skewness = df[numerical_columns].apply(lambda x: x.skew())

# Skewness visualized 
plt.figure(figsize = (12, 6))
sns.barplot(x = skewness.index, y = skewness.values, color = 'lightblue')
plt.title('Skewness of Numerical Columns')
plt.xlabel('Columns')
plt.ylabel('Skewness')
plt.xticks(rotation = 45, ha = 'right')
plt.show()

# Histogram for 'Vict Age' attribute
plt.figure(figsize = (8, 6))
sns.histplot(df['Vict Age'], kde = True, color = 'lightblue')
plt.title('Distribution of Vict Age')
plt.xlabel('Vict Age')
plt.ylabel('Frequency')
plt.show()

# Map numeric codes to descriptions
sex_mapping = {1: 'Male', 2: 'Female', 0: 'Unknown'}

# Create a new column 'Vict Sex Desc' based on the mapping
df['Vict Sex Desc'] = df['Vict Sex'].map(sex_mapping)

# Save the modified dataset as a new CSV file
new_file_name = "Preprocessed_Data.csv"
df.to_csv(new_file_name, index = False)

print(f"\nFile saved as '{new_file_name}'")

# Visualization Columns
visualization_columns = ['Vict Sex Desc', 'Vict Descent Desc', 'DATE OCC', 'AREA NAME', 'Vict Age']

# Visualize Categorical Columns
categorical_columns = df[visualization_columns].select_dtypes(include='object').columns
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    df[col].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Bar Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

#Sample with 10000
df_sample = df.sample(10000)
df_sample = df_sample.dropna(subset=['Vict Sex Desc'])


# Create a donut chart for 'Vict Sex Desc'
fig_pie_sex = px.pie(df_sample, names='Vict Sex Desc', title='Distribution of Victim Sex',
                     labels={'Vict Sex Desc': 'Victim Sex'},
                     hole=0.4)  # Use 'hole' to create a donut chart

# Add percentage labels
fig_pie_sex.update_traces(textinfo='percent+label', pull=[0.1, 0.1, 0.1])

# Extract the year from the 'DATE OCC' column
df_sample['Year'] = df_sample['DATE OCC'].dt.year

# Create a donut chart for the distribution of crimes across all years
fig_pie_year = px.pie(df_sample, names='Year', title='Crime Distribution Across Years',
                      hover_data=['DATE OCC'],
                      hole=0.4)  # Use 'hole' to create a donut chart

# Create a 3D scatter plot with 'AREA NAME', 'Vict Age', and 'Vict Descent Desc'
fig_3d_date_age_sex = px.scatter_3d(df_sample, x='AREA NAME', y='Vict Age', z='Vict Descent Desc', color='Vict Descent Desc',
                                     color_discrete_map={'Asian': 'red', 'Black': 'blue', 'White': 'green', 'Other': 'orange'})



# Create a 3D scatter plot with 'Vict Descent Desc', 'Vict Sex Desc', and 'AREA NAME'
fig_3d_descent_sex_area = px.scatter_3d(df_sample, x='Vict Descent Desc', y='Vict Sex Desc', z='AREA NAME', color='Vict Sex Desc',
                                         color_discrete_map={'Male': 'red', 'Female': 'blue', 'Other': 'green'})



# Create subplot
fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                    subplot_titles=['Distribution of Victim Sex', 'Crime Distribution Across Years',
                                     '3D scatter plot with AREA NAME, Vict Age, and Vict Descent Desc', 
                                    '3D scatter plot with Vict Descent Desc, Vict Sex Desc, and AREA NAME'])


# Add donut charts and 3D scatter plots to subplot
fig.add_trace(fig_pie_sex.data[0], row=1, col=1)
fig.add_trace(fig_pie_year.data[0], row=1, col=2)
fig.add_trace(fig_3d_date_age_sex.data[0], row=2, col=1)
fig.add_trace(fig_3d_descent_sex_area.data[0], row=2, col=2)

# Update layout
fig.update_layout(height=800, showlegend=True)

# Show the combined plot
fig.show()

areas = df_sample['AREA NAME'].value_counts().index.tolist()
sex = df_sample['Vict Sex Desc'].value_counts().index.tolist()
years = df_sample['Year'].value_counts().index.astype(str).tolist()
descent = df_sample['Vict Descent Desc'].value_counts().index.tolist()

# Function to filter and display the bar chart
def display_filtered_chart():
    area = area_dropdown.get()
    selected_year = year_dropdown.get()
    selected_sex = sex_dropdown.get()
    selected_descent = descent_dropdown.get()

    filtered_data = df_sample.copy()
    
    if area != 'All':
        filtered_data = filtered_data[filtered_data['AREA NAME'] == area]

    if selected_sex != 'All':
        filtered_data = filtered_data[filtered_data['Vict Sex Desc'] == selected_sex]

    if selected_descent != 'All':
        filtered_data = filtered_data[filtered_data['Vict Descent Desc'] == selected_descent]
    
    if selected_year != 'All':
        filtered_data = filtered_data[filtered_data['Year'] == int(selected_year)]
    
    # Clear previous plot
    ax.clear()
    
    if not filtered_data.empty:
        # Create bar chart using matplotlib
        filtered_data['AREA NAME'].value_counts().plot(kind='bar', ax=ax, rot=30)  # Rotate x-axis labels
        ax.set_title(f'Crime Distribution by Area - Year {selected_year} - Sex {selected_sex} - Descent {selected_descent}')
        ax.set_xlabel('Area')
        ax.set_ylabel('Count')
    else:
        ax.set_title('No data for selected filters')
        ax.axis('off')  # Turn off axes if there is no data
    
    # Update the existing output area with new plot
    canvas.draw()

# Create Tkinter GUI
root = tk.Tk()
root.title("Crime Distribution")

# Dropdowns
area_label = ttk.Label(root, text="Area:")
area_dropdown = ttk.Combobox(root, values=['All'] + areas)
area_dropdown.set('All')

year_label = ttk.Label(root, text="Year:")
year_dropdown = ttk.Combobox(root, values=['All'] + years)
year_dropdown.set('All')

sex_label = ttk.Label(root, text="Sex:")
sex_dropdown = ttk.Combobox(root, values=['All'] + sex)
sex_dropdown.set('All')

descent_label = ttk.Label(root, text="Descent:")
descent_dropdown = ttk.Combobox(root, values=['All'] + descent)
descent_dropdown.set('All')

# Button
filter_button = ttk.Button(root, text="Filter", command=display_filtered_chart)

# Output area (use matplotlib and FigureCanvasTkAgg)
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()

# Layout
area_label.grid(row=0, column=0, padx=3, pady=5)
area_dropdown.grid(row=0, column=1, padx=3, pady=5)
year_label.grid(row=0, column=2, padx=3, pady=5)
year_dropdown.grid(row=0, column=3, padx=3, pady=5)
sex_label.grid(row=0, column=4, padx=3, pady=5)
sex_dropdown.grid(row=0, column=5, padx=3, pady=5)
descent_label.grid(row=0, column=6, padx=3, pady=5)
descent_dropdown.grid(row=0, column=7, padx=3, pady=5)
filter_button.grid(row=0, column=8, padx=3, pady=5)
canvas_widget.grid(row=1, column=0, columnspan=9, padx=8, pady=8)

# Run Tkinter event loop
root.mainloop()
