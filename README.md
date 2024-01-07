# Project in the subject "Data Preparation and Visualization" - Computer Engineering Master's Program 23/24 - [University of Prishtina](https://fiek.uni-pr.edu)

## Overview

This project focuses on preprocessing and visualizing a dataset consisting of crime rate data in LA. The dataset will be preprocessed utilizing Python and its libraries, in order to become more suitable for visualization.

## Dataset

The dataset consists of data related to crime in LA from 2020 to present however we will only analyze the data between 2020 - 2022 range. This data has been recorded from the police department and it gets updated frequently. The dataset has 839 thousand rows and 28 columns where each row is a crime incident. The version of the dataset that is being used for this project is downloaded in Nov, 2023. 
The dataset is attached to this repository, and additionally you can find it in this [LINK](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8). 

## How to run the script

The script is written in Python so you will need to install Python in your machine. More information about it you can find [Python's Official Page](https://www.python.org/).

After installing Python, these are the libraries required to run the scripts: 

- **pandas**: Used for data manipulation and analysis.
- **matplotlib**: Utilized for creating visualizations to better understand the dataset.
- **seaborn**: A statistical data visualization library based on Matplotlib.
- **imblearn**: Used for over-sampling with SMOTE to balance class distribution.
- **scikit-learn**: A machine learning library, here used for various tasks including imputation, splitting the dataset, and label encoding.
- **numpy**: A library for numerical operations in Python.
- **scipy**: A library for scientific and technical computing.
- **plotly**: A library for interactive and browser-based plotting.
- **tkinter**: The standard Python interface to the Tk GUI toolkit.

You can run this to install the libraries.
```bash
pip install pandas matplotlib seaborn imbalanced-learn scikit-learn numpy scipy plotly tk
```

As a last step you will need to extract the dataset and place it into the same directory as the script you are running, or change the path inside the code.

## Steps Taken

 1. **Loading the data:**
   - The dataset is read from a path and loaded into a pandas DataFrame using the `pd.read_csv` function.

 2. **Checking for Null Values and Visualizing:**
   - A null value count check and visualization (for easier judgement) is performed on each column of the dataset.

 3. **Handling Null Values:**
   - Specific columns, including 'Vict Sex,' 'Vict Descent,' 'DATE OCC,' 'AREA NAME,' and 'Vict Age,' are analyzed for null values.
   - Null values in 'Vict Sex' and 'Vict Descent' are filled with 'X' for Unknown.

 4. **Analyzing 'Vict Age':**
   - The number of rows with 'Vict Age' as 0 is counted and printed.
   - The average age is calculated, and null values and rows with 'Vict Age' as 0 are filled with this average.

 5. **Filtering Rows Based on Date Range:**
   - The 'DATE OCC' that represents date of occurrence is converted to the  datetime format.
   - Rows outside the date range of 2020-2022 are removed.

 6. **Binarization of 'Sex' Column:**
   - The 'Vict Sex' column is binarized, mapping 'M' to 1, 'F' to 2, and 'X' to 0.

 7. **Handling Class Imbalance with SMOTE:**
   - A subset of the data is sampled for faster visualization and testing.
   - Class distribution before and after applying SMOTE is visualized.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/3f0d0770-b7ff-4683-a5ed-c8045acd6801)

 8. **Handling 'Vict Age' Outliers and/or Anomalies:**
   - Rows with 'Vict Age' as -1 are identified and removed.
   - Z-score is calculated to identify outliers, and visualization is done before and after removing outliers.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/fe403c85-7f0b-471f-9488-5233dfb2c453)

 9. **Handling/Filtering 'Vict Descent' Categories:**
   - Categories representing less than 3% of the total numbers in 'Vict Descent' are removed.

10. **Correlation and covariance examples:**
   - Correlation and covariance matrices are generated on numerical attributes.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/8911cc04-90d3-469d-bea9-fc3fce28377e)

11. **Skewness of numerical attributes:**
   - Skewness of numerical attributes has been generated, as shown below.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/3624996c-a26e-493b-b8a0-0478af43f751)

12. **Saving the Preprocessed Data:**
   - The preprocessed dataset, containing only specified columns we'll later need for visualizing, is saved as a new CSV file named "Preprocessed_Data.csv."

## Visualization
This part focuses on visualizing numerical and categorical columns in the dataset using bar plots, donut charts, 3D scatter plots and a filtered bar chart. Each of visualizations represents the distribution of values.

 1. **Distribution of Victims Age**
   - This is a histogram illustrating the distribution of ages ('Vict Age') among crime victims in the dataset.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/54f89736-b828-4282-8660-c0184c186385)

 2. **Distribution of Victims Sex**
   - This is a bar plot illustrating the distribution of Victims Sex ('Vict Sex') among crime victims in the dataset.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/101abbba-0bfc-405b-b0c1-c0f15e58eebc)

 3. **Distribution of Victims Descent**
   - This is a bar plot illustrating the distribution of Victims Descent ('Vict Descent') among crime victims in the dataset.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/c20344c9-a733-4def-a252-126dcf23db71)

 4. **Distribution of Areas**
   - This is a bar plot illustrating the distribution of Areas ('Area name') where crimes happend from the data in the dataset.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/b9c53290-0961-4cb1-bbf5-807e081b06a8)

 5. **Distribution of Victims Sex and Crime Distribution Across Years**
   - Two donut charts are created to illustrate the distribution of crime based on victim sex and across different years (Sample 10000). These charts provide a clear visual representation of the data, with percentage labels for enhanced understanding.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/a137a55e-1464-400a-9128-762125fae2bb)
   
 6. **3D scatter plots showing relationships between Areas, Victims Age, Victims Descent and Victims Sex**
   - Two 3D scatter plots are generated to explore relationships between 'AREA NAME', 'Vict Age', 'Vict Descent Desc', and 'Vict Sex Desc' (Sample 10000). These visualizations offer a unique perspective on the dataset.
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/6685a300-00a5-453d-aac9-6ebf22c8b18f)
   
 7. **Filtered Bar Chart**
   - A function and GUI are implemented to allow users to filter the data and generate a bar chart based on selected criteria. The bar chart dynamically updates based on user-selected options for area, year, sex, and descent (Sample 10000).
   - ![image](https://github.com/LirimM/DPV_G11/assets/46811308/df6c02ee-3a19-45bd-81aa-187ccde7d16f)


## Current Status

The initial phase involved reading and preprocessing the original dataset, saving the cleaned version as 'Preprocessed_Data.csv' in the same directory. Subsequently, the project advanced through successive phases, incorporating further data processing, in-depth analysis, and the implementation of comprehensive visualizations. The culmination of these efforts is evident in the completion of the last project phase, providing a holistic exploration and understanding of the crime dataset.

## Contributors

Contributors: Artan Thaqi & Lirim Maloku.  
If you'd like to contribute or improve the project, feel free to raise a pull request. 


