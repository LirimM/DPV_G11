# Data Preparation and Visualization - Project

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

You can run this to install the libraries.
```bash
pip install pandas matplotlib
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

7. **Saving the Preprocessed Data:**
   - The preprocessed dataset, containing only specified columns we'll later need for visualizing, is saved as a new CSV file named "Preprocessed_Data.csv."

## Current Status

The original dataset is read from the path and preprocessed, but is not overwritten. The preprocessed dataset can be found in the same diretory as "Preprocessed_Data.csv", after you successfully execute the script. This is the progress until now, other phases will include further processing & analysing and visualizing will be done in the last phase of the project.

## Contributors

Contributors: Artan Thaqi & Lirim Maloku.  
If you'd like to contribute or improve the project, feel free to raise a pull request. 


