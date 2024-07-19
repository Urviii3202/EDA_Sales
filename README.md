
# Data Analysis with Pandas and Visualization Libraries

This repository contains Python code for data analysis and visualization using pandas, matplotlib, seaborn, numpy, and scikit-learn.

## Overview

The script `data_analysis.py` performs several operations on the dataset `sales.csv`. Here’s a breakdown of what each section of the code does:

### 1. Importing Libraries and Loading Data

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
x = pd.read_csv("sales.csv")
```

### 2. Exploring the Dataset

```python
# Renaming columns for clarity
x.columns = ['Year', 'Product', 'line', 'Product.1', 'type', 'Product.2', 'Order', 'method', 'type.1', 'Retailer', 'country', 'Revenue']

# Displaying the first 5 rows
print(x.head(5))

# Unique countries in the dataset
print(x['country'].unique())

# Filtering data for a specific country (e.g., 'States')
print(x[x['country'] == 'States'].head())

# Shape of the dataset
print(x.shape)

# Summary statistics
print(x.describe())

# Information about the dataset
print(x.info())

# Checking for missing values
print(x.isnull())
print(x.isnull().sum())

# Handling missing values (e.g., setting a value to None)
x['country'][3] = None
print(x.isnull())
print(x.isnull().sum())
```

### 3. Data Preprocessing and Visualization

```python
# Encoding categorical variables (e.g., 'Year')
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x['Year'] = labelencoder.fit_transform(x['Year'])

# Count plot of Revenue values
sns.countplot(x['Revenue'].values)
plt.xlabel('Revenue')
plt.ylabel('country')
plt.show()

# Box plot by Year
fig = plt.figure()
plt.boxplot([x[x['Year'] == 0]['Revenue'], x[x['Year'] == 1]['Revenue'], x[x['Year'] == 2]['Revenue']])
plt.show()

# Scatter plot of Revenue for class 0
fig = plt.figure()
plt.scatter(x=[i for i in range(len(x[x['Year'] == 0]['Revenue']))], y=x[x['Year'] == 0]['Revenue'])
plt.ylabel('Revenue for class 0')

# Standard scaling of Year and Revenue columns
from sklearn.preprocessing import StandardScaler
Stdscaler = StandardScaler()
m = x[['Year', 'Revenue']]
M = Stdscaler.fit_transform(m)
print(M[:5])

# Normalizing Year and Revenue columns
from sklearn import preprocessing
X = x[['Year', 'Revenue']]
y = preprocessing.normalize(X)
print(y[:5])
```

## Running the Code

To run the code:

1. Ensure you have Python installed on your system.
2. Install necessary libraries: `pandas`, `matplotlib`, `seaborn`, `numpy`, and `scikit-learn`.
3. Place your dataset `sales.csv` in the same directory as `data_analysis.py`.
4. Execute the script:

   ```bash
   python data_analysis.py
   ```

## Notes

- Adjust paths and dataset names as per your actual setup.
- This script provides basic examples of data exploration, preprocessing, and visualization using Python's popular data science libraries.

Feel free to modify and extend the code for your specific data analysis needs!


This README file provides an overview of what each part of the code does, instructions for running the script, and notes for customization and extension. Adjust the details based on your specific dataset and analysis goals.
