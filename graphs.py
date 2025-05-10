# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visualization
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
print("=== Task 1: Load and Explore the Dataset ===")

# Load the Iris dataset
try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    # Create a DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore the structure
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Clean the dataset (if needed)
# Since Iris dataset is clean, no action is required here
if df.isnull().sum().sum() > 0:
    try:
        df = df.fillna(df.mean())  # Fill missing numerical values with mean
        print("Missing values filled with mean.")
    except Exception as e:
        print(f"Error handling missing values: {e}")
else:
    print("No missing values found.")

# Task 2: Basic Data Analysis
print("\n=== Task 2: Basic Data Analysis ===")

# Compute basic statistics
print("\nBasic Statistics of Numerical Columns:")
print(df.describe())

# Group by species and compute mean for numerical columns
print("\nMean of Numerical Columns by Species:")
grouped_means = df.groupby('species').mean()
print(grouped_means)

# Observations
print("\nObservations:")
print("1. The mean sepal length varies across species, with virginica having the highest.")
print("2. Petal length and width show significant differences, useful for species classification.")

# Task 3: Data Visualization
print("\n=== Task 3: Data Visualization ===")

# Create a figure with subplots
plt.figure(figsize=(15, 10))

# 1. Line Chart: Average feature values per species
plt.subplot(2, 2, 1)
for feature in iris.feature_names:
    plt.plot(grouped_means.index, grouped_means[feature], marker='o', label=feature)
plt.title('Average Feature Values per Species')
plt.xlabel('Species')
plt.ylabel('Mean Value (cm)')
plt.legend()
plt.grid(True)

# 2. Bar Chart: Average sepal length per species
plt.subplot(2, 2, 2)
sns.barplot(x=grouped_means.index, y=grouped_means['sepal length (cm)'])
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')

# 3. Histogram: Distribution of petal length
plt.subplot(2, 2, 3)
plt.hist(df['petal length (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

# 4. Scatter Plot: Sepal length vs Petal length
plt.subplot(2, 2, 4)
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', size='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Additional Observations
print("\nVisualization Insights:")
print("1. The line chart shows virginica has the highest average values for most features.")
print("2. The bar chart confirms virginica has the longest average sepal length.")
print("3. The histogram indicates petal length has a multimodal distribution, suggesting distinct species clusters.")
print("4. The scatter plot shows clear separation between species based on sepal and petal length.")