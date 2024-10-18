import numpy as np
import pandas as pd

# 1. Define two custom numpy arrays, A and B

a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

vertical_stack = np.vstack((a,b))
horizontal_stack = np.hstack((a,b))

print(vertical_stack)
print(horizontal_stack)

# 2. Find common elements between A and B

common_elem = np.intersect1d(a,b)
print("Common Elements:", common_elem)

# 3. Extract all numbers from A which are within a specific range, e.g., between 5 and 10
range_filter = a[(a >= 5) & (a <= 10)]
print("Numbers in A between 5 and 10:", range_filter)

# 4. Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])

filtered_iris = iris_2d[(iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)]
print("Filtered Iris Data:\n", filtered_iris)

# Pandas Questions

# 5. Filter the 'Manufacturer', 'Model', and 'Type' for every 20th row starting from 1st (row 0)
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
filtered_df = df.loc[::20, ['Manufacturer', 'Model', 'Type']]
print("Every 20th row:\n", filtered_df)

# 6. Replace missing values in Min.Price and Max.Price columns with their respective mean
df['Min.Price'].fillna(df['Min.Price'].mean(), inplace=True)
df['Max.Price'].fillna(df['Max.Price'].mean(), inplace=True)
print("Missing values replaced with mean in Min.Price and Max.Price")

# 7. Get the rows of a dataframe with row sum > 100
df_random = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
rows_with_sum_gt_100 = df_random[df_random.sum(axis=1) > 100]
print("Rows with sum > 100:\n", rows_with_sum_gt_100)
