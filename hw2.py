url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0, 1, 2, 3])

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[4, 5, 6], [7, 8, 9]])

vertical_stack = np.vstack((A, B))
horizontal_stack = np.hstack((A, B))

# Question 2: Find common elements between A and B
common_elements = np.intersect1d(A, B)

# Question 3: Extract all numbers from A which are within a specific range (5 to 10)
filtered_A = A[(A >= 5) & (A <= 10)]

# Question 4: Filter the rows of iris_2d that have petal length (3rd column) > 1.5 and sepal length (1st column) < 5.0
filtered_iris = iris_2d[(iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)]

# Part 2 - Pandas

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# Question 1: Filter 'Manufacturer', 'Model', and 'Type' for every 20th row starting from the first row (row 0)
filtered_df_20th_rows = df.loc[::20, ['Manufacturer', 'Model', 'Type']]

# Question 2: Replace missing values in 'Min.Price' and 'Max.Price' columns with their respective means
df[['Min.Price', 'Max.Price']] = df[['Min.Price', 'Max.Price']].apply(lambda x: x.fillna(x.mean()), axis=0)

# Question 3: Get the rows of a dataframe where the row sum is greater than 100
df_random = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
rows_with_sum_gt_100 = df_random[df_random.sum(axis=1) > 100]

# Display results: 
tools.display_dataframe_to_user(name="Stacked Vertically", dataframe=pd.DataFrame(vertical_stack))
tools.display_dataframe_to_user(name="Stacked Horizontally", dataframe=pd.DataFrame(horizontal_stack))
tools.display_dataframe_to_user(name="Common Elements", dataframe=pd.DataFrame(common_elements))
tools.display_dataframe_to_user(name="Filtered A (values between 5 and 10)", dataframe=pd.DataFrame(filtered_A))
tools.display_dataframe_to_user(name="Filtered Iris Dataset", dataframe=pd.DataFrame(filtered_iris))
tools.display_dataframe_to_user(name="Filtered Car Dataset", dataframe=filtered_df_20th_rows)
tools.display_dataframe_to_user(name="Rows with Sum > 100", dataframe=rows_with_sum_gt_100)
