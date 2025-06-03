
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('student_performance.csv')
print(df.head())

print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())


# Check for duplicates
print(df.duplicated().sum())
# Remove duplicates
df = df.drop_duplicates()
print(df.duplicated().sum())


df.to_csv ("cleaned_student_performance.csv", index=False)
print(df)



# EDA
# Univariate analysis
sns.histplot(df['Previous_Grades'], kde=True)
plt.title('Final Grade Distribution')
plt.xlabel('Previous_Grades')
plt.ylabel('Frequency')
plt.show()

# Bivariate analysis
sns.scatterplot(x='Study_Hours_Per_Week', y='Previous_Grades', data=df)
plt.title('Previous_Grades v study_Hours_Per_Week ')
plt.xlabel('study_Hours_Per_Week ')
plt.ylabel('Previous_Grades')
plt.show()


# Multivariate analysis
sns.pairplot(df, vars=['Previous_Grades', 'Study_Hours_Per_Week', 'Attendance_Percentage'])
plt.title('Pairplot of Previous_Grades with Study_Hours_Per_Week and Attendance_Percentage')
plt.show()

#linear regression model
X = df[['Study_Hours_Per_Week', 'Attendance_Percentage']]
y = df['Previous_Grades']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Previous Grades')
plt.ylabel('Predicted Previous Grades')
plt.title('Actual vs Predicted Previous Grades')
plt.show()





