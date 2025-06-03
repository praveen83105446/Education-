import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df= pd.read_csv('live_traffic_prediction_dataset.csv')
print(df.head())


print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())


# Check for duplicates
print(df.duplicated().sum())
# Remove duplicates
df = df.drop_duplicates()
print(df.duplicated().sum())\


df.to_csv ("cleaned_live_traffic_prediction_datast.csv", index=False)
print(df)


#EDA
#univariate analysis
sns.histplot(df['Traffic Level'], kde=True)
plt.title('Traffic Volume Distribution')
plt.xlabel('Traffic Level')
plt.ylabel('Frequency')
plt.show()

#bivariate analysis
sns.scatterplot(x='Timestamp', y='Traffic Level', data=df)
plt.title('Traffic Level vs Timestamp')
plt.xlabel('Timestamp')
plt.ylabel('Traffic Level')
plt.show()

#multivariate analysis
sns.pairplot(df, vars=['Traffic Level', 'Weather', 'Suggested Alternate Route'])
plt.title('Pairplot of Traffic Level with Weather and Suggested Alternate Route')
plt.show()

#linear regression model
X = df[['Speed (km/h)']]
y = df['Congestion Probability (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()  
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualizing the predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Traffic Level')
plt.ylabel('Predicted Traffic Level')
plt.title('Actual vs Predicted Traffic Level')
plt.show()