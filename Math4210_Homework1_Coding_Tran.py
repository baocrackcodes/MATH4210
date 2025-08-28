import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm

#Question 1: Least Squares Regression
df = pd.read_csv('Auto.csv')
df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce') #cleaning up data as some columns contain '?'
df = df.dropna(subset=['mpg', 'horsepower'])
response = df['mpg'].values.astype(float)
predictor = df['horsepower'].values.astype(float)

matrix = np.vstack([predictor,np.ones(len(predictor))]).T
response = response[:,np.newaxis]

beta, *_ = np.linalg.lstsq(matrix, response, rcond=None)
m, b = beta

#sort for a clean line in the form of y = mx + b
order = np.argsort(predictor)
x_sorted = predictor[order]
y_fit_sorted = m * x_sorted + b

residuals = response - y_fit_sorted
RSS = np.sum(residuals **2)
print(f"The RSS value is {RSS}.")
#plotting
plt.title("Linear Regression through Least Squares Method for 'mpg' vs 'horsepower'")
plt.scatter(predictor, response, alpha=0.7, label='Given Data')
plt.plot(x_sorted, y_fit_sorted)
plt.xlabel('Predictor (Horsepower)')
plt.ylabel('Response (MPG)')
plt.legend()
plt.show()

#AI Statement: used python numerical methods Berkeley website to find pre-built functions and syntax to apply least squares

#Question 2: Multiple Linear Regression on the Auto Dataset
df_numeric = df.drop(columns=["name"])
df_numeric = df_numeric.apply(pd.to_numeric, errors = 'coerce')
df_numeric = df_numeric.dropna()

#correlation matrix
correlation_matrix = df_numeric.corr()
y = df_numeric['mpg'].values
X = df_numeric.drop(columns=['mpg']).values

Xc = sm.add_constant(X) #adding a column of 1s
model = sm.OLS(y, Xc).fit()

coefficients = model.params
RSS = float((model.resid**2).sum())

print("Coefficients:\n", coefficients, "\n")
print("RSS for multiple linear regressions:", RSS)

plt.figure(figsize=(10, 8))
seaborn.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Auto Dataset")
plt.show()

#AI Statement: Used ChatGPT to figure out how to plot a heatmap and learn about statsmodels.api to try a different method of doing least squares

#Question 3: Exercise 13 part (a) to (i)
np.random.seed(1)
feature = np.random.normal(0,1,100)
eps = np.random.normal(0,0.5,100)
Y = -1 + 0.5*feature + eps

matrix1 = np.vstack([feature,np.ones(len(feature))]).T
Y = Y[:,np.newaxis]

beta, *_ = np.linalg.lstsq(matrix1, Y, rcond=None)
m, b = beta

#sort for a clean line in the form of y = mx + b
order = np.argsort(feature)
x_sorted = feature[order]
y_fit_sorted = m * x_sorted + b


plt.scatter(feature,Y)
plt.title('Scatter Plotting X against Y')
plt.plot(x_sorted,y_fit_sorted,label='Regression Line')
plt.plot(feature, -1 + 0.5*feature,label='Population Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

X_quad = np.c_[np.ones_like(feature), feature, feature**2]
b_quad, *_ = np.linalg.lstsq(X_quad, Y, rcond=None)
b0,b1,b2 = b_quad
y_fit_sorted_quad = b0 + b1*x_sorted + b2*x_sorted**2

plt.scatter(feature,Y)
plt.title('Scatter Plotting X against Y Quadratic')
plt.plot(x_sorted,y_fit_sorted_quad,label="Quadratic Fit")
plt.plot(x_sorted,y_fit_sorted,label='Regression Line')
plt.plot(feature, -1 + 0.5*feature,label='Population Line')
plt.legend()
plt.show()