import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error

#Part a)
rng = np.random.default_rng()
x = rng.normal(size=100)
y = x - 2 * x**2 + rng.normal(size=100)

#Part b)
plt.scatter(x,y)
plt.title('Scatter plot of x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


#Part c)
#Define the data
n = 100
X = rng.normal(size=n).reshape(-1,1)
y = (X[:,0] - 2 * X[:,0]**2 + rng.normal(size=n)).reshape(-1,1)

#LOOCV helper
loo = LeaveOneOut()

def loocv_mse(degree: int) -> float:
    Xp = PolynomialFeatures(degree=degree, include_bias=True).fit_transform(X)
    y_pred = cross_val_predict(LinearRegression(), Xp, y, cv=loo, method='predict')
    return mean_squared_error(y, y_pred)

#Compute LOOCV for degrees 1-4
results = {d: loocv_mse(d) for d in range(1,5)}
for d, mse in results.items():
    print(f"Degree {d}: LOOCV MSE = {mse:.6f}")

#AI Statement: Used AI to help generate code for the LOOCV helper function as I did not know how that works. 