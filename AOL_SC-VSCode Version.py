import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve

# Load the data, skipping the header row
data_path = 'D:/AOL/AOL_SC/aol_data.csv'
data = pd.read_csv(data_path, header=None, skiprows=1)

# Extract the relevant columns
months = np.arange(1, len(data.columns) + 1)  # 1 to 144
production = data.iloc[0].values  # Get the first row

# Define models
def exponential_model(x, a, b, c):
    return a * np.exp(b * x) + c

def polynomial_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Fit models
popt_exp, _ = curve_fit(exponential_model, months, production, maxfev=10000)
popt_poly, _ = curve_fit(polynomial_model, months, production)

# Evaluate models
exp_pred = exponential_model(months, *popt_exp)
poly_pred = polynomial_model(months, *popt_poly)

# Plot the data and models
plt.figure(figsize=(14, 7))
plt.scatter(months, production, label='Actual Data')
plt.plot(months, poly_pred, label='Exponential Model', color='red',linestyle= '--')
plt.plot(months, production, label='Polynomial Model', color='green')
plt.xlabel('Month')
plt.ylabel('Production')
plt.title('Monthly Bag Production')
plt.legend()
plt.show()

# Print the numerical model
numerical_model = f'{popt_poly[0]}*x**3 + {popt_poly[1]}*x**2 + {popt_poly[2]}*x + {popt_poly[3]}'
print('Numerical Model:', numerical_model)

# Define the equation to find the root
def warehouse_capacity(x):
    return polynomial_model(x, *popt_poly) - 25000

# Solve for the root
month_exceed = fsolve(warehouse_capacity, 150)[0]

# Calculate the start of new warehouse construction
start_month = month_exceed - 13
print(f"Month to start building a new warehouse: {start_month:.2f}")