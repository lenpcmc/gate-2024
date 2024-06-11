from scipy.optimize import least_squares 
import numpy as np 

# Sample data (replace with your actual data) 
x = np.array([1, 2, 3, 4, 5]) 
y = np.array([2, 3, 5, 7, 11]) 

# True parameters of the model (for illustration only) 
true_a = 2 
true_b = 3 

# Define the model function (linear in this case) 
def model_function(params, x): 
    a, b = params 
    return a * x + b 

# Generate some random data with noise (replace with your actual data generation) 
def generate_data(num_points, true_a, true_b, noise_std): 
    x = np.linspace(0, 10, num_points) 
    y = model_function([true_a, true_b], x) + np.random.randn(num_points) * noise_std 
    return x, y 

# Generate sample data with noise 
x, y = generate_data(5, true_a, true_b, 1.0) 

# Initial guess for parameters (replace with better guess if possible) 
params_guess = (1.0, 1.0) 

# Solve least squares 
solution = least_squares(model_function, params_guess, args=(x,))

# Access results 
optimal_params = solution.x 



# Print the results 
print("True parameters (a, b):", true_a, true_b) 
print("Optimal parameters (a, b) from least squares:", optimal_params)

# Calculate residuals (errors) between data and model with optimal parameters 
residuals = y - model_function(optimal_params, x) 
print("Residuals (errors):", residuals)
