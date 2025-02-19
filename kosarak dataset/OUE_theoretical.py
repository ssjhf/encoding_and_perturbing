import numpy as np
from scipy.optimize import minimize_scalar
from math import exp


def read_data_to_array(filename):
    with open(filename, 'r') as file:
        # 读取数据到列表，同时过滤掉999999和999998
        data = [int(num) for line in file for num in line.split() if int(num) not in [9999999, 9999998]]
    return np.array(data)

# Helper functions
def calculate_real_distribution(samples, domain_size):
    distribution = np.zeros(domain_size)
    for sample in samples:
        distribution[sample] += 1
    return distribution / len(samples)

def find_optimal_p_q_for_fi(epsilon, fi, N):
    def objective(q):
        p = exp(epsilon) * q / (1 + q * (exp(epsilon) - 1))
        if not (0 < q < 1 and 0 < p < 1 and abs(p - q) > 0.01 and p > 1 / N and q > 1 / N):
            return np.inf
        return (N*(fi * p * (1 - p) + (1 - fi) * q * (1 - q))/(p-q)**2)
    result = minimize_scalar(objective, bounds=(0.01, 0.99), method='bounded')
    if result.fun == np.inf:
        return np.nan, np.nan, np.nan
    optimal_q = result.x
    optimal_p = exp(epsilon) * optimal_q / (1 + optimal_q * (exp(epsilon) - 1))
    return optimal_p, optimal_q, result.fun

def calculate_optimal_p_q_per_fi(distribution, N, epsilon_values):
    average_variances_per_epsilon = []
    for epsilon in epsilon_values:
        ps, qs, variances = [], [], []
        for fi in distribution:
            p, q, variance = find_optimal_p_q_for_fi(epsilon, fi, N)
            ps.append(p)
            qs.append(q)
            variances.append(variance)
        average_variances_per_epsilon.append(np.nanmean(variances) / N)
    return average_variances_per_epsilon

# Load samples from saved file
filename = 'kosarak.dat'
epsilon_values = np.linspace(0.5, 5, 10)
sample = read_data_to_array(filename)
samples = (sample + 20000) // 1000
N_SAMPLES = len(samples)
print(N_SAMPLES)
DOMAIN_SIZE = max(samples) + 1
NUM_OF_EPSILON = len(epsilon_values)
EPOCHS = 100
real_distribution = calculate_real_distribution(samples, DOMAIN_SIZE)

# Calculate average variances for each epsilon
average_variances_per_epsilon = calculate_optimal_p_q_per_fi(real_distribution, N_SAMPLES, epsilon_values)

# Save the results to a text file
output_file_path = 'OUE_theoretical.txt'
with open(output_file_path, 'w') as file:
    for epsilon, avg_variance in zip(epsilon_values, average_variances_per_epsilon):
        file.write(f"Epsilon: {epsilon}, Average Variance/N: {avg_variance}\n")

