import numpy as np
from scipy.stats import zipf

# Constants

def read_data_to_array(filename):
    with open(filename, 'r') as file:
        # 读取数据到列表，同时过滤掉999999和999998
        data = [int(num) for line in file for num in line.split() if int(num) not in [9999999, 9999998]]
    return np.array(data)

def calculate_real_distribution(samples, domain_size):
    """Calculate the actual distribution of the samples."""
    distribution = np.zeros(domain_size)
    for sample in samples:
        distribution[sample] += 1
    return distribution / len(samples)

# Generate samples and calculate statistical probabilities
filename = 'usa_00010.dat'
epsilon_values = np.linspace(0.5, 5, 10)
sample = read_data_to_array(filename)
samples = (sample + 20000) // 1000
N_SAMPLES = len(samples)
DOMAIN_SIZE = max(samples) + 1
NUM_OF_EPSILON = len(epsilon_values)
statistical_probabilities = calculate_real_distribution(samples, DOMAIN_SIZE)

# Theoretical variance calculation
theoretical_variance = np.zeros(NUM_OF_EPSILON)

for i, epsilon_value in enumerate(epsilon_values):
    x = np.exp(epsilon_value)
    fj1 = (1 - np.sqrt(2 / (np.sqrt(x) + 1))) / 2
    fj2 = (1 + np.sqrt(2 / (np.sqrt(x) + 1))) / 2
    theoretical_variance[i] = sum(
        ((1 + 4 * fj * (1 - fj)) / 2 / (N_SAMPLES - 1) / (np.sqrt(x) - 1) if fj1 <= fj <= fj2 else
         (1 / (N_SAMPLES - 1) * (2 * np.sqrt(fj * (1 - fj)) / (np.sqrt(x - 1)) + 1 / (x - 1)) if (fj2 < fj <1 or 0< fj < fj1) else
          0)) for fj in statistical_probabilities
    )/ np.count_nonzero(statistical_probabilities) * N_SAMPLES

print("Theoretical Variance:", theoretical_variance)

# Write results to file
output_file_path = 'SWOR_theoretical.txt'
with open(output_file_path, 'w') as file:
    for epsilon, avg_variance in zip(epsilon_values, theoretical_variance):
        file.write(f"Epsilon: {epsilon}, Average Variance/N: {avg_variance}\n")

