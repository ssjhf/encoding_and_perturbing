import numpy as np

def read_data_to_array(filename):
    with open(filename, 'r') as file:
        # 读取数据到列表，同时过滤掉999999和999998
        data = [int(num) for line in file for num in line.split() if int(num) not in [9999999, 9999998]]
    return np.array(data)

# Constants
filename = 'usa_00010.dat'
epsilon_values = np.linspace(0.5, 5, 10)
sample = read_data_to_array(filename)
samples = (sample + 20000) // 1000
N_SAMPLES = len(samples)
DOMAIN_SIZE = max(samples) + 1
NUM_OF_EPSILON = len(epsilon_values)
EPOCHS = 10

def calculate_theoretical_variance(domain_size, epsilon_values):
    """Calculate theoretical variance for given epsilon values."""
    variances = np.zeros(len(epsilon_values))
    for i, epsilon in enumerate(epsilon_values):
        variances[i] = (domain_size - 2 + np.exp(epsilon)) / (np.exp(epsilon) - 1)**2
    return variances

def perform_random_mapping(samples, p, domain_size):
    """Randomly map samples based on probability p."""
    mapped_samples = samples.copy()
    for i, sample in enumerate(samples):
        if np.random.random() >= p:
            other_values = list(set(range(0, domain_size)) - {sample})
            mapped_samples[i] = np.random.choice(other_values)
    return mapped_samples

def simulate_variance(n, samples, domain_size, epsilon_values, epochs):
    """Simulate variance for different epsilon values."""
    variances = np.zeros(len(epsilon_values))
    for i, epsilon in enumerate(epsilon_values):
        p = np.exp(epsilon) / (np.exp(epsilon) + domain_size - 1)
        q = 1 / (np.exp(epsilon) + domain_size - 1)
        pi_A_estimators = np.zeros(epochs)

        for epoch in range(epochs):
            print(epoch)
            mapped_samples = perform_random_mapping(samples, p, domain_size)
            statistics = np.bincount(mapped_samples, minlength=domain_size)
            pi_A_estimators[epoch] = (statistics[0] - n * q) / (p - q)

        variances[i] = np.var(pi_A_estimators) / n
    return variances

# Main execution
theoretical_variance = calculate_theoretical_variance(DOMAIN_SIZE, epsilon_values)
simulation_variance = simulate_variance(N_SAMPLES, samples, DOMAIN_SIZE, epsilon_values, EPOCHS)

# Output results
print("Theoretical Variance:", theoretical_variance)
print("Simulation Mean Squared Error:", simulation_variance)

# 根据变量名生成文件名
file_name = f"DE{DOMAIN_SIZE}.txt"

# 写入文件
with open(file_name, 'w') as file:
    for i, epsilon in enumerate(epsilon_values):
        file.write(f"{epsilon}\t{theoretical_variance[i]}\t{simulation_variance[i]}\n")

print("Data written to file.")


# Save the results to a text file
output_file_path = 'DE_theoretical.txt'
with open(output_file_path, 'w') as file:
    for epsilon, avg_variance in zip(epsilon_values, theoretical_variance):
        file.write(f"Epsilon: {epsilon}, Average Variance/N: {avg_variance}\n")

# Save the results to a text file
output_file_path = 'DE_numerical.txt'
with open(output_file_path, 'w') as file:
    for epsilon, avg_variance in zip(epsilon_values, simulation_variance):
        file.write(f"Epsilon: {epsilon}, Average Variance/N: {avg_variance}\n")