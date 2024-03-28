import numpy as np

# Constants
zipf_samples_path = "zipf_samples.npy"
samples = np.load(zipf_samples_path)
N_SAMPLES = len(samples)
DOMAIN_SIZE = 2
LOWER_BOUND = 0
EPOCHS = 1000
NUM_OF_EPSILONS = 4
epsilon_values = [0.5, 1, 2, 4]

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
            other_values = list(set(range(LOWER_BOUND, domain_size)) - {sample})
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