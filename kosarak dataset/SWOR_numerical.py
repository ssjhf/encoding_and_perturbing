import numpy as np
from scipy.stats import zipf
from scipy.optimize import minimize_scalar
from math import exp

def read_data_to_array(filename):
    with open(filename, 'r') as file:
        # 读取数据到列表，同时过滤掉999999和999998
        data = [int(num) for line in file for num in line.split() if int(num) not in [9999999, 9999998]]
    return np.array(data)

def calculate_real_distribution(samples, domain_size):
    """Calculate the real distribution of Zipf-generated samples."""
    distribution = np.zeros(domain_size)
    for sample in samples:
        distribution[sample] += 1
    return distribution / len(samples)

def perturb_data(N, binary_matrix, domain_size, statistical_probs, epsilon):
    """Apply the perturbation process without replacement."""
    perturbed_matrix = np.zeros_like(binary_matrix)
    corrected_means = np.zeros(domain_size)

    for j in range(domain_size):
        fj = statistical_probs[j]
        x = np.exp(epsilon)
        fj1 = (1 - np.sqrt(2 / (np.sqrt(x) + 1))) / 2
        fj2 = (1 + np.sqrt(2 / (np.sqrt(x) + 1))) / 2
        if fj >= fj1 and fj <= fj2:
            p = (np.sqrt(x)-1)/(np.sqrt(x)+1)
            if (1 - 4 * p / (1 - p) / (1 - p) / (x - 1)) >= 0:
                pi_B = (1 - np.sqrt(1 - 4 * p / (1 - p) / (1 - p) / (x - 1))) / 2
            else:
                pi_B = 0
            # Generate random cards for two groups
            black_cards = np.array([1] * int(N * pi_B) + [0] * (N - int(N * pi_B)))
            np.random.shuffle(black_cards)
            red_cards = np.array([1] * int(N * p) + [0] * (N - int(N * p)))
            np.random.shuffle(red_cards)
            # Apply vectorized perturbation
            perturbed_matrix[:, j] = np.where(red_cards == 1, binary_matrix[:, j], black_cards)

            # Calculate corrected means
            corrected_means[j] = (np.sum(perturbed_matrix[:, j]) - (1 - p) * pi_B * N) / p
        elif fj != 0 and fj !=1:
            p = np.sqrt((x-1)*fj*(1-fj))/(1+np.sqrt((x-1)*fj*(1-fj)))
            if (1 - 4 * p / (1 - p) / (1 - p) / (x - 1)) >= 0:
                pi_B = (1 - np.sqrt(1 - 4 * p / (1 - p) / (1 - p) / (x - 1))) / 2
            else:
                pi_B = 0            # Generate random cards for two groups
            black_cards = np.array([1] * int(N * pi_B) + [0] * (N - int(N * pi_B)))
            np.random.shuffle(black_cards)
            red_cards = np.array([1] * int(N * p) + [0] * (N - int(N * p)))
            np.random.shuffle(red_cards)
            # Apply vectorized perturbation
            perturbed_matrix[:, j] = np.where(red_cards == 1, binary_matrix[:, j], black_cards)

            # Calculate corrected means
            corrected_means[j] = (np.sum(perturbed_matrix[:, j]) - (1 - p) * pi_B * N) / p
        else:
            corrected_means[j] = 0
    return perturbed_matrix, corrected_means

def simulate_variance(N, domain_size, epsilon, epochs):
    """Simulate variance for different epsilon values."""
    filename = 'kosarak.dat'
    sample = read_data_to_array(filename)
    samples = (sample + 20000) // 1000
    statistical_probs = calculate_real_distribution(samples, domain_size)
    corrected_means_all = np.zeros((epochs, domain_size))

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        binary_matrix = np.zeros((N, domain_size), dtype=int)
        binary_matrix[np.arange(N), samples] = 1
        perturbed_matrix, corrected_means = perturb_data(N, binary_matrix, domain_size, statistical_probs, epsilon)
        corrected_means_all[epoch] = corrected_means

    variances = np.mean(np.var(corrected_means_all, axis=0)) / N * domain_size / np.count_nonzero(statistical_probs)
    return variances

# Set parameters
filename = 'kosarak.dat'
epsilon_values = np.linspace(0.5, 5, 10)
sample = read_data_to_array(filename)
samples = (sample + 20000) // 1000
N_SAMPLES = len(samples)
DOMAIN_SIZE = max(samples) + 1
NUM_OF_EPSILON = len(epsilon_values)
EPOCHS = 100
simulation_variance = np.zeros(len(epsilon_values))

# Calculate variance for each epsilon
for i, eps in enumerate(epsilon_values):
    simulation_variance[i] = simulate_variance(N_SAMPLES, DOMAIN_SIZE, eps, EPOCHS)
    print(f"Simulation Variance for epsilon={eps}: {simulation_variance[i]}")

# Write results to file
with open('SWOR_numerical.txt', 'w') as file:
    for i in range(len(epsilon_values)):
                file.write(f"Epsilon: {epsilon_values[i]}, Average Variance/N: {simulation_variance[i]}\n")

print("Data written to file.")
