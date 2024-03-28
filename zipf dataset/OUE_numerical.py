# Correcting the provided code for epsilon values [0.5, 1, 2, 4] and saving results to a txt file.

import numpy as np
from scipy.stats import zipf
from scipy.optimize import minimize_scalar
from math import exp

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
    return optimal_p, optimal_q

def perturb_data_without_replacement(N, binary_matrix, p, q):
    # red_cards_group1 = np.array([1] * int(N * p) + [0] * (N - int(N * p)))
    # np.random.shuffle(red_cards_group1)
    # red_cards_group2 = np.array([1] * int(N * q) + [0] * (N - int(N * q)))
    # np.random.shuffle(red_cards_group2)
    # perturbed_matrix = np.where(binary_matrix, red_cards_group1, red_cards_group2)
    # np.random.shuffle(red_cards_group1)
    # np.random.shuffle(red_cards_group2)
    perturbed_matrix = np.zeros_like(binary_matrix)
    perturbed_matrix[binary_matrix == 1] = np.random.choice([1, 0], size=(binary_matrix == 1).sum(), p=[p, 1 - p])
    perturbed_matrix[binary_matrix == 0] = np.random.choice([1, 0], size=(binary_matrix == 0).sum(), p=[q, 1 - q])

    return perturbed_matrix

def simulate_variance_with_optimal_p_q(N, domain_size, ps, qs, samples, EPOCHS):
    variances = np.zeros(domain_size)
    for i in range(domain_size):
        p, q = ps[i], qs[i]
        estimated_freqs = np.zeros(EPOCHS)
        for epoch in range(EPOCHS):
            binary_vector = (samples == i).astype(int)
            perturbed_vector = perturb_data_without_replacement(N, binary_vector, p, q)
            estimated_freq = (np.sum(perturbed_vector) - N * q) / (p - q)
            estimated_freqs[epoch] = estimated_freq
        variances[i] = np.var(estimated_freqs)
    average_variance = np.mean(variances)
    return average_variance / N

# Parameters and simulation setup
N = 10000
domain_size = 100
epsilon_values = [0.5, 1, 2, 4]
EPOCHS = 100  # Adjusted for execution constraints
zipf_samples_path = "zipf_samples.npy"
samples = np.load(zipf_samples_path)

results = []
for epsilon in epsilon_values:
    ps, qs = [], []
    real_distribution = calculate_real_distribution(samples, domain_size)
    for fi in real_distribution:
        p, q = find_optimal_p_q_for_fi(epsilon, fi, N)
        ps.append(p)
        qs.append(q)
    average_variance_N = simulate_variance_with_optimal_p_q(N, domain_size, ps, qs, samples, EPOCHS)
    results.append(average_variance_N)

# Saving results to a txt file
results_path = "OUE_numerical.txt"
with open(results_path, "w") as file:
    for epsilon, variance in zip(epsilon_values, results):
        file.write(f"Epsilon: {epsilon}, Average Variance / N: {variance}\n")