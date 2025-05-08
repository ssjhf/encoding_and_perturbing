import numpy as np
from scipy.stats import zipf

# Constants
N_SAMPLES = 10000
DOMAIN_SIZE = 100
EPOCHS = 10000
NUM_OF_EPSILON = 4
ZIPF_PARAM = 1.1  # Parameter for the Zipf distribution


def calculate_real_distribution(samples, domain_size):
    """Calculate the actual distribution of the samples."""
    distribution = np.zeros(domain_size)
    for sample in samples:
        distribution[sample] += 1
    return distribution / len(samples)

# Generate samples and calculate statistical probabilities
N = 10000
domain_size = 100
epsilon_values = [0.5, 1, 2, 4]
zipf_samples_path = "zipf_samples.npy"
samples = np.load(zipf_samples_path)
statistical_probabilities = calculate_real_distribution(samples, DOMAIN_SIZE)

# Theoretical variance calculation
theoretical_variance = np.zeros(NUM_OF_EPSILON)

for i, epsilon_value in enumerate(epsilon_values):
    x = np.exp(epsilon_value)
    fj1 = (1 - np.sqrt(2 / (np.sqrt(x) + 1))) / 2
    fj2 = (1 + np.sqrt(2 / (np.sqrt(x) + 1))) / 2
    theoretical_variance[i] = sum(
        ((1 + 4 * fj * (1 - fj)) / 2 / (N_SAMPLES - 1) / (np.sqrt(x) - 1) if fj1 < fj < fj2 else
         (1 / (N_SAMPLES - 1) * (2 * np.sqrt(fj * (1 - fj)) / (np.sqrt(x - 1)) + 1 / (x - 1)))
         ) for fj in statistical_probabilities
    ) / DOMAIN_SIZE * N_SAMPLES

print("Theoretical Variance:", theoretical_variance)

# Write results to file
with open('SWOR_theoretical.txt', 'w') as file:
    for i in range(len(epsilon_values)):
        file.write(f"{epsilon_values[i]}\t{theoretical_variance[i]}\n")

print("Data written to file.")
