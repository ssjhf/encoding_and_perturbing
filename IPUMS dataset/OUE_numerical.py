# Correcting the provided code for epsilon values [0.5, 1, 2, 4] and saving results to a txt file.

import numpy as np
from scipy.optimize import minimize_scalar
from math import exp
from scipy.sparse import csr_matrix

def read_data_to_array(filename):
    with open(filename, 'r') as file:
        # 读取数据到列表，同时过滤掉999999和999998
        data = [int(num) for line in file for num in line.split() if int(num) not in [9999999, 9999998]]
    return np.array(data)

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


def perturb_data_without_replacement(N, binary_vector, p, q):
    # 确保 binary_vector 是一个稀疏矩阵
    indices = binary_vector.nonzero()[1]  # 获取非零元素的索引，适用于二维稀疏矩阵
    # 生成扰动后的数据
    perturbed_data = np.random.choice([1, 0], size=indices.size, p=[p, 1 - p])

    # 构建扰动后的稀疏向量
    perturbed_vector = csr_matrix((perturbed_data, (np.zeros(indices.size), indices)), shape=(1, N))

    # 处理原向量中为0的元素
    non_indices_size = N - indices.size
    if non_indices_size > 0:
        non_perturbed_data = np.random.choice([1, 0], size=non_indices_size, p=[q, 1 - q])
        non_indices = np.setdiff1d(np.arange(N), indices, assume_unique=True)
        non_indices = non_indices[non_perturbed_data == 1]
        non_data = np.ones(non_indices.size)

        # 合并原有和新增的扰动数据
        all_data = np.concatenate([perturbed_data, non_data])
        all_indices = np.concatenate([indices, non_indices])
        perturbed_vector = csr_matrix((all_data, (np.zeros(all_indices.size), all_indices)), shape=(1, N))

    return perturbed_vector

def simulate_variance_with_optimal_p_q(N, domain_size, ps, qs, samples, EPOCHS):
    variances = np.zeros(domain_size)
    for i in range(domain_size):
        p, q = ps[i], qs[i]
        estimated_freqs = np.zeros(EPOCHS)
        for epoch in range(EPOCHS):
            print(epoch)
            # 将binary_vector转换为稀疏矩阵格式
            binary_vector = csr_matrix((samples == i).astype(int).reshape(1, -1))
            perturbed_vector = perturb_data_without_replacement(N, binary_vector, p, q)
            estimated_freq = (np.sum(perturbed_vector) - N * q) / (p - q)
            estimated_freqs[epoch] = estimated_freq
        variances[i] = np.var(estimated_freqs)
    average_variance = np.mean(variances)
    return average_variance / N

# Parameters and simulation setup
filename = 'usa_00010.dat'
epsilon_values = np.linspace(0.5, 5, 1)
sample = read_data_to_array(filename)
samples = (sample + 20000) // 1000
N_SAMPLES = len(samples)
DOMAIN_SIZE = max(samples) + 1
NUM_OF_EPSILON = len(epsilon_values)
EPOCHS = 10

results = []
for epsilon in epsilon_values:
    ps, qs = [], []
    real_distribution = calculate_real_distribution(samples, DOMAIN_SIZE)
    for fi in real_distribution:
        p, q = find_optimal_p_q_for_fi(epsilon, fi, N_SAMPLES)
        ps.append(p)
        qs.append(q)
    average_variance_N = simulate_variance_with_optimal_p_q(N_SAMPLES, DOMAIN_SIZE, ps, qs, samples, EPOCHS)
    results.append(average_variance_N)

# Saving results to a txt file
results_path = "OUE_numerical.txt"
with open(results_path, "w") as file:
    for epsilon, variance in zip(epsilon_values, results):
        file.write(f"Epsilon: {epsilon}, Average Variance / N: {variance}\n")