import numpy as np
from scipy.stats import zipf
import pandas as pd
import random
import matplotlib.pyplot as plt


'''
DE,d=2
'''
N = 10000
lower_bound = 0
domain_size = 2
EPOCHS = 10000
num_of_epsilon = 10
s = 1.1  # Zipf 分布参数

def generate_zipf_integers(s, n_samples, lower_bound, domain_size):
    samples = np.random.zipf(s, n_samples)
    samples = np.clip(samples, lower_bound, domain_size - 1)  # 确保值在 domain_size 范围内
    return samples

def calculate_real_distribution(samples, domain_size):
    real_dist = np.zeros(domain_size)
    for sample in samples:
        real_dist[sample] += 1
    real_dist /= len(samples)
    return real_dist


# 计算 Zipf 分布下各个值的出现概率
probabilities = [zipf.pmf(i, s) for i in range(lower_bound, lower_bound + domain_size)]

# 理论方差计算
epsilon_values = np.linspace(0.5, 5, num_of_epsilon)
# epsilon = np.linspace(1, 1, num_of_epsilon)
# epsilon = [1,2,4]
var_theory = np.zeros(num_of_epsilon)

for i in range(num_of_epsilon):
    # 提取 epsilon 数组中的单个值
    epsilon = epsilon_values[i]

    var_theory[i] = (domain_size-2+np.exp(epsilon)) / (np.exp(epsilon)-1)**2

print("Theoretical Variance:", var_theory)
plt.plot(epsilon_values, np.log10(N*var_theory), color='k', marker ='none', markerfacecolor='none', linewidth=1, markersize=10, linestyle=':', label='DE,d=2')

'''
DE,d=4
'''
N = 10000
lower_bound = 0
domain_size = 4
EPOCHS = 10000
num_of_epsilon = 10
s = 1.1  # Zipf 分布参数

def generate_zipf_integers(s, n_samples, lower_bound, domain_size):
    samples = np.random.zipf(s, n_samples)
    samples = np.clip(samples, lower_bound, domain_size - 1)  # 确保值在 domain_size 范围内
    return samples

def calculate_real_distribution(samples, domain_size):
    real_dist = np.zeros(domain_size)
    for sample in samples:
        real_dist[sample] += 1
    real_dist /= len(samples)
    return real_dist


# 计算 Zipf 分布下各个值的出现概率
probabilities = [zipf.pmf(i, s) for i in range(lower_bound, lower_bound + domain_size)]

# 理论方差计算
epsilon_values = np.linspace(0.5, 5, num_of_epsilon)
# epsilon = np.linspace(1, 1, num_of_epsilon)
# epsilon = [1,2,4]
var_theory = np.zeros(num_of_epsilon)

for i in range(num_of_epsilon):
    # 提取 epsilon 数组中的单个值
    epsilon = epsilon_values[i]

    var_theory[i] = (domain_size-2+np.exp(epsilon)) / (np.exp(epsilon)-1)**2

print("Theoretical Variance:", var_theory)
plt.plot(epsilon_values, np.log10(N*var_theory), color='k', marker ='none', markerfacecolor='none', linewidth=1, markersize=10, linestyle='-.', label='DE,d=4')

'''
DE,d=2
'''
N = 10000
lower_bound = 0
domain_size = 16
EPOCHS = 10000
num_of_epsilon = 10
s = 1.1  # Zipf 分布参数

def generate_zipf_integers(s, n_samples, lower_bound, domain_size):
    samples = np.random.zipf(s, n_samples)
    samples = np.clip(samples, lower_bound, domain_size - 1)  # 确保值在 domain_size 范围内
    return samples

def calculate_real_distribution(samples, domain_size):
    real_dist = np.zeros(domain_size)
    for sample in samples:
        real_dist[sample] += 1
    real_dist /= len(samples)
    return real_dist


# 计算 Zipf 分布下各个值的出现概率
probabilities = [zipf.pmf(i, s) for i in range(lower_bound, lower_bound + domain_size)]

# 理论方差计算
epsilon_values = np.linspace(0.5, 5, num_of_epsilon)
# epsilon = np.linspace(1, 1, num_of_epsilon)
# epsilon = [1,2,4]
var_theory = np.zeros(num_of_epsilon)

for i in range(num_of_epsilon):
    # 提取 epsilon 数组中的单个值
    epsilon = epsilon_values[i]

    var_theory[i] = (domain_size-2+np.exp(epsilon)) / (np.exp(epsilon)-1)**2

print("Theoretical Variance:", var_theory)
plt.plot(epsilon_values, np.log10(N*var_theory), color='k', marker ='none', markerfacecolor='none', linewidth=1, markersize=10, linestyle='--', label='DE,d=16')

'''
DE,d=2
'''
N = 10000
lower_bound = 0
domain_size = 128
EPOCHS = 10000
num_of_epsilon = 10
s = 1.1  # Zipf 分布参数

def generate_zipf_integers(s, n_samples, lower_bound, domain_size):
    samples = np.random.zipf(s, n_samples)
    samples = np.clip(samples, lower_bound, domain_size - 1)  # 确保值在 domain_size 范围内
    return samples

def calculate_real_distribution(samples, domain_size):
    real_dist = np.zeros(domain_size)
    for sample in samples:
        real_dist[sample] += 1
    real_dist /= len(samples)
    return real_dist


# 计算 Zipf 分布下各个值的出现概率
probabilities = [zipf.pmf(i, s) for i in range(lower_bound, lower_bound + domain_size)]

# 理论方差计算
epsilon_values = np.linspace(0.5, 5, num_of_epsilon)
# epsilon = np.linspace(1, 1, num_of_epsilon)
# epsilon = [1,2,4]
var_theory = np.zeros(num_of_epsilon)

for i in range(num_of_epsilon):
    # 提取 epsilon 数组中的单个值
    epsilon = epsilon_values[i]

    var_theory[i] = (domain_size-2+np.exp(epsilon)) / (np.exp(epsilon)-1)**2

print("Theoretical Variance:", var_theory)
plt.plot(epsilon_values, np.log10(N*var_theory), color='k', marker ='none', markerfacecolor='none', linewidth=1, markersize=10, linestyle='-', label='DE,d=128')

'''
DE,d=2
'''
N = 10000
lower_bound = 0
domain_size = 2048
EPOCHS = 10000
num_of_epsilon = 10
s = 1.1  # Zipf 分布参数

def generate_zipf_integers(s, n_samples, lower_bound, domain_size):
    samples = np.random.zipf(s, n_samples)
    samples = np.clip(samples, lower_bound, domain_size - 1)  # 确保值在 domain_size 范围内
    return samples

def calculate_real_distribution(samples, domain_size):
    real_dist = np.zeros(domain_size)
    for sample in samples:
        real_dist[sample] += 1
    real_dist /= len(samples)
    return real_dist


# 计算 Zipf 分布下各个值的出现概率
probabilities = [zipf.pmf(i, s) for i in range(lower_bound, lower_bound + domain_size)]

# 理论方差计算
epsilon_values = np.linspace(0.5, 5, num_of_epsilon)
# epsilon = np.linspace(1, 1, num_of_epsilon)
# epsilon = [1,2,4]
var_theory = np.zeros(num_of_epsilon)

for i in range(num_of_epsilon):
    # 提取 epsilon 数组中的单个值
    epsilon = epsilon_values[i]

    var_theory[i] = (domain_size-2+np.exp(epsilon)) / (np.exp(epsilon)-1)**2

print("Theoretical Variance:", var_theory)
plt.plot(epsilon_values, np.log10(N*var_theory), color='k', marker ='*', markerfacecolor='none', linewidth=1, markersize=10, linestyle='--', label='DE,d=2048')




import numpy as np
from scipy.stats import zipf
import matplotlib.pyplot as plt
import numpy as np

'''
OUE
'''

N = 10000
domain_size = 100
EPOCHS = 10000
# num_of_epsilon = 9
s = 1.1  # Zipf 分布参数


def generate_zipf_integers(s, n_samples, domain_size):
    samples = []
    while len(samples) < n_samples:
        sample = np.random.zipf(s)
        if sample < domain_size:
            samples.append(sample)
    return np.array(samples)


def calculate_real_distribution(samples, domain_size):
    real_dist = np.zeros(domain_size)
    for sample in samples:
        real_dist[sample] += 1
    real_dist /= len(samples)
    return real_dist

samples = generate_zipf_integers(s, N, domain_size)
probabilities_statistical = calculate_real_distribution(samples, domain_size)

# 理论方差计算
# epsilon_values = np.linspace(0.2, 2, num_of_epsilon)
# epsilon = [1,2,4]
p = 1 / 2
q = 1 / (np.exp(epsilon_values) + 1)
var_theory = np.zeros(num_of_epsilon)

for i in range(num_of_epsilon):
    # 提取 epsilon 数组中的单个值
    epsilon_value = epsilon_values[i]
    p = 1 / 2
    q = 1 / (np.exp(epsilon_value) + 1)

    var_theory[i] = sum(fi * p * (1 - p) + (1 - fi) * q * (1 - q) for fi in probabilities_statistical) / (p - q) ** 2 / domain_size

# print("Theoretical Variance:", var_theory)
plt.plot(epsilon_values, np.log10(N*var_theory), color='b', marker ='^', markerfacecolor='none', linewidth=1, markersize=10, linestyle='-', label='OUE')



'''
Simmons
'''
import numpy as np
from scipy.stats import zipf
import matplotlib.pyplot as plt
import numpy as np

N = 10000
domain_size = 100
EPOCHS = 10000
# num_of_epsilon = 1
s = 1.1  # Zipf 分布参数


def generate_zipf_integers(s, n_samples, domain_size):
    samples = []
    while len(samples) < n_samples:
        sample = np.random.zipf(s)
        if sample < domain_size:
            samples.append(sample)
    return np.array(samples)


def calculate_real_distribution(samples, domain_size):
    real_dist = np.zeros(domain_size)
    for sample in samples:
        real_dist[sample] += 1
    real_dist /= len(samples)
    return real_dist

samples = generate_zipf_integers(s, N, domain_size)
probabilities_statistical = calculate_real_distribution(samples, domain_size)

# 理论方差计算
# epsilon = np.linspace(4, 4, num_of_epsilon)
var_theory = np.zeros(num_of_epsilon)

for i in range(num_of_epsilon):
    # 提取 epsilon 数组中的单个值
    epsilon_value = epsilon_values[i]
    x = np.exp(epsilon_value)
    fj1=(1-np.sqrt(2/(np.sqrt(x)+1)))/2
    fj2=(1+np.sqrt(2/(np.sqrt(x)+1)))/2
    var_theory[i] = sum(
        ((1 + 4 * fj * (1 - fj)) / 2 / (N - 1) / (np.sqrt(x) - 1) if fj1 < fj < fj2 else
         (1 / (N - 1) * (2 * np.sqrt(fj * (1 - fj)) / (np.sqrt(x- 1)) + 1 / (x - 1)))
         ) for fj in probabilities_statistical
    ) / domain_size*N

print("Theoretical Variance:", var_theory)
plt.plot(epsilon_values, np.log10(N*var_theory), color='g', marker ='o', markerfacecolor='none', linewidth=1, markersize=10, linestyle='-', label='WORS')


'''
KUK
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from math import exp
from scipy.stats import zipf

# 定义新的 p 关于 q 和 epsilon 的表达式
def p_from_q_epsilon(q, epsilon):
    # 计算 p 的值
    p = q * exp(epsilon) / (1 + q * (exp(epsilon) - 1))
    # 检查 p 是否在 (0, 1) 范围内
    if 0 < p < 1:
        return p
    else:
        return None

# 定义函数来计算 expr 的值
def calculate_expr(epsilon, q, N, probabilities_statistical, domain_size):
    p = p_from_q_epsilon(q, epsilon)
    var_theory = sum(N**2*fi*(fi - 1)*(p**2 - p + q**2 - q)/((N - 1)*(p - q)**2) for fi in probabilities_statistical) / domain_size
    return var_theory

# 定义函数来找到 expr 关于 q 的最小值，并返回最小值及相应的 q
def find_min_expr(epsilon, N):
    result = minimize_scalar(lambda q: calculate_expr(epsilon, q, N, probabilities_statistical, domain_size), bounds=(0, 1), method='bounded')
    return result.fun, result.x

def generate_zipf_integers(s, n_samples, domain_size):
    samples = []
    while len(samples) < n_samples:
        sample = np.random.zipf(s)
        if sample < domain_size:
            samples.append(sample)
    return np.array(samples)

def calculate_real_distribution(samples, domain_size):
    real_dist = np.zeros(domain_size)
    for sample in samples:
        real_dist[sample] += 1
    real_dist /= len(samples)
    return real_dist

EPOCHS = 10000
# num_of_epsilon = 1
s = 1.1  # Zipf distribution parameter
N = 10000
domain_size = 100
samples = generate_zipf_integers(s, N, domain_size)
# 使用统计方法计算概率
probabilities_statistical = calculate_real_distribution(samples, domain_size)
var_theory = np.zeros(num_of_epsilon)
# 参数设置
# epsilon_values = np.linspace(0.5, 0.5, 1)
# epsilon_values = epsilon

# 初始化列表来存储结果
min_expr_values = []
p_values = []
q_values = []

# 计算每个 epsilon 下 expr 的最小值以及相应的 p 和 q 值
for i, epsilon in enumerate(epsilon_values):
    min_expr, q = find_min_expr(epsilon, N)
    p = p_from_q_epsilon(q, epsilon)
    # 使用整数索引 i 而不是 epsilon
    var_theory[i] = min_expr / N
print("Theoretical Variance:", var_theory)
plt.plot(epsilon_values, np.log10(N*var_theory), color='r', marker ='s', markerfacecolor='none', linewidth=1, markersize=10, linestyle='-', label='WORK')
plt.xlabel('epsilon($\epsilon$)')
plt.ylabel('lg(var)')
plt.xlim(0.5, 5)
plt.ylim(1, 8)
plt.xticks(np.linspace(0.5, 5, num=len(epsilon_values)))
plt.title('Comparison of mechanisms')
plt.legend(ncol=2)  # 设置图例为两列
plt.savefig('Comparison of mechanisms.pdf')
plt.show()
