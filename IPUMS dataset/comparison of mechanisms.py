import matplotlib.pyplot as plt
import numpy as np
# Define a function to parse data from the file
N = 2209532
def parse_data(file_path):
    epsilon_values = []
    avg_variance_values = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            epsilon = float(parts[0].split(': ')[1])
            avg_variance = np.log(float(parts[1].split(': ')[1]) / N)
            epsilon_values.append(epsilon)
            avg_variance_values.append(avg_variance)
    return epsilon_values, avg_variance_values

# File paths for theoretical and numerical data
de_theoretical_path = 'DE_theoretical.txt'
oue_theoretical_path = 'OUE_theoretical.txt'
wors_theoretical_path = 'WORS_theoretical.txt'
work_theoretical_path = 'WORK_theoretical.txt'

de_numerical_path = 'DE_numerical.txt'  # Assume these files exist
oue_numerical_path = 'OUE_numerical.txt'
wors_numerical_path = 'WORS_numerical.txt'
work_numerical_path = 'WORK_numerical.txt'

# Parsing the data
de_epsilon, de_avg_variance = parse_data(de_theoretical_path)
oue_epsilon, oue_avg_variance = parse_data(oue_theoretical_path)
wors_epsilon, wors_avg_variance = parse_data(wors_theoretical_path)
work_epsilon, work_avg_variance = parse_data(work_theoretical_path)

de_epsilon_num, de_avg_variance_num = parse_data(de_numerical_path)
oue_epsilon_num, oue_avg_variance_num = parse_data(oue_numerical_path)
wors_epsilon_num, wors_avg_variance_num = parse_data(wors_numerical_path)
work_epsilon_num, work_avg_variance_num = parse_data(work_numerical_path)

# Plotting the data
plt.figure(figsize=(12, 8))

# Theoretical plots
plt.plot(de_epsilon, de_avg_variance, color='k', marker='*', markerfacecolor='none', linewidth=1, markersize=14, linestyle='-', label='DE Theoretical')
plt.plot(oue_epsilon, oue_avg_variance, color='b', marker='^', markerfacecolor='none', linewidth=1, markersize=14, linestyle='-', label='OUE Theoretical')
plt.plot(wors_epsilon, wors_avg_variance, color='g', marker='o', markerfacecolor='none', linewidth=1, markersize=14, linestyle='-', label='WORS Theoretical')
plt.plot(work_epsilon, work_avg_variance, color='r', marker='s', markerfacecolor='none', linewidth=1, markersize=14, linestyle='-', label='WORK Theoretical')

# Numerical plots
plt.plot(de_epsilon_num, de_avg_variance_num, color='k', marker='*', markerfacecolor='none', linewidth=1, markersize=14, linestyle='--', label='DE Numerical')
plt.plot(oue_epsilon_num, oue_avg_variance_num, color='b', marker='^', markerfacecolor='none', linewidth=1, markersize=14, linestyle='--', label='OUE Numerical')
plt.plot(wors_epsilon_num, wors_avg_variance_num, color='g', marker='o', markerfacecolor='none', linewidth=1, markersize=14, linestyle='--', label='WORS Numerical')
plt.plot(work_epsilon_num, work_avg_variance_num, color='r', marker='s', markerfacecolor='none', linewidth=1, markersize=14, linestyle='--', label='WORK Numerical')

plt.title('Comparison of Variances(Theoretical and Numerical)',fontsize=16)
plt.xlabel("Privacy budget ($\epsilon$)",fontsize=16)
plt.ylabel("Logarithm of variance base 10 (lg(var))",fontsize=16)
# Enhance legend and tick font size
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend()
plt.savefig('Comparison of mechanisms for real world dataset.pdf', format='pdf')
plt.show()
