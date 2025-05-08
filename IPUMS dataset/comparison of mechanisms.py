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
swor_theoretical_path = 'SWOR_theoretical.txt'
kwor_theoretical_path = 'KWOR_theoretical.txt'

de_numerical_path = 'DE_numerical.txt'  # Assume these files exist
oue_numerical_path = 'OUE_numerical.txt'
swor_numerical_path = 'SWOR_numerical.txt'
kwor_numerical_path = 'KWOR_numerical.txt'

# Parsing the data
de_epsilon, de_avg_variance = parse_data(de_theoretical_path)
oue_epsilon, oue_avg_variance = parse_data(oue_theoretical_path)
swor_epsilon, swor_avg_variance = parse_data(swor_theoretical_path)
kwor_epsilon, kwor_avg_variance = parse_data(kwor_theoretical_path)

de_epsilon_num, de_avg_variance_num = parse_data(de_numerical_path)
oue_epsilon_num, oue_avg_variance_num = parse_data(oue_numerical_path)
swor_epsilon_num, swor_avg_variance_num = parse_data(swor_numerical_path)
kwor_epsilon_num, kwor_avg_variance_num = parse_data(kwor_numerical_path)

# Plotting the data
plt.figure(figsize=(12, 8))

# Theoretical plots
plt.plot(de_epsilon, de_avg_variance, color='k', marker='*', markerfacecolor='none', linewidth=1, markersize=18, linestyle='-', label='DE Theoretical')
plt.plot(oue_epsilon, oue_avg_variance, color='b', marker='^', markerfacecolor='none', linewidth=1, markersize=18, linestyle='-', label='OUE Theoretical')
plt.plot(swor_epsilon, swor_avg_variance, color='g', marker='o', markerfacecolor='none', linewidth=1, markersize=18, linestyle='-', label='SWOR Theoretical')
plt.plot(kwor_epsilon, kwor_avg_variance, color='r', marker='s', markerfacecolor='none', linewidth=1, markersize=18, linestyle='-', label='KWOR Theoretical')

# Numerical plots
plt.plot(de_epsilon_num, de_avg_variance_num, color='k', marker='*', markerfacecolor='none', linewidth=1, markersize=18, linestyle='--', label='DE Numerical')
plt.plot(oue_epsilon_num, oue_avg_variance_num, color='b', marker='^', markerfacecolor='none', linewidth=1, markersize=18, linestyle='--', label='OUE Numerical')
plt.plot(swor_epsilon_num, swor_avg_variance_num, color='g', marker='o', markerfacecolor='none', linewidth=1, markersize=18, linestyle='--', label='SWOR Numerical')
plt.plot(kwor_epsilon_num, kwor_avg_variance_num, color='r', marker='s', markerfacecolor='none', linewidth=1, markersize=18, linestyle='--', label='KWOR Numerical')

plt.title('Comparison of Variances(Theoretical and Numerical)',fontsize=20)
plt.xlabel("Privacy budget ($\epsilon$)",fontsize=20)
plt.ylabel("Logarithm of variance base 10 (lg(var))",fontsize=20)
# Enhance legend and tick font size
plt.legend(fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(fontsize=18)
plt.savefig('Comparison of mechanisms for IPUMS dataset.pdf', format='pdf')
plt.show()
