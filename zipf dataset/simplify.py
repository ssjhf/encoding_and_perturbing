from sympy import symbols, simplify

# 定义符号
n, pi_A, p, q = symbols('n pi_A p q')

# 原始公式
formula = (n * (pi_A * p + (1 - pi_A) * q) +
           (n * pi_A * (n * pi_A - 1) * n * p * (n * p - 1)) / ((n - 1) * n) +
           2 * p * q * n * pi_A * (n - n * pi_A) +
           ((n - n * pi_A) * (n - n * pi_A - 1) * n * q * (n * q - 1)) / ((n - 1) * n) -
           n**2 * (pi_A * p + (1 - pi_A) * q)**2) / (n**2 * (p - q)**2)

# 尝试化简公式
simplified_formula = simplify(formula)

print("原始公式：")
print(formula)
print("\n化简后的公式：")
print(simplified_formula)

# 验证两组数值
values_1 = {n: 10000, pi_A: 0.2, p: 0.7, q: 0.4}
values_2 = {n: 100, pi_A: 0.1, p: 0.8, q: 0.3}

result_1 = simplified_formula.subs(values_1).evalf()
result_2 = simplified_formula.subs(values_2).evalf()

print("\n第一组参数结果（n=10000, πA=0.2, p=0.7, q=0.4）:")
print(result_1)

print("\n第二组参数结果（n=100, πA=0.1, p=0.8, q=0.3）:")
print(result_2)