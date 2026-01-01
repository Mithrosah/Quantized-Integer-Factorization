import numpy as np
import matplotlib.pyplot as plt

with open('./logs/result2.txt', 'r') as f:
    lines = f.readlines()

p_nums = []
rule = []
sig = []
det = []
max_steps = []

for line in lines:
    if line[:3] == "Ins":
        p_nums.append(int(line.split()[1][2:-1]))
        max_steps.append(1000*(int(line.split()[2][7:-1]) + int(line.split()[3][7:])))

    elif line[:3] == "[ru":
        rule.append(float(line.split()[-1][13:-2]))
    elif line[:3] == "[si":
        sig.append(float(line.split()[-1][13:-2]))
    elif line[:3] == "[de":
        det.append(float(line.split()[-1][13:-2]))
        

print(np.mean(np.array(rule) / np.array(max_steps)))
print(np.mean(np.array(sig) / np.array(max_steps)))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

plt.figure(figsize=(7, 4.5), dpi=120)

plt.plot(
    p_nums, rule,
    label='quantized',
    color='#1f77b4',
    linewidth=2.0,
    marker='o',
    markersize=5,
)
plt.plot(
    p_nums, sig,
    label='sigmoid',
    color='#d62728',
    linewidth=2.0,
    linestyle='--',
    marker='s',
    markersize=5,
)
plt.plot(
    p_nums, det,
    label='deterministic',
    color='#2ca02c',
    linewidth=2.0,
    linestyle='-.',
    marker='^',
    markersize=6,
)

plt.plot(
    p_nums, max_steps,
    label='failure',
    color='#9467bd',
    linewidth=2.0,
    linestyle=':',
    marker='x',
    markersize=6,
)


plt.xlabel('prime number', fontsize=12)
plt.ylabel('median steps of first hit', fontsize=12)

plt.xlim(100, 1000)
plt.xticks(p_nums)

# log plot
plt.yscale('log')
plt.xscale('log')

plt.grid(True, alpha=0.4)

plt.legend(fontsize=11, frameon=True, framealpha=0.9)
plt.tight_layout()
plt.show()
# plt.savefig('tmp.png')