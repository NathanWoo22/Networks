import matplotlib.pyplot as plt
import re
import matplotlib as mpl

# mpl.rcParams['font.weight'] = 'bold'

mu = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
sigma = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
end_epoch = [357, 780, 779, ]
mf = [9.57459261290645, 6.070304429532713, 4.9936576166730395, 5.965806722166998, 4.85470672919648, 4.905222715917048, 3.6166116283481275, 3.37838633140914, 3.1971718493052284, 3.536908480071909]

plt.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold
plt.rcParams['xtick.labelsize'] = 12  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 12  # Set y-axis tick label size
plt.rcParams['font.weight'] = 'bold'  # Make tick labels bold

fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed

ax.scatter(sigma, mf, s=50)
ax.set_xlabel('Noise ($\sigma$)', fontsize=15, fontweight='bold')
ax.set_ylabel('Merit Factor', fontsize=15, fontweight='bold')
ax.set_title('Proton-iron merit factor against baseline noise', fontweight='bold', fontsize=18)

# Adjust the bottom margin to ensure all elements are visible
plt.subplots_adjust(bottom=0.15)

plt.savefig('MF_noise_baseline.pdf')
print("DONE!")