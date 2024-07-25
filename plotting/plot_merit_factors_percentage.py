import matplotlib.pyplot as plt
import re

mu = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
sigma = [0, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
percentage = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20.0, 22.5, 25.0]
mf = [5.698993731349619, 4.943527486443177, 5.053781950451939, 5.212235022909994, 5.325521507150884, 5.644889710326008, 4.958969355265876, 4.9932821508725524, 4.385351736655471, 4.660196427628206, 4.896285201709306]

plt.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold
plt.rcParams['xtick.labelsize'] = 12  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 12  # Set y-axis tick label size
plt.rcParams['font.weight'] = 'bold'  # Make tick labels bold

fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed

ax.scatter(percentage, mf)
ax.set_xlabel('Noise ($\sigma$)', fontsize=15, fontweight='bold')
ax.set_ylabel('Merit Factor', fontsize=15, fontweight='bold')
ax.set_title('Proton-iron merit factor against percentage noise', fontweight='bold', fontsize=18)

# Adjust the bottom margin to ensure all elements are visible
plt.subplots_adjust(bottom=0.15)

plt.savefig('MF_noise_percent.png')
print("DONE!")