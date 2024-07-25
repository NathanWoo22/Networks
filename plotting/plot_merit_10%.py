import matplotlib.pyplot as plt
import re
import math
import statistics as stats
import scipy.stats

# percentage = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
fivePercent = [4.9151170219252265, 4.5827214517729615, 5.8257108348449504, 5.068299322924047, 4.996004488479974, 5.600743404298149, 5.002797913261172, 5.178747896920288, 4.882798906540053, 4.637445617587843]
tenPercent = [5.5033442633936005, 4.832877041972458, 5.058185619470652, 4.872792746554865, 4.828422230113828, 5.390253279559623, 4.731927974473826, 4.555874611373827, 5.370934219537565, 4.918669702727316]
twentyPercent = [4.3266127488994215, 4.462511681508537, 4.443048898664363, 4.964699411405564, 4.7448198345420005, 4.443213003293194, 4.672574119670276, 4.588012189611327, 4.726917945153024]
fourtyPercentRaw = [4.566845775268113, 4.295709130401323, 4.523062411362831, 4.4057104789026615, 0.0, 4.455470774240254, 4.699079802720102, 0.023201854509070093, 4.673349482438317, 4.438256237949631]
fourtyPercent = [4.566845775268113, 4.295709130401323, 4.523062411362831, 4.4057104789026615, 4.455470774240254, 4.699079802720102, 4.673349482438317, 4.438256237949631]
end_epoch = [357, 780, 718, 720, 780, 458, 696, 716, 689, 702]
y = [6.070304429532713, stats.mean(fivePercent), stats.mean(tenPercent), stats.mean(twentyPercent), stats.mean(fourtyPercent)]
yerr = [0, scipy.stats.sem(fivePercent), scipy.stats.sem(tenPercent), scipy.stats.sem(twentyPercent), scipy.stats.sem(fourtyPercent)]
x = [0, 5, 10, 20, 40]

# plt.xscale('symlog')
plt.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold
plt.rcParams['xtick.labelsize'] = 12  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 12  # Set y-axis tick label size
plt.rcParams['font.weight'] = 'bold'  # Make tick labels bold

fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as needed

ax.scatter(x, y, s=25)
ax.errorbar(x, y, yerr=yerr, fmt='None', capsize=3)
ax.set_xlabel('Noise ($\sigma$, %)', fontsize=15, fontweight='bold')
ax.set_ylabel('Merit Factor', fontsize=15, fontweight='bold')
ax.set_title('Proton-iron merit factor against percentage noise', fontweight='bold', fontsize=18)

# Adjust the bottom margin to ensure all elements are visible
plt.subplots_adjust(bottom=0.15)

plt.savefig('MF_noise_percent.pdf')
print("DONE!")