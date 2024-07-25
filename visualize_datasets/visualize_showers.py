import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold
plt.rcParams['xtick.labelsize'] = 12  # Set x-axis tick label size
plt.rcParams['ytick.labelsize'] = 12  # Set y-axis tick label size
plt.rcParams['font.weight'] = 'bold'  # Make tick labels bold

def main():
    showers = np.load("Epos.npz")
    X = showers['showers']
    mass = X[:, :, 4]
    print(mass.shape)
    X = X[:, :, 1]
    x = range(0, 2000, 10)

    single_mass = []
    for unique_mass in mass:
        single_mass.append(unique_mass[0])

    proton_mass = 0
    iron_mass = math.log(56)
    protons = []
    irons = []
    for i, element in enumerate(single_mass):
        if element == proton_mass and max(X[i]) > 19.4:
            protons.append(i)
        if element == math.log(56) and max(X[i]) > 19.4:
            irons.append(i)
    print(len(protons))
    print(len(irons))
    plt.rcParams['axes.labelweight'] = 'bold'  # Make axis labels bold
    plt.rcParams['xtick.labelsize'] = 12  # Set x-axis tick label size
    plt.rcParams['ytick.labelsize'] = 12  # Set y-axis tick label size
    plt.rcParams['font.weight'] = 'bold'  # Make tick labels bold

    fig, ax = plt.subplots(figsize=(9, 6))
    # Plot protons
    for i in range(25):
        X[protons[i]] = np.array([math.exp(x) for x in X[protons[i]]])
        ax.plot(x, X[protons[i]][:200], linewidth=0.5, color=(66/255, 135/255, 245/255), label='Proton' if i == 0 else '')
        # plt.scatter(x, X[protons[i]][:200], s=0.5, color=(66/255, 135/255, 245/255), label='Proton' if i == 0 else '')

    # Plot irons
    for i in range(25):
        X[irons[i]] = np.array([math.exp(x) for x in X[irons[i]]])
        ax.plot(x, X[irons[i]][:200], linewidth=0.5, color=(230/255, 158/255, 79/255), label='Iron' if i == 0 else '')
        # plt.scatter(x, X[irons[i]][:200], s=0.5, color=(230/255, 158/255, 79/255), label='Iron' if i == 0 else '')

    # Add legend
    ax.legend()
    ax.set_xlabel("Depth "r"$\left[\mathrm{\frac{g}{cm^2}}\right]$", fontsize=15, fontweight="bold")
    ax.set_ylabel("dE/dX "r"$\left[\mathrm{\frac{eV} {(g/cm^2)}}\right]$", fontsize=15, fontweight='bold')
    ax.set_title("Simulated showers profiles for proton and iron primaries", fontweight='bold', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("Visualization_Showers.png")


if __name__ == main():
    main()