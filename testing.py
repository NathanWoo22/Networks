import numpy as np
import matplotlib.pyplot as plt
import sys

import generate_plots as gen

def main():
    gen.generate_plots(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == main():
    main()