import matplotlib.pyplot as plt
from numpy import log

from config import TIMINGS_PATH

N, R = [], []
a, b = 1, -9.3

with open(TIMINGS_PATH, 'r') as timings_file:
    line = timings_file.readline()
    while line:
        words = line.split()
        n, timing = int(words[0]), float(words[2])
        R.append(a * log(n) + b)
        N.append(log(n))
        line = timings_file.readline()
        plt.plot(log(n), log(timing), marker='x', color='blue')
plt.plot(N, R, label=r'$ y = x + c $', color='red')
plt.xlabel(r'$\log n$')
plt.ylabel(r'$\log t$')
plt.legend()
plt.title('Temps de segmentation en fonction de '+r'$n=n_x n_y$')
plt.show()
