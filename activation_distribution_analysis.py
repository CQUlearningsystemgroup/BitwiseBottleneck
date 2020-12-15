#-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import heapq

name = './activations/5_bit/block_layer2_2_0.txt'
content = np.loadtxt(name)


conv0_data = np.array(content, dtype=np.float)

data_abs = np.abs(conv0_data)
max_array = heapq.nlargest(30, data_abs)
print('The most max array: \n', np.reshape(max_array, (1, -1)))

plt.hist(conv0_data, bins=100, density=True)
plt.title(name)
plt.show()
