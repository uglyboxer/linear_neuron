import numpy as np
from matplotlib import pyplot as plt
from network import Network
from matplotlib import cm


x = np.random.normal(1, 1, (8, 8))
plt.imshow(x, cmap = cm.Greys_r)
plt.show()

print(x)