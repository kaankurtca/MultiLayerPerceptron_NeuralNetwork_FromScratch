import numpy as np
import matplotlib.pyplot as plt

image1 = np.ones((5, 10))
image1[:, 0] = 0
image1[-1, :] = 0
noisy1_tum=np.zeros((4,5,10))
for ind in range(4):
    noisy1 = image1 + 0.5 * np.random.rand(5, 10)
    noisy1 = noisy1/noisy1.max()
    noisy1_tum[ind] = noisy1


image2 = np.ones((5, 10))
image2[0, :] = 0
image2[:, 5] = 0
noisy2_tum=np.zeros((4,5,10))
for ind in range(4):
    noisy2 = image2 + 0.5 * np.random.rand(5, 10)
    noisy2 = noisy2/noisy2.max()
    noisy2_tum[ind]=noisy2


image3 = np.ones((5, 10))
for i in range(5):
    image3[i,i]=0
    image3[4-i,5+i]=0
noisy3_tum = np.zeros((4, 5, 10))
for ind in range(4):
    noisy3 = image3 + 0.5 * np.random.rand(5, 10)
    noisy3 = noisy3/noisy3.max()
    noisy3_tum[ind]=noisy3


image4 = np.ones((5, 10))
for j in range(5):
    image4[4-j, j] = 0
    image4[j, 5+j] = 0
    image4[2,j+2]=0
noisy4_tum = np.zeros((4, 5, 10))
for ind in range(4):
    noisy4 = image4 + 0.5 * np.random.rand(5, 10)
    noisy4 = noisy4/noisy4.max()
    noisy4_tum[ind] = noisy4


fig, axs = plt.subplots(4,4)
for i in range(4):
    axs[0,i].imshow(noisy1_tum[i], cmap='gray')
    axs[1,i].imshow(noisy2_tum[i], cmap='gray')
    axs[2, i].imshow(noisy3_tum[i], cmap='gray')
    axs[3, i].imshow(noisy4_tum[i], cmap='gray')
plt.show()