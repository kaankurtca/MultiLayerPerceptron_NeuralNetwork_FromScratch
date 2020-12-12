import numpy as np
import matplotlib.pyplot as plt
from MLP_Class import CKA
from MLP import MultiP


image1 = np.ones((5, 10))
image1[:, 0], image1[-1, :] = 0, 0  # L harfi oluşturuldu.

image2 = np.ones((5, 10))
image2[0, :], image2[:, 5] = 0, 0  # T harfi oluşturuldu.

image3 = np.ones((5, 10))
for i in range(5):
    image3[i, i], image3[4 - i, 5 + i] = 0, 0  # V harfi oluşturdu.

image4 = np.ones((5, 10))
for j in range(5):
    image4[4 - j, j], image4[j, 5 + j], image4[2, j + 2] = 0, 0, 0  # A harfi oluşturuldu.

images = np.array([image1] + [image2] + [image3] + [
    image4])  # bozuk ve gürültülü hallerini oluşturmak için harf matrisleri birleştirildi.

noisyAll = np.zeros((4, 5, 10))
distordedAll = np.zeros((4, 5, 10))  # gürültülü ve bozuk örüntüleri yerleştireceğimiz matrisler tanımlandı.
for i, image in enumerate(images):
    noisy = image + 0.5 * np.random.rand(5, 10)
    noisy = noisy / noisy.max()
    noisyAll[i] = noisy  # her örüntü için gri seviye gürültü eklendi.
    rand_index = np.random.randint(0, 32)
    distorded = np.array(image)
    distorded[int(rand_index % 4), int(rand_index / 4)] = np.where(
        distorded[int(rand_index % 4), int(rand_index / 4)] == 1, 0.0, 1.0)
    distordedAll[i] = distorded  # her örüntünün bir pikseli bozuldu.

fig, axs = plt.subplots(3, 4)
for i in range(4):
    axs[0, i].imshow(images[i], cmap='gray')
    axs[1, i].imshow(noisyAll[i], cmap='gray')
    axs[2, i].imshow(distordedAll[i], cmap='gray')
plt.show()  # burada, öncelikle eğitim kümemize koyacağımız 12 orjinal,gürültülü ve bozuk örüntüler çizdirildi.

images = np.where(images == 0, 0.1, 0.9)
distordedAll = np.where(distordedAll == 0, 0.1, 0.9)  # matrislerdeki 1->0.9 , 0->0.1 yapıldı

imagesVector = images.reshape(4, 50)
noisyAllVector = noisyAll.reshape(4, 50)
distordedAllVector = distordedAll.reshape(4, 50)  # örüntüler ağa vektör şeklinde verilebilmesi için vektörize edildi.

temp = np.concatenate([imagesVector, noisyAllVector], axis=0)
trainSet = np.concatenate([temp, distordedAllVector],
                          axis=0)  # vektörize edilen örüntüler birleştirldi ve eğitim kümesi oluşturuldu.

testİmages = np.zeros((4, 5, 10))
for i, image in enumerate(images):
    test = image + 0.5 * np.random.rand(5, 10)
    test = test / test.max()
    testİmages[i] = test  # Test için farklı olarak 4 tane gürültülü örüntü oluşturuldu.

testSet = testİmages.reshape(4, 50)  # örüntüler ağa vektör şeklinde verilebilmesi için vektörize edildi.
targetTest=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

# create a dataset to train a network for the sum operation
items = trainSet    # 12x50
targets = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
     [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])   # 12x4
trainAll=np.concatenate([items, targets],axis=1)

# # create a Multilayer Perceptron with one hidden layer
cka = CKA(50, [20, 10], 4)
mlp=MultiP()

mlp.train(items,targets,500)
# cka.egitim(items, targets, 300, 0.4)


for k in range(len(testSet)):


    output=mlp.feedForward(testSet[k])
    print(output)
    print(targetTest[k])


