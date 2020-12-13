import numpy as np
from sklearn.datasets import load_iris
from MLP import MultiP
from numpy import genfromtxt

my_data = genfromtxt('Iris.csv', delimiter=',')     #İris veriseti numpy array'ine dönüştürüldü.
dataset_X=my_data[1:,1:5]       # verisetinin index ve sınıf bilgisi kısımlarını ayıkladık. Feature'lar kaldı.  #150x4 array
dataset_y=np.zeros((150, 3))
dataset_y[:50, 0], dataset_y[50:100, 1], dataset_y[100:, 2] = 1, 1, 1   # Setosa->1 0 0, Versicolor->0 1 0, Virginica->0 0 1  # 150x3 array
dataset=np.concatenate([dataset_X,dataset_y],axis=1)                    # giriş ve çıkışlar birleştirildi.  # 150x7 array

randomIndex_1=np.random.choice(50,40,replace=False).reshape(1,40)
randomIndex_2=50+np.random.choice(50,40,replace=False).reshape(1,40)
randomIndex_3=100+np.random.choice(50,40,replace=False).reshape(1,40)   # Her sınıftan rastgele 40 örnek seçilecek şekilde indis seçildi
temp=np.concatenate([randomIndex_1,randomIndex_2],axis=0)
trainSet_Index=np.concatenate([temp,randomIndex_3],axis=0).reshape(120,) # Seçilen indisler birleştirildi, bir arrayde tutuldu.

trainSet=dataset[trainSet_Index,:]  #her sınıftan rastgele 40'ar tane seçilerek, 120 veriden oluşan eğitim kümesi oluşturuldu.  # 120x7
np.random.shuffle(trainSet)         #karıştırıldı


# for i in range(trainSet.shape[1]):
#     for j in range(trainSet.shape[0]):
#          trainSet[j,i] = (trainSet[j,i]- np.min(trainSet[:,i])) / (np.max(trainSet[:,i])- np.min(trainSet[:,i]))

trainSet_X=trainSet[:,:4]   # İlk 4 sütun girişler
trainSet_y=trainSet[:,-3:]  # Son 3 sütun sınıf bilgisi

mlp=MultiP(4,30,20,3)       # Class çağrıldı (giriş 4 boyutlu,gizli katmanlar 32'şer nöron, çıkış 3 boyutlu.)


mlp.train(trainSet_X,trainSet_y,500)  #eğitim yapıldı.


testSet=np.delete(dataset,trainSet_Index,axis=0)    # Test kümesi oluşturuldu.  # 30x7 array
np.random.shuffle(testSet)                          #karıştırıldı.
testSet_X=testSet[:,:4]                             # İlk 4 sütun girişler
testSet_y=testSet[:,-3:]                            # Son 3 sütun sınıf bilgisi


testMSE=0
print("\n\n", "Tahmin edilen değerler ve test verileri: ", "\n")
for k in range(len(testSet)):
    output = np.around(mlp.feedForward(testSet_X[k]).reshape(-1, 1),3)  # tahmin edilen çıkış
    testDesired = np.around((testSet_y[k].reshape(-1, 1)),3)  # arzu edilen çıkış
    outAndDesired = np.concatenate([output, testDesired], axis=1)  # karşılaştırma daha iyi gözlemlenmesi için birleştirilerek yazdırıldı.
    print("\n\n", outAndDesired, "\n\n")
    error=mlp.meanSE(testDesired,output)
    testMSE += error
print("Test ortalama kare hatası: ",testMSE/len(testSet))
