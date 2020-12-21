import numpy as np
import matplotlib.pyplot as plt
from MLP import MultiP

e=0.3*np.random.rand(150)    # Billing Sistemine eklenecek olan gürültü vektörü
y=np.zeros(150)
y[:2]=0.3*np.random.rand(2)               # Billing Sistemiminin ilk iki değerleini (y[0],y[1]) başlangıçta sıfır olarak ayarlandı.

for k in range(2,150):
    y[k] = (0.8-(0.5*np.exp(-y[k-1]**2)))*y[k-1]-(0.3+(0.9*np.exp(-y[k-1]**2)))*y[k-2] + (0.1*np.sin(np.pi*y[k-1]))+e[k]
    # Sistemin formülüne göre çıkışlar oluşturuldu.

firstFeatureTrain = y[:100].reshape(-1, 1)          # y[k-2]
secondFeatureTrain = y[1:101].reshape(-1, 1)        # y[k-1]
thirdFeatureTrain = e[2:102].reshape(-1, 1)         # e[k]
# Eğitim kümesinin sütunları(feature) oluşturuldu.

temp1=np.concatenate([firstFeatureTrain, secondFeatureTrain], axis=1)
trainSet_X=np.concatenate([temp1, thirdFeatureTrain], axis=1)   # Eğitim kümesi girdileri oluşturuldu.

trainSet_y = y[2:102].reshape(-1,1)      # Eğitim kümesi çıktıları oluşturuldu. #y[k]

mlp=MultiP(3,8,4,1,"tanh")                        # Sınıf çağrıldı. Değerler -1,1 aralığında değiştiği için Aktivasyon fonksiyonu "tanh" olarak seçildi.

mlp.train(trainSet_X,trainSet_y,300,0.5)  #eğitim yapıldı.

firstFeatureTest = y[100:148].reshape(-1, 1)    # y[k-2]
secondFeatureTest = y[101:149].reshape(-1, 1)   # y[k-1]
thirdFeatureTest = e[102:150].reshape(-1, 1)    # e[k]

temp2=np.concatenate([firstFeatureTest, secondFeatureTest], axis=1)
testSet_X=np.concatenate([temp2, thirdFeatureTest], axis=1)     # Test kümesi girdileri oluşturuldu.

testSet_y = y[102:150].reshape(-1,1)                            # Test kümesi çıktıları oluşturuldu. #y[k]

testMSE=0
outputs=np.zeros(len(testSet_y))
for k in range(len(testSet_X)):
    output = np.around(mlp.feedForward(testSet_X[k]).reshape(-1, 1), 3)  # tahmin edilen çıkış
    outputs[k]=output
    testDesired = np.around((testSet_y[k].reshape(-1, 1)), 3)  # arzu edilen çıkış
    error = mlp.meanSE(testDesired, output)
    testMSE += error
print("Test ortalama kare hatası: ",testMSE/len(testSet_X))

plt.figure()
plt.scatter(outputs[:-1],outputs[1:],label="Tahminler",color='r')
plt.scatter(testSet_y[:-1],testSet_y[1:],label="Gerçek Değerler",color='b')
plt.legend(loc="upper left")
plt.xlabel("y[k-1]")
plt.ylabel("y[k]")
plt.title("Durum Uzayı Karşılaştırması")

plt.figure()
plt.scatter(range(len(outputs)),outputs,label="Tahminler",color='r')
plt.plot(range(len(testSet_y)),testSet_y,label="Gerçek Değerler",color='b')
plt.legend(loc="lower left")

plt.ylabel("Sistem Çıkış Değerleri")
plt.title("Gerçek ve Tahmin Değerleri Karşılaştırması")

plt.show()