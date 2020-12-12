import numpy as np




class CKA(object):

    def __init__(self, giris_sayisi=50, gizli_katmanlar=[20, 10], çıktı_Sayisi=4):
        # giris sayısı  5x10=50
        # gizli katman sayısı gizli_katmanlar listesinin eleman sayısı kadar listenin içindeki değerler de her gizli katmandaki
        # nöron sayısını belirtiyor
        # 4 sınıf olduğu için çıkış katmanına 2 nöron koyduk

        self.giris_sayisi = giris_sayisi
        self.gizli_katmanlar = gizli_katmanlar
        self.çıktı_Sayisi = çıktı_Sayisi

        # katmanları ve katmanlardaki nöron sayısını tek bir listede tutuyoruz.
        katmanlar = [giris_sayisi] + gizli_katmanlar + [çıktı_Sayisi]

        # başlangıç ağırlıkları
        agirliklar = []
        for i in range(len(katmanlar) - 1):  # ağırlık matris sayısı katman sayısından 1 eksik
            w = 0.6*np.random.rand(katmanlar[i]+1, katmanlar[i + 1])-0.3
            agirliklar.append(w)
        self.agirliklar = agirliklar

        # hata fonksiyonunun her ağırlığa göre türevini tutmak için
        türevler = []
        for i in range(len(katmanlar) - 1):
            d = np.zeros((katmanlar[i], katmanlar[i + 1]))
            türevler.append(d)
        self.türevler = türevler

        # her katmanın çıkışındaki y değerleri
        y_degerleri = []
        for i in range(len(katmanlar)):
            a = np.zeros(katmanlar[i])
            y_degerleri.append(a)
        self.y_degerleri = y_degerleri

        v_degerleri = []
        for i in range(len(katmanlar)):
            a = np.zeros(katmanlar[i])
            v_degerleri.append(a)
        self.v_degerleri = v_degerleri

    def ileri_yol_agi(self, girisler):


        # giris katmanı için girişler aynı zamanda y değerleri olduğu için
        y_degerleri = girisler.reshape(1,50)

        # geriye yayılım için y değerlerini tutuyoruz.
        self.y_degerleri[0] = y_degerleri

        # katmanlar boyunca ileri yolda ilerliyoruz.
        for i, w in enumerate(self.agirliklar):
            # y değerleri her bir katmandaki nöronun girişi olacağı için bu değerleri ağırlık matrisleriyle çarpıyoruz
            bias=np.ones((y_degerleri.shape[0],1))
            y_bias=np.concatenate([y_degerleri,bias],axis=1)
            v_değerleri = np.dot(y_bias, w)

            # v değerlerine sigmoid aktivasyon fonksiyonu uygulayarak yeni y değerlerini buluyoruz
            y_degerleri = self._sigmoid(v_değerleri)
            # geriye yayılım için y değerlerini tutuyoruz.
            self.y_degerleri[i + 1] = y_degerleri

        # çıkıştaki y değerleri
        return y_degerleri

    def geriye_yayılım(self, error):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # geriye doğru yayılım için reversed ifadesini kullandık.
        for i in reversed(range(len(self.türevler))):
            # sigmoid fonksiyonun türevi s'(x)=s(x)*(1-s(x)) olduğu için
            # yerel gradyen hesabında s(v_degerleri)=y_degerleri yazabildik

            # bir önceki katman y_degerlerini aldık
            y_degerleri = self.y_degerleri[i + 1]

            yerel_gradyenler = error * self._sigmoid_derivative(y_degerleri)

            yerel_gradyenler_re = yerel_gradyenler.reshape(yerel_gradyenler.shape[0], -1).T

            # get activations for current layer
            yi_degerleri = self.y_degerleri[i]

            # reshape activations as to have them as a 2d column matrix
            yi_degerleri = yi_degerleri.reshape(yi_degerleri.shape[0], -1)

            # save derivative after applying matrix multiplication
            self.türevler[i] = np.dot(yi_degerleri.T, yerel_gradyenler_re.T)

            # bir önceki katmandaki yerel gradyenler ile bir sonraki katmandaki yerel gradyenler arasındaki ilişki
            ww=self.agirliklar[i].T
            error = np.dot(yerel_gradyenler, ww[:,:-1])

    def egitim(self, girisler, hedefler, epochs, ögrenme_hızı):
        """Trains model running forward prop and backprop
        Args:
            inputs (ndarray): X
            targets (ndarray): Y
            epochs (int): Num. epochs we want to train the network for
            learning_rate (float): Step to apply to gradient descent
        """

        for i in range(epochs):
            toplam_error = 0

            for j, giris in enumerate(girisler):
                hedef = hedefler[j]
                print(hedef)

                cıkıs = self.ileri_yol_agi(giris)
                print(cıkıs)

                error = hedef - cıkıs

                self.geriye_yayılım(error)

                # Gradient descent'i uygulayarak ağırlıkları güncelliyoruz

                self.gradient_descent(ögrenme_hızı)

                # ortalama karesel hatayı hesaplıyoruz
                toplam_error += self._okh(hedef, cıkıs)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(toplam_error / len(girisler), i + 1))

        print("Training complete!")
        print("=====")

    def gradient_descent(self, ögrenme_hızı=0.21):

        # update the weights by stepping down the gradient
        for i in range(len(self.agirliklar)):
            tempW = self.agirliklar[i]
            agirliklar = tempW[:-1,:]
            türevler = self.türevler[i]
            agirliklar += türevler * ögrenme_hızı

    def _sigmoid(self, x):

        y = 1.0 / (1 + np.exp(-x))
        return y

    def _sigmoid_derivative(self, x):

        return x * (1.0 - x)

    def _okh(self, hedef, cıkıs):

        return np.average((hedef - cıkıs) ** 2)