import numpy as np

class MultiP():
    def __init__(self):

        self.w1=0.6*np.random.rand(20,51)-0.3
        self.w2=0.6*np.random.rand(10,21)-0.3
        self.w3=0.6*np.random.rand(4,11)-0.3

    def feedForward(self,X):
        self.X=X    # 1x50
        X=X.reshape(50,1)   # 50x1
        bias_0=np.ones((X.shape[1],1))   # 1x1
        X_bias=np.concatenate([X,bias_0],axis=0)    # 51x1
        self.X_bias=X_bias
        v1=np.dot(self.w1,X_bias)    # 20x1 = [20x51]*[51x1]
        self.v1=v1
        y1= self.sigmoid(v1)        # 20x1
        self.y1=y1
        bias_1=np.ones((y1.shape[1],1))     # 1x1
        y1_bias=np.concatenate([y1,bias_1],axis=0)  #21x1
        self.y1_bias = y1_bias
        v2=np.dot(self.w2,y1_bias)      # 10x1 = [10x21]*[21x1]
        self.v2=v2
        y2=self.sigmoid(v2)
        self.y2=y2
        bias_2=np.ones((y2.shape[1],1))     # 1x1
        y2_bias=np.concatenate([y2,bias_2],axis=0)      # 11x1
        self.y2_bias=y2_bias
        v3=np.dot(self.w3,y2_bias)      # 4x1 = [4x11]*[11x1]
        self.v3=v3
        y3=self.sigmoid(v3)
        self.y3=y3  # 4x1
        return y3

    def backProp(self,error):

        grad_3=error*self.sigmoid_derivative(self.y3)       # 4x1
        self.grad_3 = grad_3
        w3=(self.w3.T)[:-1,:]       # 10x4
        grad_2=np.dot(w3,grad_3)* self.sigmoid_derivative(self.y2)       # 10x1
        self.grad_2 = grad_2
        w2 = (self.w2.T)[:-1, :]    # 20x10
        grad_1 = np.dot(w2, grad_2) * self.sigmoid_derivative(self.y1)      # 20x1
        self.grad_1=grad_1

    def gradDescent(self,lr):
        self.lr=lr

        self.w3 += lr * np.dot(self.grad_3,self.y2_bias.T)      # 4x11
        self.w2 += lr * np.dot(self.grad_2, self.y1_bias.T)     # 10x21
        x=self.X_bias.reshape(1,-1)     # 1x51
        self.w1 += lr * np.dot(self.grad_1, x)                  # 20x51

    def train(self,X,y,epochs):

        for i in range(epochs):
            toplam_error = 0

            for j, inp  in enumerate(X):
                target=y[j].reshape(-1,1)  # 4x1
                print(target)

                out= self.feedForward(inp)
                print(out)

                error=target-out

                self.backProp(error)

                self.gradDescent(0.4)

                toplam_error += self.meanSE(target, out)

            print("Error: {} at epoch {}".format(toplam_error / X.shape[0], i + 1))

    def sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def meanSE(self, hedef, cıkıs):

        return np.average((hedef - cıkıs) ** 2)
