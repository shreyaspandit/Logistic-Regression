import numpy as np
import matplotlib.pyplot as plt

class Logistic_Regression(object):
    def __init__(self, X, y, learning_rate, iterations):
        self.X=X
        self.y=y
        self.lr = learning_rate
        self.it = iterations
        self.theta = [] # parameters
        self.cost = []
    
    # appends 1 to start of each input vector to make dot product cleaner
    
    def append_one(self):
        self.X = np.insert(self.X, 0, 1.0, axis=1)
        
    def sigmoid(self, z):
        return (1/(1+np.exp(-z)))
    def loss(self, y, h):
        return ((-y*np.log(h)) - ((1-y)*np.log(1-h))).mean()
    
    def learn(self):
        
        self.theta = np.zeros(self.X.shape[1])
        
        for i in range(self.it):
            
            z = self.X @ self.theta.T
            h = self.sigmoid(z)
            dJ = (self.X.T @ (h - self.y))*(1/self.y.size)
            self.theta -= self.lr*dJ 
            
            self.cost.append(self.loss(self.y, h))
    
        # just applying the above formulae
        
        return self.theta, self.cost
    
    def classify(self, x):
        
        x = np.insert(x, 0, 1.0, axis=0)
        
        print("probability: " + str(self.sigmoid(self.theta @ x.T)))
        
        # note that sigmoid(x) >= 1/2 iff x >= 0, we classify as 1 if prob(y=1|x) = sigmoid(x) >= 1/2
        
        if (self.theta @ x.T) >= 0:
            return 1
        else:
            return 0
