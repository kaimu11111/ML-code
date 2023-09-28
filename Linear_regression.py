import numpy as np
class Linear_regression:
    def __init__(self,data,labels):
        self.standardize_data = Linear_regression.standardize(data)
        self.data = data
        self.labels = labels
        self.theta = np.random.rand((self.data.shape[1],1))

    def train(self,ETA,num_iteration = 500):
        cost_history = self.gradient_descend(ETA,num_iteration)
        return self.theta,cost_history
    
    def gradient_descend(self,ETA,num_iteration = 500):
        cost_history = []
        for _ in range(num_iteration):
           self.gradient_step(ETA)
           cost_history.append(self.cost_function(self.data,self.labels))
        return
    def cost_funtion(self,data,labels):
        num_example = data.shape[0]
        delta = Linear_regression.hypothesis(self.data,self.theta) - labels
        cost = (0.5 * np.dot(delta.T,delta))/num_example
        return cost[0][0]
    def gradient_step(self,ETA):
        num_example = self.data.shape[0]
        prediction = Linear_regression.hypothesis(self.data,self.theta)
        theta = self.theta
        theta = theta - ETA * (1/num_example) * (np.dot((prediction - self.labels),self.data))
        self.theta = theta
        return theta

    @staticmethod
    def hypothesis(data,theta):
        return np.dot(data,theta)


    @staticmethod
    def standardize(self,data):
        mu = np.mean(data)
        std = np.std(data)
        return (data - mu) / std    