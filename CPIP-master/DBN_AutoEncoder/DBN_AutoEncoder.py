import numpy as np
import DBN_AutoEncoder.dA as AE
import DBN_AutoEncoder.corClust as CC
from DBN_AutoEncoder.dbn import *
from sklearn.decomposition import PCA

class DBN_AutoEncoder:
    #n: the number of features in your input dataset (i.e., x \in R^n)
    #m: the maximum size of any autoencoder in the ensemble layer
    #AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    #FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period
    #learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the DBN_AutoEncoder instance.
    #hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    #feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
    #           where the i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
    #           For example, [[2,5,3],[4,0,1],[6,7]]
    def __init__(self,n,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75, feature_map = None):
        # Parameters:
        self.AD_grace_period = AD_grace_period
        if FM_grace_period is None:
            self.FM_grace_period = AD_grace_period
        else:
            self.FM_grace_period = FM_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n
        # Variables
        self.n_trained = 0 # the number of training instances so far
        self.n_executed = 0 # the number of executed instances so far
        self.v = feature_map
        if self.v is None:
            print("Feature-Mapper: train-mode, Anomaly-Detector: off-mode")
        else:
            self.__createAD__()
            print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")

        # use DBN instead of CC & ensembleLayer
        sizes = [int(self.n), int(self.n*0.6), int(self.n*0.4), int(self.n*0.15)]
        numepochs = 30
        self.FM = dbn(sizes, 0.06, numepochs)
        self.output_size = self.FM.sizes[-1]


        self.train_x = [] # the data used to train the DBN
        self.train_x_n = 0
        self.outputLayer = None

    #If FM_grace_period+AM_grace_period has passed, then this function executes DBN_AutoEncoder on x. Otherwise, this function learns from x.
    #x: a numpy array of length n
    #Note: DBN_AutoEncoder automatically performs 0-1 normalization on all attributes.
    def process(self,x):
        if self.n_trained > self.FM_grace_period + self.AD_grace_period: #If both the FM and AD are in execute-mode
            return self.execute(x)
        else:
            self.train(x)
            return 0.0

    #force train DBN_AutoEncoder on x
    #returns the anomaly score of x during training (do not use for alerting)
    def train(self,x):
        if self.n_trained <= self.FM_grace_period and self.v is None: #If the FM is in train-mode, and the user has not supplied a feature mapping
            #update the incremetnal correlation matrix
            # train_x = np.asarray([x])
            # self.FM.train(train_x)
            self.train_x.append(x)
            self.train_x_n += 1
            if self.train_x_n == 5000:
                self.train_x = np.asarray(self.train_x)
                self.FM.train(self.train_x)
                self.train_x = []
                self.train_x_n = 0

            if self.n_trained == self.FM_grace_period: #If the feature mapping should be instantiated
                self.v = self.FM
                self.__createAD__()
                # DBN_output_size = self.FM.hidden_layers_structure[-1]
                print("The DBN found a mapping: "+str(self.n)+" statics to "+str(self.output_size)+" features.")
                print("DBN: execute-mode, Anomaly-Detector: train-mode")
        else: #train
            S_l1 = self.FM.predict(x).reshape(-1,self.v.sizes[-1]).getA()[0]
            self.outputLayer.train(S_l1)
            if self.n_trained == self.AD_grace_period+self.FM_grace_period:
                print("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode")
        self.n_trained += 1

    #force execute DBN_AutoEncoder on x
    def execute(self,x):
        if self.v is None:
            raise RuntimeError('DBN_AutoEncoder Cannot execute x, because a feature mapping has not yet been learned or provided. Try running process(x) instead.')
        else:
            self.n_executed += 1
            S_l1 = self.FM.predict(x).reshape(-1,self.v.sizes[-1]).getA()[0]
            ## OutputLayer
            return self.outputLayer.execute(S_l1)

    def __createAD__(self):
        params = AE.dA_params(self.output_size, n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr)
        self.outputLayer = AE.dA(params)
