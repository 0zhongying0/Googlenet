from AfterImage.FeatureExtractor import *
from DBN_AutoEncoder.DBN_AutoEncoder import DBN_AutoEncoder
class CPIP:
    def __init__(self,file_path,limit,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75,):
        #init packet feature extractor (AfterImage)
        self.FE = FE(file_path,limit)

        #init DBN_AutoEncoder
        self.AnomDetector = DBN_AutoEncoder(self.FE.get_num_features(),max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)

    def proc_next_packet(self):
        # create feature vector
        x = self.FE.get_next_vector()
        if len(x) == 0:
            return -1 #Error or no packets left

        # process DBN_AutoEncoder
        return self.AnomDetector.process(x)  # will train during the grace periods, then execute on all the rest.

