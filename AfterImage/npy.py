import numpy as np
import FeatureExtractor as FE
import copy
import csv

if __name__ == "__main__":
    # Transfer from the .tsv file to the .npy file
    # features = FE.FE("/Users/chenyifan/Downloads/CPIP-master/mawilab_300000.tsv")
    features = FE.FE("/Users/chenyifan/Downloads/201806081400.pcap")
    result = []
    result = np.array(result)
    label = []
    for i in range(200):
        for j in range(10):
            result = np.hstack((result,features.get_next_vector()))
        result = np.hstack((result,np.array([0]*25)))
    s = []
    for i in range(15):
        s.append(i)
    result = np.hstack((result,s*15))
    result = result.reshape(1801,15,15)
    np.save("result.npy",result)
    print(result)
    print("Image count:",len(np.load("result.npy")))
    
    tsv_file = csv.reader(open("/Users/chenyifan/Downloads/CPIP-master/results/mawilab_300000_labeled.tsv"))
    for row in tsv_file:
        # Noraml labels are marked by 0 and anolomous labels are marked by 1
        if row[-1] == "normal":
            label.append(0)
        else:
            label.append(1)
    label.pop(0)
    label = label[0:1801]
    print("Label count:",len(label))
    np.save("label.npy",label)
    # print(np.load("label.npy"))