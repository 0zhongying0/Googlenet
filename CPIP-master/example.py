from CPIP import CPIP
import numpy as np
import time
from LSTM import lstm 

# File location
path = "mawilab_300000.tsv" #the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf #the number of packets to process
seq_path = "seq.csv"


# DBN_AutoEncoder params:
maxAE = 10
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 5000 #the number of instances used to train the anomaly detector (ensemble itself)

# LSTM params
LSTMgrace = 5000 #the number of instances used to train the LSTM
seqlen = 50 # the length of sequence needed to predict the
LSTMbatchsize = 5000
# run params
predictgrace = 5000

# Build CPIP
K = CPIP(path,packet_limit,maxAE,FMgrace,ADgrace)
print("Running DBN_AutoEncoder:")
RMSEs = []
i = 0
start = time.time()

# train FM and AD
for i in range(FMgrace + ADgrace):
    if (i + 1) % 1000 == 0:
        print(i + 1)
    rmse = K.proc_next_packet()
    RMSEs.append(rmse)

# generate the sequence needed to train the LSTM
for i in range(seqlen):
    rmse = K.proc_next_packet()
    RMSEs.append(rmse)

print("> Building the LSTM")
model = lstm.build_model([1, 50, 100, 1])
seq_n = 0
train_time = 0
print("> Generating the sequence to train the LSTM")
for i in range(LSTMgrace):
    if (i + 1) % 1000 == 0:
        print(i + 1)
    rmse = K.proc_next_packet()
    RMSEs.append(rmse)
    seq_n += 1
    if seq_n == LSTMbatchsize:
        # saving the sequence
        train_time += 1
        print("> Start Training LSTM for time " + str(train_time) + "...")
        with open(seq_path, "w") as f:
            for rmse in RMSEs[-LSTMbatchsize:]:
                f.write(str(rmse) + "\n")

        print('> Loading data... ')
        x_train, y_train = lstm.load_data(seq_path, seqlen, False)
        print('> Data Loaded. Compiling...')
        model.fit(
            x_train,
            y_train,
            batch_size=512,
            nb_epoch=10,
            validation_split=0.05)
        print("> Finished Training LSTM for time " + str(train_time) + "...")
        seq_n = 0

print("> LSTM training finished")
LSTM_predict_seq = []
# start prediction
for i in range(predictgrace):
    if (i + 1) % 1000 == 0:
        print("predicting " + str(i + 1))

    rmse = K.proc_next_packet()
    RMSEs.append(rmse)
    lstm_result = 0
    if len(LSTM_predict_seq) < seqlen:
        lstm_result = rmse
    else:
        x_to_predict = np.asarray(RMSEs[-seqlen:])
        x_to_predict = np.reshape(x_to_predict, (1, x_to_predict.shape[0], 1))
        lstm_result = lstm.predict_point_by_point(model, x_to_predict)[0]
    LSTM_predict_seq.append(lstm_result)
    # lstm_result = 0
    # if len(predict_seq) < seqlen:
    #     lstm_result = rmse
    # else:
    #     x_to_predict = np.asarray(predict_seq[-seqlen:])
    #     # print(x_to_predict)
    #     x_to_predict = np.reshape(x_to_predict, (1, x_to_predict.shape[0], 1))
    #     lstm_result = lstm.predict_point_by_point(model, x_to_predict)[0]
    #     # print("LSTM predicting result = " + str(lstm_result))
    # predict_seq.append(lstm_result)
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))


# saving the rsmes
with open("rmse.csv", "w") as f:
    for i in RMSEs:
        f.write(str(i) + "\n")

# saving LSTM results
with open("LSTM_result.csv", "w") as f:
    for i in LSTM_predict_seq:
        f.write(str(i) + "\n")


