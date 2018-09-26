import pandas as pd
import numpy as np
from math import exp
import copy
from pandas import DataFrame
from sklearn import linear_model, datasets, metrics

LSTMPATH = "LSTM_result.csv"
LABELPATH = "../mawilab_300000_labeled.tsv"
RMSEPATH = "rmse.csv"

# size of the lists
predict_size = 20000

# params for the p-fomula
ps = [(x / 20) for x in range(10, 19)]
fixnormal_ratio = 0.65
fixp = 0.8
normal_ratios = [(x * 0.03 + 0.5) for x in range(11)]


def p_fomula(rmse, lstm_state, p):
	r1 = exp(rmse)
	r2 = exp(sum(lstm_state))
	result = p * r1 + (1 - p) * r2
	return result


def get_index(labels, predict_labels):
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(len(labels)):
		if(labels[i] == True and predict_labels[i] == True):
			TP += 1
		elif(labels[i] == True and predict_labels[i] == False):
			FN += 1
		elif(labels[i] == False and predict_labels[i] == True):
			FP += 1
		elif(labels[i] == False and predict_labels[i] == False):
			TN += 1
	R = TP / (TP + FN)
	P = TP / (TP + FP)
	F = 2 * P * R / (P + R)
	return P, R, F


def report_precision(rmses, lstms, labels, p, normal_ratio):
	print("> Reporting Precision for values : p = " + str(p) + " normal_ratio : " + str(normal_ratio))
	results = [labels[0], labels[1]]
	for i in range(2, predict_size):
		lstm_state = lstms[i-2:i]
		rmse = rmses[i]
		results.append(p_fomula(rmse, lstm_state, p))


	sort_results = copy.deepcopy(results)
	sort_results.sort()
	print("the number at normal_ratio of the results is : " + str(sort_results[int(normal_ratio * predict_size)]))


	bound = sort_results[int(normal_ratio * predict_size)]
	predict_labels = [(x > bound) for x in results]

	p, r, f, s = metrics.precision_recall_fscore_support(labels, predict_labels)
	ap = sum([(p[i] * s[i]) for i in range(len(s))]) / sum(s)
	ar = sum([(r[i] * s[i]) for i in range(len(s))]) / sum(s)
	af = sum([(f[i] * s[i]) for i in range(len(s))]) / sum(s)


	return get_index(labels, predict_labels)
	# return ap, ar, af

if __name__ == '__main__':
	LSTMs = pd.read_csv(LABELPATH, skipinitialspace=True)
	RMSEs = pd.read_csv(RMSEPATH, skipinitialspace=True)
	LABELDATA = pd.read_csv(LABELPATH, skipinitialspace=True)
	LABEL = LABELDATA["label"]
	LABEL = np.array(LABEL).tolist()
	labels = [(x != "normal") for x in LABEL]
	rmses = np.array(pd.read_csv(RMSEPATH, header=None,skipinitialspace=True, dtype='float64')).reshape(1,-1)[0].tolist()


	rmse_size = len(rmses)

	labels = labels[rmse_size-predict_size:rmse_size]
	rmses = rmses[-predict_size:]
	lstms = np.array(pd.read_csv(LSTMPATH, header=None, skipinitialspace=True, dtype='float64')).reshape(1,-1)[0].tolist()

	print("length of rmses : " + str(len(rmses)))
	print("length of lstms : " + str(len(lstms)))

	print("> Finding the best value for normal_ratio ...")
	df = DataFrame(columns=('bound', 'precision', 'recall', "fscore"))
	for normal_ratio in normal_ratios:
		P,R,F = report_precision(rmses, lstms, labels, fixp, normal_ratio)
		to_append = pd.Series({'bound':normal_ratio, 'precision':P, 'recall':R, "fscore":F})
		df = df.append(to_append, ignore_index=True)
	df.to_csv("./params/bound_positive.csv")
	print(df)

	print("> Finding the best value for p ...")
	df = DataFrame(columns=('p', 'precision', 'recall', "fscore"))
	for p in ps:
		P,R,F = report_precision(rmses, lstms, labels, p, fixnormal_ratio)
		to_append = pd.Series({'p':p, 'precision':P, 'recall':R, "fscore":F})
		df = df.append(to_append, ignore_index=True)
	df.to_csv("./params/p_positive.csv")
	print(df)



