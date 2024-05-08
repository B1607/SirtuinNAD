#!/usr/bin/env python
# coding: utf-8

# # Setting Up the Environment
import numpy as np
import math
from sklearn import metrics
from sklearn.metrics import roc_curve
import tensorflow as tf
from tensorflow.keras import layers,Model
from sklearn.model_selection import KFold
import gc
import MCNN
import import_NADBP as load_data


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ds","--dataset", type=str, default="Sirtuin7", help="Choosing of dataset, 'Sirtuin7' or 'NAD_dependent'")
parser.add_argument("-n_dep","--num_dependent", type=int, default=7, help="the number of dependent variables")
parser.add_argument("-n_fil","--num_filter", type=int, default=1024, help="the number of filters in the convolutional layer")
parser.add_argument("-ws","--window_sizes", nargs="+", type=int, default=[2,4,6,8,10], help="the window sizes for convolutional filters")
parser.add_argument("-vm","--validation_mode", type=str, default="cross")

args=parser.parse_args()

# # Parameter Setup for Machine Learning
DATASET=args.dataset
NUM_DEPENDENT =args.num_dependent
NUM_FILTER = args.num_filter
WINDOW_SIZES = args.window_sizes
VALIDATION_MODE=args.validation_mode

MAXSEQ = NUM_DEPENDENT*2+1
NUM_FEATURE = 1024
NUM_HIDDEN = 1000#100
BATCH_SIZE  = 1024
EPOCHS      = 20
K_Fold = 5
NUM_CLASSES = 2
CLASS_NAMES = ['Negative','Positive']


# # Data Loading
x_train,y_train,x_test,y_test = load_data.MCNN_data_load(NUM_CLASSES,NUM_DEPENDENT,DATASET)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# # Function Definition
def model_test(model, x_test, y_test):

    print(x_test.shape)
    pred_test = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:,1], pred_test[:, 1])
    AUC = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='mCNN')
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]

    y_pred = (pred_test[:, 1] >= threshold).astype(int)

    TN, FP, FN, TP =  metrics.confusion_matrix(y_test[0:][:,1], y_pred).ravel()

    Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
    Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
    Acc = (TP+TN)/(TP+FP+TN+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
    F1 = 2*TP/(2*TP+FP+FN)
    Prec=TP/(TP+FP)
    Recall=TP/(TP+FN)
    return TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC,display



# # Cross Validation
if(VALIDATION_MODE=="cross"):
	
	kfold = KFold(n_splits = K_Fold, shuffle = True, random_state = 2)
	results=[]
	i=1
	for train_index, test_index in kfold.split(x_train):
		print(i,"/",K_Fold,'\n')
		# 取得訓練和測試數據
		X_train, X_test = x_train[train_index], x_train[test_index]
		Y_train, Y_test = y_train[train_index], y_train[test_index]
		print(X_train.shape)
		print(X_test.shape)
		print(Y_train.shape)
		print(Y_test.shape)
		# 重新建模
		model = MCNN.DeepScan(
            input_shape=(1, MAXSEQ, NUM_FEATURE),
            num_class=NUM_CLASSES,
            maxseq=MAXSEQ,
    		num_filters=NUM_FILTER,
			num_hidden=NUM_HIDDEN,
			window_sizes=WINDOW_SIZES)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.build(input_shape=X_train.shape)
		# 在測試數據上評估模型
		history=model.fit(
			X_train,
			Y_train,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
			verbose=1,
			shuffle=True
		)
		TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC, display = model_test(model, X_test, Y_test)
		results.append([TP, FP, TN, FN, Sens, Spec, Acc, MCC, AUC])
		i+=1
		
		del X_train
		del X_test
		del Y_train
		del Y_test
		gc.collect()
		
	mean_results = np.mean(results, axis=0)
	print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}\n')


# # Independent Test
if(VALIDATION_MODE=="independent"):
	model = MCNN.DeepScan(
            input_shape=(1, MAXSEQ, NUM_FEATURE),
            num_class=NUM_CLASSES,
            maxseq=MAXSEQ,
    		num_filters=NUM_FILTER,
			num_hidden=NUM_HIDDEN,
			window_sizes=WINDOW_SIZES)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.build(input_shape=x_train.shape)
	model.summary()

	model.fit(
		x_train,
		y_train,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
		shuffle=True,
	)
	
	TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC,display = model_test(model, x_test, y_test)
	print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}\n')
	display.plot()





