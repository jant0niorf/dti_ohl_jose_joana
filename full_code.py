import os
import json
import tensorflow as tf
import numpy as np
import itertools
import tensorflow as tf
import keras
from keras import *
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.optimizers import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from tensorflow.python.ops import math_ops


# FUNCTIONS

def sigmoid_to_binary(predicted_labels):
    binary_labels=[]
    for i in predicted_labels:
        if i>0.5:
            binary_labels.append(1)
        else:
            binary_labels.append(0)
    return binary_labels

def data_conversion(dataset,dictionary,max_length):
    matrix=np.zeros([len(dataset),max_length])
    for i in range(len(dataset)):
        dataset[i][1]=list(dataset[i][1])
        for j in range(len(dataset[i][1])):
            for k in dictionary:
                if dataset[i][1][j]==k:
                    dataset[i][1][j]=dictionary.get(k)
        if len(dataset[i][1])<max_length:
            matrix[i,0:len(dataset[i][1])]=dataset[i][1]
        else:
            matrix[i,0:max_length]=dataset[i][1][0:max_length]
    return matrix.astype('int32')

def generate_input(shape_size,dtype):
    data_input=Input(shape=(shape_size,),dtype=dtype)
    return data_input

def OneHot(input_dim=None, input_length=None):
    def one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),num_classes=num_classes)
    return Lambda(one_hot, arguments={'num_classes': input_dim}, input_shape=(input_length,))

def sensitivity(y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TP,TP+FN)
    return metric

def specificity(y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TN,TN+FP)
    return metric

def f1_score(y_true,y_pred):
    y_pred=math_ops.round(y_pred)
    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)
    metric=tf.divide(TN,TN+FP)
    precision = tf.divide(TP,TP + FP)
    sensitivity = tf.divide(TP,TP+FN)
    metric = tf.divide(tf.multiply(2*precision,sensitivity),precision + sensitivity)
    return metric


def metrics_function(sensitivity,specificity,f1,accuracy,auc_value,auprc_value,binary_labels,predicted_labels,labels_test,confusion_matrix):
    sensitivity_value=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0])
    specificity_value= confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
    precision=confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    f1_value=2*(precision*sensitivity_value)/(precision+sensitivity_value)
    accuracy=accuracy_score(labels_test,np.array(binary_labels))
    auc=roc_auc_score(labels_test,predicted_labels)
    auprc=average_precision_score(labels_test,predicted_labels)
    metrics=[]
    if sensitivity:
        metrics.append('Sensitivity:'+str(sensitivity_value))
    if specificity:
        metrics.append('Specificity:'+str(specificity_value))
    if f1:
        metrics.append('F1_Score:'+str(f1_value))
    if accuracy:
        metrics.append('Accuracy:'+str(accuracy))
    if auc_value:
        metrics.append('AUC:'+str(auc))
    if auprc_value:
        metrics.append('AUPRC: '+str(auprc))
    return metrics

def data_treat():

    # PARAMETERS

    prot_max_length=1205
    drug_max_length=90

    # PROTEIN DATA

    prot_train=[i.rstrip().split(',') for i in open('/content/Protein_Train_Dataset.csv')]
    prot_test=[i.rstrip().split(',') for i in open('/content/Protein_Test_Dataset.csv')]

    # DRUG DATA

    drug_train=[i.rstrip().split(',') for i in open('/content/Smile_Train_Dataset.csv')]
    drug_test=[i.rstrip().split(',') for i in open('/content/Smile_Test_Dataset.csv')]

    # DICTIONARIES

    prot_dictionary=json.load(open('/content/aa_properties_dictionary.txt'))
    prot_dict_size=len(prot_dictionary)
    drug_dictionary=json.load(open('/content/smile_dictionary.txt'))
    drug_dict_size=len(drug_dictionary)

    # CONVERSION: SEQUENCE TO INTEGERS

    prot_train_data=data_conversion(prot_train,prot_dictionary,prot_max_length)
    prot_test_data=data_conversion(prot_test,prot_dictionary,prot_max_length)

    drug_train_data=data_conversion(drug_train,drug_dictionary,drug_max_length)
    drug_test_data=data_conversion(drug_test,drug_dictionary,drug_max_length)

    # LABELS

    labels_train=np.load('/content/labels_train.npy')
    labels_test=np.load('/content/labels_test.npy')

    # INPUTS TENSOR BASED

    prot_input=generate_input(prot_max_length,'int32')
    drug_input=generate_input(drug_max_length,'int32')

    prot_encoded=OneHot(input_dim=prot_dict_size,input_length=prot_max_length)(prot_input)
    drug_encoded=OneHot(input_dim=drug_dict_size,input_length=drug_max_length)(drug_input)

    return prot_train, drug_train, prot_test, drug_test, prot_dict_size, drug_dict_size, prot_train_data, drug_train_data, prot_test_data, drug_test_data, labels_train, labels_test, prot_input, drug_input, prot_encoded, drug_encoded


# OBJECTS

prot_train, smile_train, prot_test, smile_test, prot_dict_size, smile_dict_size, prot_train_data, smile_train_data, prot_test_data, smile_test_data, labels_train, labels_test, prot_input, smile_input, prot_encoded, smile_encoded=data_treat()

# CLASSIFIER DEFINITION

# SMILES CNN

smile_cnn=Conv1D(128, kernel_size=3, activation='relu')(smile_encoded)
smile_cnn=Conv1D(256, kernel_size=4, activation='relu')(smile_cnn)
smile_cnn=Conv1D(384, kernel_size=5, activation='relu')(smile_cnn)
smile_pool=GlobalMaxPooling1D()(smile_cnn)

# protS CNN

prot_cnn=Conv1D(128, kernel_size=3, activation='relu')(prot_encoded)
prot_cnn=Conv1D(256, kernel_size=4, activation='relu')(prot_cnn)
prot_cnn=Conv1D(384, kernel_size=5, activation='relu')(prot_cnn)
prot_pool=GlobalMaxPooling1D()(prot_cnn)

# Concatenation of the models

features=Concatenate()([prot_pool,smile_pool])

# FCNN - Dense layers and output

fcnn_layer=Dense(128,activation='relu')(features)
fcnn_layer=Dropout(0.1)(fcnn_layer)
fcnn_layer=Dense(128,activation='relu')(fcnn_layer)
fcnn_layer=Dropout(0.1)(fcnn_layer)
fcnn_layer=Dense(128,activation='relu')(fcnn_layer)
output=Dense(1,activation='sigmoid')(fcnn_layer)

# Model programation
model=Model(inputs=[prot_input,smile_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class_weight={0:0.36,1:0.64}

model.fit([prot_train_data,smile_train_data],labels_train, class_weight=class_weight, epochs=10, batch_size=32)
model.evaluate([prot_test_data,smile_test_data],labels_test)
predicted_labels=model.predict([prot_test_data,smile_test_data])
binary_labels=sigmoid_to_binary(predicted_labels)
cm=confusion_matrix(labels_test,np.array(binary_labels))
#print(cm)
metric_values=metrics_function(True,True,True,True,True,True,binary_labels,predicted_labels,labels_test,cm)
print(metric_values)
model.summary()

