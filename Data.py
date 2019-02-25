from __future__ import print_function
import numpy as np
import pandas as pd
import os

iris_path="Data/IRIS/iris.data"
earthquake_path="Data/Earthquake/Earthquakes"
osuleaf_path="Data/OSULeaf/OSULeaf"
face_path="/media/parthosarothi/OHWR/Dataset/UCI_CMU_Faces"

def load_iris_data(iris_csv):
    f=open(iris_csv)
    line=f.readline()
    iris_df=pd.DataFrame()
    while line:
        info=line.strip("\n").split(",")
        if(len(info)==5):
            label=info[-1]
            features=info[:4]
            features.append(label)
            #print(features)
            df = pd.DataFrame([features], columns=['SL', 'SW', 'PL', 'PW', 'class'])
            iris_df=iris_df.append(df)
        line=f.readline()
    return iris_df

def load_and_split_iris(iris_path,split_ratio):
    iris_df = load_iris_data(iris_path)
    class_labels = list(set(iris_df['class']))
    print(class_labels)
    train_set=pd.DataFrame()
    test_set=pd.DataFrame()
    for cl in class_labels:
        class_loc=iris_df.loc[iris_df['class']==cl]
        nb_samples=len(class_loc)
        class_loc=class_loc.sample(nb_samples)
        #print(class_loc)
        test_volume=int(nb_samples*split_ratio)
        train_volume=nb_samples-test_volume
        #print(class_loc.iloc[train_volume:,:])
        train_set=train_set.append(class_loc.iloc[:train_volume,:])
        test_set=test_set.append(class_loc.iloc[train_volume:,:])

    nbtests=len(test_set)
    test_set=test_set.sample(nbtests)

    nbtrain = len(train_set)
    train_set = train_set.sample(nbtrain)
    print("Train samples=%d Test samples=%d"%(nbtrain,nbtests))
    return train_set,test_set,class_labels

def make_one_hot(batch_labels,all_labels):
    total_classes=len(all_labels)
    total_samples=len(batch_labels)
    one_hot=np.zeros([total_samples,total_classes],dtype=float)
    for ts in range(total_samples):
        one_hot[ts][all_labels.index(batch_labels[ts])]=1
    return one_hot

def load_earthquake_data(earthquake_csv):
    f=open(earthquake_csv)
    line=f.readline()
    sequence_length=[]
    labels=[]
    features=[]
    while line:
        info=line.strip("\n").split(",")
        label=info[0]
        label_one_hot=[0,0]
        label_one_hot[int(label)]=1
        feat=info[1:]
        nbfeatures=len(feat)
        sequence_length.append(nbfeatures)
        labels.append(label_one_hot)
        features.append(feat)
        line=f.readline()
    max_length=max(sequence_length)
    min_length=min(sequence_length)
    print("Maximum sequence length: ",max_length," minimum sequence length: ",min_length)
    return np.asarray(features),np.asarray(labels),sequence_length

def load_timeseries_data(timeseries_csv,nbclass,class_label_from_0=False):
    f=open(timeseries_csv)
    line=f.readline()
    sequence_length=[]
    labels=[]
    features=[]
    while line:
        info=line.strip("\n").split(",")
        label=info[0]
        label_one_hot=np.zeros([nbclass])
        if(class_label_from_0):
            label_index=int(label)
        else:
            label_index=int(label)-1
        label_one_hot[label_index]=1
        feat=info[1:]
        nbfeatures=len(feat)
        sequence_length.append(nbfeatures)
        labels.append(label_one_hot)
        features.append(feat)
        line=f.readline()
    max_length=max(sequence_length)
    min_length=min(sequence_length)
    print("Maximum sequence length: ",max_length," minimum sequence length: ",min_length)
    return np.asarray(features),np.asarray(labels),sequence_length

def split_cmu_face_images(face_path,resolution,test_split):
    all_files=[]
    for root,sd,files in os.walk(face_path):
        for fn in files:
            # print(fn)
            if(fn[-3:]=='pgm'):
                filename = fn[:-4]
                attributes=filename.split("_")
                if(attributes[-1]==resolution):
                    all_files.append(os.path.join(root,fn))
    total=len(all_files)
    nbtests=int(total*test_split)
    nbtrain=total-nbtests
    train_data=all_files[:nbtrain]
    test_data=all_files[nbtrain:]
    print("Directory scan complete: %d samples Train %d Test %d" % (total,len(train_data),len(test_data)))
    return train_data,test_data

def load_cmu_face_images(face_path,resolution,test_split):
    train,test=split_cmu_face_images(face_path,resolution,test_split)



load_cmu_face_images(face_path,"4",0.4)