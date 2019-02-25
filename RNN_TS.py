from __future__ import print_function
from keras.layers import Dense,Input,RNN,LSTM
from keras.models import Model
from Data import *

time=427
feat=1
datapath=osuleaf_path
nbclass=6

input_layer=Input(batch_shape=(None,time,feat))
rnn_layer1=LSTM(64,return_sequences=False)
fc_layer1=Dense(nbclass,activation='softmax')
print("Nodes created")

input_rnn1=rnn_layer1(input_layer)
rnn1_fc1=fc_layer1(input_rnn1)

network=Model(input_layer,rnn1_fc1)
network.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='sgd')
network.summary()
print("Network ready")


x_train,y_train,train_sl=load_timeseries_data(datapath+"_TRAIN",nbclass)
x_train=np.reshape(x_train,[len(x_train),time,feat])
print("X_Train: ",x_train.shape," Y_Train: ",y_train.shape)

for i in range(50):
    history=network.fit(x_train,y_train,batch_size=8,epochs=1,verbose=0)
    loss=history.history['loss'][-1]
    acc=history.history['acc'][-1]
    network.save("Weights/last")
    print("Loss %f Accuracy %f"%(loss,acc))

x_test,y_test,test_sl=load_timeseries_data(datapath+"_TEST",nbclass)
x_test=np.reshape(x_test,[len(x_test),time,feat])
print("X_Test: ",x_test.shape," Y_Test: ",y_test.shape)

predictions=network.predict(x_test,verbose=0)
score=network.evaluate(x_test,y_test,verbose=0)
print("Predictions: ",predictions.shape," score ",score)