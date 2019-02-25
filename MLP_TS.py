from __future__ import print_function
from keras.layers import Dense,Input
from keras.models import Model
from Data import *

input_layer=Input(batch_shape=(None,512))
fc_layer1=Dense(128,activation='tanh')
fc_layer2=Dense(64,activation='tanh')
fc_layer3=Dense(2,activation='softmax')
print("Nodes created")

input_fc1=fc_layer1(input_layer)
fc1_fc2=fc_layer2(input_fc1)
fc2_fc3=fc_layer3(fc1_fc2)

network=Model(input_layer,fc2_fc3)
network.compile(loss='mse',metrics=['accuracy'],optimizer='sgd')
network.summary()
print("Network ready")


x_train,y_train,train_sl=load_earthquake_data(earthquake_path+"_TRAIN")
print("X_Train: ",x_train.shape," Y_Train: ",y_train.shape)

for i in range(25):
    history=network.fit(x_train,y_train,batch_size=8,epochs=3,verbose=0)
    loss=history.history['loss'][-1]
    acc=history.history['acc'][-1]
    network.save("Weights/last")
    print("Loss %f Accuracy %f"%(loss,acc))

x_test,y_test,test_sl=load_earthquake_data(earthquake_path+"_TEST")
print("X_Test: ",x_test.shape," Y_Test: ",y_test.shape)

predictions=network.predict(x_test,verbose=0)
score=network.evaluate(x_test,y_test,verbose=0)
print("Predictions: ",predictions.shape," score ",score)