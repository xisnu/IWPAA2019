from __future__ import print_function
from keras.layers import Dense,Input
from keras.models import Model
from Data import *

input_layer=Input(batch_shape=(None,4))
fc_layer1=Dense(5,activation='tanh')
fc_layer2=Dense(3,activation='softmax')
print("Nodes created")

input_fc1=fc_layer1(input_layer)
fc1_fc2=fc_layer2(input_fc1)

network=Model(input_layer,fc1_fc2)
network.compile(loss='mse',metrics=['accuracy'],optimizer='sgd')
network.summary()
print("Network ready")


train,test,labels=load_and_split_iris(iris_path,0.4)
x_train=np.asarray(train.iloc[:,:4].values,dtype=float)
y_train=make_one_hot(train['class'].values,labels)
print("X_Train: ",x_train.shape," Y_Train: ",y_train.shape)

for i in range(100):
    history=network.fit(x_train,y_train,batch_size=8,epochs=3,verbose=0)
    loss=history.history['loss'][-1]
    acc=history.history['acc'][-1]
    network.save("Weights/last")
    print("Loss %f Accuracy %f"%(loss,acc))

x_test=np.asarray(test.iloc[:,:4].values,dtype=float)
y_test=make_one_hot(test['class'].values,labels)
print("X_Train: ",x_test.shape," Y_Train: ",y_test.shape)

predictions=network.predict(x_test,verbose=0)
score=network.evaluate(x_test,y_test,verbose=0)
print("Predictions: ",predictions.shape," score ",score)