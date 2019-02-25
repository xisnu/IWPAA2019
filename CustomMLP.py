from __future__ import print_function
import keras.backend as K
from keras.layers import Layer,Input,Dense
from keras.models import Model
from Data import *


class ResidualFC(Layer):
    def __init__(self,nodes,**kwargs):
        super(ResidualFC,self).__init__(**kwargs)
        self.nodes=nodes
        print("layer initiated")

    def build(self, input_shape):
        self.input_dim=input_shape[-1]
        self.Wk=self.add_weight('Wk',shape=[self.input_dim,self.nodes],initializer='uniform')
        self.bias=self.add_weight('bias',shape=[self.nodes],initializer='uniform')
        super(ResidualFC,self).build(input_shape)
        print("layer built")

    def call(self, inputs, **kwargs):
        self.kernel_output=K.dot(inputs,self.Wk)+self.bias
        self.layer_output=K.concatenate([self.kernel_output,inputs])
        self.layer_output=K.tanh(self.layer_output)
        print(K.int_shape(self.layer_output))
        return self.layer_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[-1]+self.nodes)


input_layer=Input(batch_shape=[None,4])
layer_1=ResidualFC(5)(input_layer)
layer_2=Dense(3)(layer_1)
network=Model(input_layer,layer_2)
network.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
network.summary()

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
