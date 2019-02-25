from __future__ import print_function
from keras.layers import Input,Conv2D,MaxPool2D,Flatten,Dense
from keras.models import Model
from Data import *

h=30
w=32
c=1
Nc=2

input_layer=Input(batch_shape=(None,h,w,c))
conv1=Conv2D(4,[3,3])(input_layer)
mp1=MaxPool2D(strides=[2,2])(conv1)
conv2=Conv2D(8,[3,3])(mp1)
mp2=MaxPool2D(strides=[2,2])(conv2)
flat=Flatten()(mp2)
fc=Dense(Nc,activation='softmax')(flat)
network=Model(input_layer,fc)
network.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
network.summary()


x_train,y_train,x_test,y_test=load_cmu_face_images(face_path,"4",0.4,["open","sunglasses"])
x_train=np.expand_dims(x_train,-1)
x_test=np.expand_dims(x_test,-1)

print(y_train[0])

for i in range(20):
    history=network.fit(x_train,y_train,batch_size=8,epochs=1,verbose=0)
    loss=history.history['loss'][-1]
    acc=history.history['acc'][-1]
    network.save("Weights/last")
    print("Loss %f Accuracy %f"%(loss,acc))

score=network.evaluate(x_test,y_test)
print(score)