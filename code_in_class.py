
import tensorflow as tf

import cProfile
tf.executing_eagerly()

x = [[2.]]
m = tf.matmul(x,x) #矩阵乘法
print("x matmul x = {}".format(m))

a = tf.constant([[1,2],
                 [3,4]])
print(a)

#Broadcasting
b = tf.add(a,1) #会把1展开为矩阵
print(b)


#element-wise multiplication
print(a*b) #点对点的乘法


print(tf.matmul(a,b)) #矩阵乘法

import numpy as np

c = np.multiply(a,b) #tensor&numpy转换、计算
print(c)


#Transfer a tensor to numpy array
print(a.numpy()) # 把tensor转为ndarray

'''
Computing gradients
Automatic differentiation is useful for implementing machine learning algorithms such as backpropagation for training neural networks. During eager execution, use tf.GradientTape to trace operations for computing gradients later
'''
# 深度学习库用于计算梯度
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w #计算梯度的函数
grad = tape.gradient(loss,w)
print(grad)


'''
train a model
'''

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:10000,:,:] #前10000张图
y_train = y_train[:10000]
x_test = x_test[:1000,:,:]
y_test = y_test[:1000]


x_train = tf.cast(x_train[...,tf.newaxis]/255, tf.float32), ## Add a channels dimension进行维度扩展
# 将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
# 那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以
x_test = tf.cast(x_test[...,tf.newaxis]/255, tf.float32),

#y_train = y_train.astype('float32')
#y_test = y_test.astype('float32')
y_train = tf.keras.utils.to_categorical(y_train,10) #是分类任务，把y_train变成one-hot
y_test = tf.keras.utils.to_categorical(y_test,10)

#Build the model using Sequential

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,[3,3],activation='relu', #filter_num,filter_size
                          input_shape=(28,28,1)),#输入形状
    tf.keras.layers.Conv2D(64,[3,3],activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25), #25%的dropout
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation="softmax")#FNN
])


mnist_model.summary()
'''
Model: "sequential" 不是非常灵活，为最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠
ResNet can skip就无法用sequential进行
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               1179776   
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
_________________________________________________________________

'''

#Build the model using Model # 可改变网络结构
inputs = tf.keras.Input(shape=(None,None,1),name="digits")
conv_1 = tf.keras.layers.Conv2D(16,[3,3],activation="relu")(inputs) #接收第一层inputs
conv_2 = tf.keras.layers.Conv2D(16,[3,3],activation="relu")(conv_1) #接收第2层conv_1
ave_pool = tf.keras.layers.GlobalAveragePooling2D()(conv_2) #接收第3层conv_2
outputs = tf.keras.layers.Dense(10)(ave_pool)
mnist_model_2 = tf.keras.Model(inputs=inputs,outputs=outputs) #提供最早的输入和最后的输出，中间的链接会自动去找


mnist_model_2.summary()
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
digits (InputLayer)          [(None, None, None, 1)]   0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, None, None, 16)    160       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, None, None, 16)    2320      
_________________________________________________________________
global_average_pooling2d (Gl (None, 16)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                170       
=================================================================
Total params: 2,650
Trainable params: 2,650
Non-trainable params: 0
_________________________________________________________________

'''

'''
Two training methods
Use keras fit method
'''
mnist_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3), #选择优化器
                    validation_split=0.1,shuffle=True, # 分为。。打乱。。
                   loss = tf.keras.losses.categorical_crossentropy,
                   metrics = ["accuracy"])

mnist_model_2.compile(loss = tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])

mnist_model.fit(x_train,y_train,batch_size=128,epochs=3)


mnist_model.evaluate(x_test,y_test) # 返回[loss, accuracy]

x_train[0]

#mnist_model.predict(x_test[0][1])

'''
Use TF 2.0
'''

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:10000,:,:]
y_train = y_train[:10000]
x_test = x_test[:1000,:,:]
y_test = y_test[:1000]

dataset = tf.data.Dataset.from_tensor_slices(
(tf.cast(x_train[...,tf.newaxis]/255, tf.float32),
 tf.cast(y_train,tf.int64)))
dataset = dataset.shuffle(1000).batch(32) #不会把所有数据放在缓存中

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_history = []

for epoch in range(5): #训练5个epoch
    for (batch, (images, labels)) in enumerate(dataset): # 每次提取一个batch的data
        with tf.GradientTape() as tape: #计算梯度
            logits = mnist_model(images, training=True)
            loss_value = loss(labels, logits) #(真实标签，预测值)

        grads = tape.gradient(loss_value, mnist_model.trainable_variables) #求所有可训练参数的梯度
        optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))# ww-a*dL/dW

    print("Epoch {} finishted".format(epoch))

