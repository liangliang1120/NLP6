'''
3.1 In the first session of the practical part,
you will implement an image classification model using any deep learning libraries that you are familiar with,
which means, except for tensorflow and keras, you can also use pytorch/caffe/... .
The dataset used in this session is the cifar10 which contains 50000 color (RGB) images,
each with size 32x32x3.
All 50000 images are classified into ten categories.
'''


import tensorflow as tf
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()

for i in range(1,11):
    plt.subplot(2,5,i)
    plt.imshow(x_train[i-1])
    plt.text(3,10,str(y_train[i-1]))
    plt.xticks([])
    plt.yticks([])
plt.show()

#It is your time to build your model. Try your best to build a model with good performance on the test set.
batch_size = 32
num_classes = 10
epochs = 1600
data_augmentation = True

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:])) #same’是周边补充空白表示特征图大小不变
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(48, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(48, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2))) #池化层
model.add(tf.keras.layers.Dropout(0.25)) #将输入单元的按比率随机设置为 0， 这有助于防止过拟合。

model.add(tf.keras.layers.Conv2D(80, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(80, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(80, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(80, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(80, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.GlobalMaxPooling2D()) #对于空域数据的全局最大池化
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(500))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(num_classes)) #Dense（10）表示把他压成和我们labels一样的维度10
model.add(tf.keras.layers.Activation('softmax')) #通过softmax进行激活（多分类用softmax）
model.summary() #打印网络结构及其内部参数

# initiate RMSprop optimizer
opt = tf.keras.optimizers.Adam(lr=0.0001)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

print("train____________")
model.fit(x_train,y_train,epochs=600,batch_size=128,)
print("test_____________")
loss,acc=model.evaluate(x_test,y_test)
print("loss=",loss)
print("accuracy=",acc)

































