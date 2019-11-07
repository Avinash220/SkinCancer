from keras.models import Sequential
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import TensorBoard
import time
import os
import tensorflow as tf

# 指定GPU，限制GPU内存
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

BATCHSIZE = 100
IMG_SIZE = (100, 100)

# 训练集，测试集文件路径
train_path = '../data/train'
test_path = '../data/test'

s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())  # 时间戳

# image_batch_generator

train_datagen = ImageDataGenerator(
    rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# 训练集batch生成器
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=IMG_SIZE,
    batch_size=BATCHSIZE,
    color_mode='grayscale',
    classes=['original', 'tampered'],
    class_mode='categorical')

# 测试集batch生成器
validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMG_SIZE,
    color_mode='grayscale',
    batch_size=BATCHSIZE,
    classes=['original', 'tampered'],
    class_mode='categorical')

# 网络结构
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# 优化器
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# logs文件路径
logs_path = 'F:/zy/logs/log_%s' %(s_time)

try:
    os.makedirs(logs_path)
except:
    pass

# 将loss ，acc， val_loss ,val_acc记录tensorboard
tensorboard = TensorBoard(log_dir=logs_path, histogram_freq=1, write_graph=True, write_batch_performance=True)

# 模型训练
model.fit_generator(
    train_generator,
    steps_per_epoch=60,
    epochs=50,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=60,
    callbacks=[tensorboard]
)


# https://download.csdn.net/download/weixin_39840914/11520098
