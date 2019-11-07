from keras_applications.inception_v3 import InceptionV3
from keras_preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# pre-trained network
base_model = InceptionV3(wights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# a fully-connected layer as first layer
x = Dense(1024, activation='relu')(x)
# a logistic layer
predictions = Dense(200, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

# deactivate all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

# train the net using the new data
model.fit_generator(...)

# train the top two inception parts, which means deactivate the front 172 layers and activate the rest layers
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(ir=0.0001, momentum=0.9),loss='categorical_crossentropy')

model.fit_generator(...)