import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
from keras.applications.mobilenet import MobileNet

#변수 정의
num_classes = 5
epochs = 10
batch_size = 32
train_dir = 'Eywa_data/train'
test_dir = 'Eywa_data/test'

# 모바일넷의 사전학습 모델을 가져옵니다.
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 새로운 레이어를 추가합니다.
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# 파인튜닝할 모델을 정의합니다.
model = Model(inputs=base_model.input, outputs=predictions)

# 모델 파라미터를 가져옵니다.
for layer in base_model.layers:
    layer.trainable = False

# 새로 추가한 레이어의 파라미터를 학습시킵니다.
for layer in model.layers:
    if 'conv' in layer.name:
        layer.trainable = True

# 데이터셋을 모델에 맞게 처리합니다.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# 모델을 학습시킵니다.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=epochs)

# 모델을 평가합니다.
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=1, class_mode='categorical', shuffle=False)
