from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


############### 패션 MNIST 데이터셋 임포트 ###############
# 70,000개의 흑백 이미지로 구성된 패션 데이터셋
# 이미지 해상도는 28X28 픽셀, 픽셀값은 0~255 사이 값

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape # 훈련세트 : 60,000개
len(train_labels) # 레이블: 0~9사이의 정수 (10개)
train_labels
test_images.shape # 테스트세트 : 10,000개
len(test_labels) # 레이블: 0~9사이의 정수 (10개)


############### 데이터 전처리 ###############
# 픽셀 값의 범위가 0~255이므로, 신경망 모델에 주입하기전에 0~1 사이로 조정
# 훈련 세트와 테스트 세트를 동일한 방식으로 전처리 하는 것이 중요

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


############### 모델 구성 ###############
# 층 설정
# Flatten : 2차원 배열 (28x28) 이미지 포맷을 1차원 배열로 변환, 가중치가 없고 데이터를 변환하기만 함
# 첫번째 Dense: 128개의 노드를 가짐
# 두번째 Dense: 현재 이미지가 10개의 레이블 중 하나에 속할 확률을 반환함

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
# 옵티마이저: 데이터와 손실함수를 바탕으로 모델의 업데이트 방법을 결정
# 손실 함수: 훈련하는 동안 모델의 오차를 측정
# 지표: 훈련 단계와 테스트 단계를 모니터링 하기위해 사용
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

############### 모델 훈련 ###############
model.fit(train_images, train_labels, epochs=100)


############### 정확도 평가 ###############
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)

############### 예측만들기 ###############
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
test_labels[0]


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

# 처음 X 개의 테스트 이미지와 예측 레이블, 진짜 레이블을 출력
# 올바른 예측은 파랑색으로 잘못된 예측은 빨강색으로 나타냄
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# 테스트 세트에서 이미지 하나를 선택
img = test_images[0]

print(img.shape)

# 이미지 하나만 사용할 때도 배치에 추가
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])