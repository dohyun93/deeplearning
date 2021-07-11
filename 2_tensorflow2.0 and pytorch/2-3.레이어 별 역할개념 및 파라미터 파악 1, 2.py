# Layer 설명
import tensorflow as tf
import os
import numpy as np
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Input image
# input으로 들어갈 데이터셋을 들여다보면서 시각화까지

# 패키지 로드
# os
# glob
# matplotlib

import matplotlib.pyplot as plt
from tensorflow.keras import datasets
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
image = train_x[0]
print(image.shape)
# 28 x 28
# plt.imshow(image, 'gray')
# plt.show()

# 학습을 위해서는 배치 크기 및 채널 차원을 앞뒤로 추가해주어야 한다.
image = image[tf.newaxis, ..., tf.newaxis]
print(image.shape)
# 이미지를 통한 classification -> 분류 는
# 1. feature extraction (이미지로부터) 를 하고, -> 특성 추출.
# 2. 추출된 feature를 활용해 classification을 한다. -> 여기서 추출된 특성을 바탕으로 분류를 하는 것

# 어떻게 특징을 추출할까? -> convolution (합성곱)
# 이미지에 대해 filter들을 input image에 합성곱 한 feature maps들을 추출

# [용어 정리]
# filter: layer에서 나갈 때 몇 개의 filter를 만들 것인지? (weights, filters, channels)
# kernel_size: filter(weight)의 사이즈
# strides : 몇 개의 pixel을 skip하면서 훑어지나갈 것인지 (사이즈에 영향을 줌)
# padding : zero padding을 만들 것인지, VALID는 padding이 없고 same은 padding이 있음
# activation : activation function을 만들 것인지. 설정 안해도 layer층을 별도로 만들 수 있다.

#tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation='relu')
# tf.keras.layers.Conv2D(3, 3, 1, 'SAME') 이렇게 해도 kernel/strides는 자동으로 두 값 동일하게 들어감. 다른값 넣을때만 위처럼.

# 시각화
image = tf.cast(image, dtype=tf.float32) # 이미지가 int로 되어있으면 모델에 넣을때 에러가나서 tf.cast로 실수형으로.
print("1, ", image)

layer = tf.keras.layers.Conv2D(5, 3, 1, padding='SAME')
# padding이 VALID면 채널 줄어듬
# layer = tf.keras.layers.Conv2D(5, 3, 1, padding='VALID') -> (1, 26, 26, 5)

output = layer(image)
# np.min(image): 0.0 np.max(image): 255.0
# np.min(output): -383.28613, np.max(output): 247.72801
print(np.min(image), np.max(image))
print(np.min(output), np.max(output))

# filter갯수만큼 채널수가 바뀜.
print("2, ", output)
plt.subplot(1, 2, 1)
plt.imshow(image[0, :, :, 0], 'gray')
plt.subplot(1, 2, 2)
plt.imshow(output[0, :, :, 0], 'gray')
plt.show()

#========================================================================

# weight 불러오기.
# layer.get_wegiths()
weight = layer.get_weights() # weight, bias 를 반환함에 주의
print("weight shape: ", weight[0].shape, "\nbias shape: ", weight[1].shape)
# weight shape:  (3, 3, 1, 5)
# bias shape:  (5,)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.hist(output.numpy().ravel(), range=[-2, 2])
plt.ylim(0, 100)
plt.subplot(132)
plt.title("used filter" + str(weight[0].shape))
plt.imshow(weight[0][:, :, 0, 0], 'gray')
plt.subplot(133)
plt.title(output.shape)
plt.imshow(output[0, :, :, 0], 'gray')
plt.colorbar()
plt.show()

# Activation function
# tf.keras.layers.ReLU()
act_layer = tf.keras.layers.ReLU()
act_output = act_layer(output)
print(act_output)
print(np.min(act_output), np.max(act_output))
# np.min(act_output), np.max(act_output) -> 0.0, 223.75673

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.hist(act_output.numpy().ravel(), range=[-2, 2])
plt.ylim(0, 100)

plt.subplot(122)
plt.title(act_output.shape)
plt.imshow(act_output[0, :, :, 0], 'gray')
plt.show()

## Pooling
# 이미지 압축.
# tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
pool_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
pool_output = pool_layer(act_output)
print(act_output.shape)
print(pool_output.shape)
print(pool_output)

plt.figure(figsize=(15, 5))
plt.subplot(121) # same as plt.subplot(1, 2, 1)
plt.hist(pool_output.numpy().ravel(), range=[-2, 2])
plt.ylim(0, 100)

plt.subplot(122)
plt.title(pool_output.shape)
plt.imshow(pool_output[0, :, :, 0], 'gray')
plt.colorbar()
plt.show()