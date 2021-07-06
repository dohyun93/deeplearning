import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

# 1. 데이터 불러오기
# Tensorflow에서 제공해주는 MNIST 예제 불러오기
from tensorflow.keras import datasets

# 데이터 shape 확인
mnist = datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# Image dataset 들여다보기
# 불러온 데이터셋에서 이미지 데이터 하나만 뽑아서 시각화까지 확인
# 데이터 하나만 뽑기
image = train_x[0]
print("불러온 한 장의 이미지 shape: ", image.shape)
plt.imshow(image, 'gray')
plt.show()

# Channel 관련
# [Batch size, Height, Width, Channel]
# Gray scale이면 1, RGB면 3으로 만들어줘야 함

# 다시 shape로 데이터 확인
print("학습 데이터 shape: ", train_x.shape)
# 1. Numpy로 데이터 차원 수 늘리기 (gray-scale이므로 channel 차원 추가필요)
#train_x = np.expand_dims(train_x, -1)
#print("Channel차원 추가한 학습 데이터 shape: ", train_x.shape)

#2. tf로 데이터 차원 수 늘리기
new_train_x = train_x[..., tf.newaxis] # -> 이게 아래 방법보다 낫다..
print(new_train_x.shape)
# 또는 아래 numpy처럼
# train_x = tf.expand_dims(train_x, -1)

# 주의사항 #
# matplotlib으로 이미지 시각화 할 때는 gray scale의 이미지는 세 번째 dimension이 없으므로,
# 2개의 dimension으로 gray scale로 차원 조절해서 넣어줘야 한다.

print(new_train_x[0].shape) # 28 x 28 x 1
squeezed = np.squeeze(new_train_x[0])
print(squeezed.shape)

plt.imshow(squeezed, 'gray')
plt.show()




# Label Dataset 들여다보기
print(train_y.shape) # (60000,)

plt.imshow(train_x[0])
plt.show()
print(train_y[0])

# label 시각화
plt.title(train_y[0])
plt.imshow(train_x[0], 'gray')
plt.show()



# OneHot Encoding
# 컴퓨터가 이해할 수 있는 형태인 바이너리로 변환해서 label을 주도록 함
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] -> 1에 대한 onehot encoding

# tensorflow.keras.utils.to_categorical
from tensorflow.keras.utils import to_categorical


# 1을 예시로 one hot encoding
print(to_categorical(1, 10, dtype=int))


# label 확인해서 to_categorical 사용
label = train_y[0]

# onehot encoding으로 바꾼 것과 이미지 확인
label_onehot = to_categorical(label, num_classes=10, dtype=int)
print(label_onehot)

plt.title(label_onehot)
plt.imshow(train_x[0], 'gray')
plt.show()