import numpy as np
import tensorflow as tf

# Tensor 생성
print([1, 2, 3])

# Array 생성
# tuple이나 list 둘다 np.array()로 씌어서 array를 만들 수 있다.

arr = np.array([1, 2, 3])
print(arr.shape)

tuple = np.array((1, 2, 3))
print(tuple.shape)

# Tensor 생성!
# np.array처럼 tf.constant()라는 함수로 Tensor를 생성해줄 수 있다.

# 1. list -> Tensor로 변환
print(tf.constant([1, 2, 3])) # tf.Tensor([1 2 3], shape=(3,), dtype=int32)

# 2. tuple -> Tensor로 변환
print(tf.constant(((1, 2, 3), (1, 2, 3))))
# tf.Tensor(
# [[1 2 3]
#  [1 2 3]], shape=(2, 3), dtype=int32)

# 3. Array -> Tensor로 변환 (중요)
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(tf.constant(arr))
tensor = tf.constant(arr)
# tf.Tensor(
# [[1 2 3]
#  [4 5 6]], shape=(2, 3), dtype=int32)



## Tensor에 담긴 정보 확인 ##

# 1. shape 확인
print(tensor.shape)

# data type 확인
# 주의: Tensor 생성할 때도 data type을 정해주지 않기 때문에 data type에 대한 혼동이 올 수 있다.
#      Data Type에 따라 모델의 무게나 성능 차이에도 영향을 줄 수 있음에 유의.
print(tensor.dtype)

# 그러면, 데이터 타입을 예측 가능하게 정의를 해주는 방법은?
tensor2 = tf.constant([1, 2, 3], dtype=tf.float32)
print(tensor2.dtype)

# data type 변환도 가능하다.
# numpy에서 astype()을 주듯, Tensorflow에서는 tf.cast()를 사용
# 아래는 numpy dtype 변환
arr = np.array([1, 2, 3], dtype=np.float32)
print("arr는 최초 np.float32형 이었음")
arr = arr.astype(np.uint8)
print("np.astype()으로 바꾼 자료형: ", arr.dtype)

# 아래는 Tensorflow에서 tf.cast로 dtype바꾸는 코드
print("tensor2는 tf.float32형 이었음")
tensor2 = tf.cast(tensor2, dtype=tf.uint8)
print("tf.cast(텐서, 자료형)으로 바꾼 자료형: ", tensor2.dtype)



# Tensor를 Numpy로 가져오기
newNP = tensor2.numpy()
print("텐서로부터 numpy로 가져온 newNP: ", newNP)
print("newNP의 shape: ", newNP.shape)
print("newNP의 dtype: ", newNP.dtype)
print("텐서의 정보와 모두 동일하다.")
# 또는 np.array(tensor2)로 해도 된다.
newNP = np.array(tensor2)
print("2. np.array(tensor2): ", newNP)
print("2. newNP의 shape: ", newNP.shape)
print("2. newNP의 dtype: ", newNP.dtype)


## 난수 생성
# numpy에서 np.random.randn() -> normal distribution을 기본적으로 생성
# tf에서는
# tf.random.normal -> Normal distribution -> 낙타 등처럼 평균값이 많이나오는 빈도.
# tf.random.uniform -> Uniform distribution -> 모든 값이 동일 빈도로 나옴

print("1. numpy 의 normal distribution: ", np.random.randn(9))
print("2. tf.random.normal: ", tf.random.normal([3, 3]))
print("3. tf.random.uniform: ", tf.random.uniform([4, 4]))