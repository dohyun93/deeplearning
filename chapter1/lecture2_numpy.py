import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(arr.dtype)

#1. 아래처럼 numpy 형변환 가능.
arr = arr.astype(np.float64)
print(arr.dtype)

# 2. 차원
print(arr.ndim)
print(len(arr))

# 3. size 확인. (size : numbers of elements in the array)
print(arr.size)

# 4. reshape (size 수가 같으면 변환 가능하다.)
# -1은 알아서 계산하게 하기.
arr = arr.reshape(3, 2)
print(arr)
arr = arr.reshape(6)
print(arr)
arr = arr.reshape(2, -1)
print(arr)

# 5. random array 생성
arr = np.random.randn(8, 8)
print(arr)

# 6. 차원 늘리기.
arr = arr.reshape(8, 8, 1, 1, 1)
print(arr.shape) # -> 8x8의 2차원 matrix였는데, 그 이상 3개 차원을 더해 5차원이 됨.

# ravel -> flattening할 수 있다.
# 1차원으로 만들어줌.
arr = arr.ravel()
print(arr.shape)

# 아래는 결국 64개의 True와 같다.
print(arr.ravel() == arr.reshape(-1))

#####################################

# np.expand_dims()
# 안의 값은 유지하고 차원수를 늘리고 싶을 때 사용
arr = np.expand_dims(arr, 0) #맨 앞에 차원 추가.
print(arr.shape)

arr = np.expand_dims(arr, -1) #맨 뒤에 차원 추가.
print(arr.shape)

##################################

# zeros & ones