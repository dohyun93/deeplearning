import numpy as np
# zeros & ones
zeros = np.zeros([2, 3])
print(zeros)

ones = np.ones([3, 4])
print(ones)

# np.arange(n) / np.arange(n, m)
arr = np.arange(5) # 0부터 4까지
print(arr)

arr = np.arange(4, 9) # 4부터 8까지
print(arr)

# 활용예시
arr = np.arange(9).reshape(3, 3)
print(arr)

# index
nums = [1, 2, 3, 4, 5]
print(nums[2:])

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
# 위 matrix의 두 번째 열만 가져와보자.
secondCol = matrix[:, 1]
print(secondCol)
# 위 matrix의 1행 이상 1열 이상만 가져와보자.
mat2 = matrix[1:, 1:]
print(mat2)

# 3x3 random.randn 만들어서 음수는 0으로 만들고 싶다.
data = np.random.randn(3, 3)
print(data)
print(data[data <= 0]) # 인덱스로도 사용가능. 0이하인 요소들 출력.

print(data<=0) # 0이하 여부에 따라 내용 출력
#[[False False False]
# [False  True False]
# [False  True  True]]

# BROADCAST !
# broadcash 는 연산하려는 서로다른 두 개의 행렬의 shape이 같지않고
# 다른 한쪽의 차원이 같거나 또는 값의 갯수가 1개일 때 여러 복사를 해서 연산을 한다.
arr = np.arange(9).reshape(3, 3)
print(arr)
arr = arr + 3 # 값 1개
print(arr)
arr = arr + np.array([1, 2, 3])
print(arr)
# 각 요소들에 더해준다.

# math function
arr = np.random.randint(2, size=27).reshape(3, 3, 3) #0부터 1까지 정수를 랜덤으로.
arr_2 = np.random.randint(2, size=9).reshape(3, 3)
print(arr)
print(arr_2)
#print(arr + arr_2)
#print(np.max(arr+arr_2))
print(arr+arr_2)
print(np.sum(arr + arr_2, 1)) # arr+arr_2의 0차원 기준으로 더해줌.

# argmax
print(np.mean(arr))

array = [6, 1, 2, 4, 100]
array_2d = [[1, 3, 5],
            [8, 3, 2],
            [2, 4, 9]]
print(np.argmax(array_2d)) # 가장 큰 값의 인덱스
print(np.argmin(array_2d)) # 가장 작은 값의 인덱스

array = np.array([3, 5, 6, 6, 3, 3, 1])
print(np.unique(array)) # unique한 값들로 해주는 np.unique