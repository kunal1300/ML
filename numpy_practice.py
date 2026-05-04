import numpy as np

print("--- 1. Array Creation ---")
# Q1: Create a 1D array and a 2D matrix
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr_1d)

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr_2d)

# Q2: Create an array of zeros, ones, and a numbered sequence
zeros = np.zeros((2, 3))  # 2 rows, 3 columns of zeros
ones = np.ones((3, 2))    # 3 rows, 2 columns of ones
seq = np.arange(0, 10, 2) # Sequence from 0 to 10 with step size 2
print("Zeros:\n", zeros)
print("Sequence:", seq)

print("\n--- 2. Array Attributes ---")
# Q3: Find the shape, size, and data type of the 2D array
print("Shape of arr_2d:", arr_2d.shape)
print("Size of arr_2d (Total elements):", arr_2d.size)
print("Data type of arr_2d:", arr_2d.dtype)

print("\n--- 3. Indexing and Slicing ---")
# Q4: Access specific elements and slice parts of an array
print("First element of 1D array:", arr_1d[0])
print("Last two elements of 1D array:", arr_1d[-2:])
print("Element at row 0, col 1 in 2D array:", arr_2d[0, 1])
print("First column of 2D array:", arr_2d[:, 0])

print("\n--- 4. Math Operations ---")
# Q5: Perform element-wise addition, multiplication, and compute the dot product
arr_a = np.array([10, 20, 30])
arr_b = np.array([1, 2, 3])
print("Element-wise Addition:", arr_a + arr_b)
print("Element-wise Multiplication:", arr_a * arr_b)
print("Dot Product (10*1 + 20*2 + 30*3):", np.dot(arr_a, arr_b))

print("\n--- 5. Statistical Operations ---")
# Q6: Calculate the mean, max, and standard deviation of an array
data = np.array([15, 25, 35, 45, 55])
print("Data:", data)
print("Mean (Average):", np.mean(data))
print("Max Value:", np.max(data))
print("Standard Deviation:", np.std(data))

print("\n--- 6. Reshaping Arrays ---")
# Q7: Reshape a 1D array of 9 elements into a 3x3 matrix
flat_arr = np.arange(1, 10)
reshaped_arr = flat_arr.reshape((3, 3))
print("Original 1D (1 to 9):\n", flat_arr)
print("Reshaped to 3x3 Matrix:\n", reshaped_arr)

print("\n--- 7. Filtering and Boolean Indexing ---")
# Q8: Filter out numbers from an array that are less than or equal to 20
test_scores = np.array([12, 45, 67, 19, 85, 23])
passing_scores = test_scores[test_scores > 20]
print("All scores:", test_scores)
print("Scores greater than 20:", passing_scores)

# Q9: Replace all values less than 30 with 0
test_scores[test_scores < 30] = 0
print("Scores after replacing values < 30 with 0:", test_scores)
