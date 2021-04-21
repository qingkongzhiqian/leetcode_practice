

# ArrayList

[TOC]

#### [矩阵置零](https://leetcode-cn.com/problems/set-matrix-zeroes/)

给定一个 *m* x *n* 的矩阵，如果一个元素为 0，则将其所在行和列的所有元素都设为 0。请使用**[原地](http://baike.baidu.com/item/原地算法)**算法**。**

示例 1:

```python
输入: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
输出: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]
```


示例 2:

```python
输入: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
输出: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]
```

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row = [1] * len(matrix)
        column = [1] * len(matrix[0])

        for a in range(len(matrix)):
            for b in range(len(matrix[a])):
                if matrix[a][b] == 0:
                    row[a] = 0
                    column[b] = 0

        for a in range(len(matrix)):
            for b in range(len(matrix[a])):
                if row[a] == 0 : matrix[a][b] = 0
                if column[b] == 0 : matrix[a][b] = 0

        return matrix 
```

#### [有效的数独](https://leetcode-cn.com/problems/valid-sudoku/)

判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。

数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。


上图是一个部分填充的有效的数独。

数独部分空格内已填入了数字，空白格用 '.' 表示。

示例 1:

```
输入:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: true
```


示例 2:

```
输入:
[
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: false
```

解释: 除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。

```python
class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # init data
        rows = [{} for i in range(9)]
        columns = [{} for i in range(9)]
        boxes = [{} for i in range(9)]

        # validate a board
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num != '.':
                    num = int(num)
                    box_index = (i // 3 ) * 3 + j // 3
                    # keep the current cell value
                    rows[i][num] = rows[i].get(num, 0) + 1
                    columns[j][num] = columns[j].get(num, 0) + 1
                    boxes[box_index][num] = boxes[box_index].get(num, 0) + 1
                    
                    # check if this value has been already seen before
                    if rows[i][num] > 1 or columns[j][num] > 1 or boxes[box_index][num] > 1:
                        return False         
        return True
```

#### [旋转矩阵](https://leetcode-cn.com/problems/rotate-matrix-lcci/)

给你一幅由 N × N 矩阵表示的图像，其中每个像素的大小为 4 字节。请你设计一种算法，将图像旋转 90 度。

不占用额外内存空间能否做到?

示例 1:

```python
给定 matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

原地旋转输入矩阵，使其变为:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
```


示例 2:

```python
给定 matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

原地旋转输入矩阵，使其变为:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
```

```python
class Solution:
    #do not in-place
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        result = [[0] * (n) for i in range(n)]

        for i in range(n):
            for j in range(n):
                result[j][n-1-i] = matrix[i][j]

        return result   
```

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """

        length = len(matrix)
        # 先在纵向上进行上下翻转
        # 切片会创建新的对象进而开辟新地址
        matrix[:] = matrix[::-1]
        # 然后沿对角线翻转
        for i in range(length):
            for j in range(i):
                matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]
```

#### [反转字符串](https://leetcode-cn.com/problems/reverse-string/)

编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。

```

示例 1：

输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]
示例 2：

输入：["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]

```

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """

        if not s:return -1
        s[:] = s[::-1]
        return s
```

思路二：双指针

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        if not s:return -1
        left,right = 0,len(s) -1
        while left < right:
            s[left],s[right] = s[right],s[left]
            left += 1
            right -= 1
        return s    
```

#### [反转字符串 II](https://leetcode-cn.com/problems/reverse-string-ii/)

给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。

如果剩余字符少于 k 个，则将剩余字符全部反转。
如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。

```
示例:

输入: s = "abcdefg", k = 2
输出: "bacdfeg"

```

```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:

        a = list(s)
        for i in range(0,len(a),2*k):
            a[i:i + k] = reversed(a[i:i+k])
        return "".join(a)    

```

#### 最长连续子串

给一个只包含0和1的数组，找出最长的全是1的子数组。

Example:

Input: [1,1,0,1,1,1]

Output: 3

```python
def find_consecutive_ones(nums):
    local = maximum = 0
    for i in nums:
        local = local + 1 if i == 1 else 0
        maximum = max(maximum, local)
    return maximum
```

#### 最大数

给定一个数组，数组里有且只有一个最大数，判断这个最大数是否是其他数的两倍或更大。如果存在这个数，则返回其index，否则返回-1。

```python
def largest_twice(nums):
    maximum = second = idx = 0
    for i in range(len(nums)):
        if (maximum < nums[i]):
            second = maximum
            maximum = nums[i]
            idx = i
        elif second < nums[i]:
            second = nums[i]
    return idx if (maximum >= second * 2) else -1
```

#### [找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)

给定一个范围在  1 ≤ a[i] ≤ n ( n = 数组大小 ) 的 整型数组，数组中的元素一些出现了两次，另一些只出现一次。找到所有在 [1, n] 范围之间没有出现在数组中的数字。

您能在不使用额外空间且时间复杂度为O(n)的情况下完成这个任务吗? 你可以假定返回的数组不算在额外空间内。

```python
输入:
[4,3,2,7,8,2,3,1]

输出:
[5,6]
```

```python
#case 1 use list count sort
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        a = [0] * len(nums)
        for i in nums:
            a[i - 1] += 1
        result = []
        for i in range(len(a)):
            if a[i] == 0 : result.append(i + 1)
        return result
```

```python
#case2 in-place
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:

        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = -abs(nums[index])
        return [i+1 for i in range(len(nums)) if nums[i] > 0]  
```

```python
#集合思想
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:

        return list(set(range(1,len(nums) + 1)) - set(nums))
```

#### [加一](https://leetcode-cn.com/problems/plus-one/)

给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。

示例 1：

```
输入：digits = [1,2,3]
输出：[1,2,4]
解释：输入数组表示数字 123。
```


示例 2：

```
输入：digits = [4,3,2,1]
输出：[4,3,2,2]
解释：输入数组表示数字 4321。
```


示例 3：

```
输入：digits = [0]
输出：[1]
```

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if len(digits)==0:return False
        addCarry=1
        for i in range(len(digits)-1,-1,-1):
            digits[i]+=addCarry
            if digits[i]==10:
                digits[i]=0
                if i==0:
                    digits.insert(0,1)
            else:
                break #精髓
        return digits
```

解释：如果当前的位置为10，则将当前的位置置为0，循环再向前走一步，若当前是首字符了，则前面插入一个1

#### [剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

```
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

示例 1：

输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
示例 2：

输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]

```

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:

        res = []
        while matrix:
            res += matrix.pop(0)
            matrix = list(zip(*matrix))[::-1]
        return res 
```

