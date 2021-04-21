# Binary Search

[TOC]

#### 二分查找模板

```Python
#iterative
def binary_search(nums,item):
  if len(nums) == 0 : return -1
  left,right = 0 ,len(nums) - 1
  while left + 1 < right:
    middle = (left + right) // 2
    if nums[middle] == item:
      right = middle
    elif nums[middle] < item:
      left = middle
    else:
      right = middle
  if nums[left] == item : return left
  if nums[right] == item : return right
  return -1

#recursive
def bi_search_recursive(nums,item):
    left,right = 0,len(nums) - 1
    
    def bi_search_helper(left,right):
        if left > right:return -1
        middle = (left + right) // 2
        if nums[middle] > item:
            return bi_search_helper(left,middle - 1)
        elif nums[middle] < item:
            return bi_search_helper(middle + 1,right)
        else:
            return middle

    return bi_search_helper(left,right) 

```

#### [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

```python
示例 1:

输入: 4
输出: 2
示例 2:

输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。
```

```python
#Solution 1
class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0 : return 0
        left,right = 1,x
        while left <= right:
            middle = left + (right - left) // 2
            if middle * middle == x:
                return middle
            elif middle * middle > x:
                right = middle - 1
            else:
                left = middle + 1    
        return right
      
#Solution 2      
class Solution:
    def mySqrt(self, x: int) -> int:
        r = x
        while r*r > x:
            r = (r + x//r) // 2
        return r      
```

牛顿法求解：

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0:return 0

        a,x0 = float(x),float(x)

        while True:
            xi = 0.5 * (x0 + a / x0)
            if abs(x0 - xi) < 1e-7:
                break
            x0 = xi 

        return int(x0) 
```

#### [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

```
示例 1:

输入: [0,1,3]
输出: 2
示例 2:

输入: [0,1,2,3,4,5,6,7,9]
输出: 8


```

思路一：二分查找

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:

        if not nums:return -1

        left,right = 0,len(nums) - 1
        while left <= right:
            middle = (right + left) // 2
            if middle == nums[middle]:
                left = middle + 1
            else:
                right = middle - 1

        return left 
```

思路二：求和  

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        return sum(range(len(nums) +1)) - sum(nums)       
```

#### [剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

```python
示例 1：

输入：[3,4,5,1,2]
输出：1
示例 2：

输入：[2,2,2,0,1]
输出：0
```

```python
#Solution 1  O(n)
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        return min(numbers)
           
#Solution 2 O(nlogn)
class Solution:
    def minArray(self, numbers: List[int]) -> int:
      	numbers.sort()
        return numbers[0]
      
#Solution 3 O(logn)
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        left, right = 0, len(numbers) - 1
        while left < right:
            middle = left + (right - left) // 2
            if numbers[middle] < numbers[right]:
                right = middle  #********
            elif numbers[middle] > numbers[right]:
                left = middle + 1
            else:
                right -= 1
        return numbers[left]
```

#### [154. 寻找旋转排序数组中的最小值 II](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array-ii/)

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

请找出其中最小的元素。

注意数组中可能存在重复的元素。

```python
示例 1：

输入: [1,3,5]
输出: 1
示例 2：

输入: [2,2,2,0,1]
输出: 0
```

```python
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        left, right = 0, len(numbers) - 1
        while left < right:
            middle = left + (right - left) // 2
            if numbers[middle] < numbers[right]:
                right = middle 
            elif numbers[middle] > numbers[right]:
                left = middle + 1
            else:
                right -= 1
        return numbers[left]
```

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        return min(nums)
```

#### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

给你一个整数数组 nums ，和一个整数 target 。

该整数数组原本是按升序排列，但输入时在预先未知的某个点上进行了旋转。（例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] ）。

请你在数组中搜索 target ，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

提示：

1 <= nums.length <= 5000
-10^4 <= nums[i] <= 10^4
nums 中的每个值都 独一无二
nums 肯定会在某个点上旋转
-10^4 <= target <= 10^4

```
示例 1：

输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
示例 2：

输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
示例 3：

输入：nums = [1], target = 0
输出：-1
```

```python
#Solution 1
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if target in nums:
            return nums.index(target)
        else:
            return -1 
```

```python
#Solution 2
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) <= 0 : return -1
        left,right = 0 ,len(nums) - 1
        while left + 1 < right:
            middle = left + (right - left) // 2
            if nums[middle] == target : return middle
            if nums[left] < nums[middle]:
                if nums[left]  <= target and target <= nums[middle]:
                    right = middle
                else:
                    left = middle
            else:
                if nums[right] >= target and target >= nums[middle]:
                    left = middle
                else:
                    right = middle
        if nums[left] == target : return left
        if nums[right] == target : return right
        return -1     

```

#### [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2] )。

编写一个函数来判断给定的目标值是否存在于数组中。若存在返回 true，否则返回 false。

```python
示例 1:

输入: nums = [2,5,6,0,0,1,2], target = 0
输出: true
示例 2:

输入: nums = [2,5,6,0,0,1,2], target = 3
输出: false

```

**进阶:**

- 这是 [搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/description/) 的延伸题目，本题中的 `nums` 可能包含重复元素。
- 这会影响到程序的时间复杂度吗？会有怎样的影响，为什么

```python
#Solution 1
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        if len(nums) <= 0 : return False
        left,right = 0,len(nums) - 1
        while left + 1 < right:
            middle = left + (right - left) // 2
            if target == nums[middle] : return True
            if target == nums[left] : return True
            if target == nums[right] : return True
            if nums[left] < nums[middle]:
                if target > nums[left] and target < nums[middle]:
                    right = middle
                else:
                    left = middle    
            elif nums[left] > nums[middle]:
                if target > nums[middle] and target < nums[right]:
                    left = middle
                else:
                    right = middle    
            else:        
                right -= 1 #****有重复元素时，需要移动的指针

        if nums[left] == target : return True
        if nums[right] == target : return True
        return False 
```

#### [35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

你可以假设数组中无重复元素。

```python
示例 1:

输入: [1,3,5,6], 5
输出: 2
示例 2:

输入: [1,3,5,6], 2
输出: 1
示例 3:

输入: [1,3,5,6], 7
输出: 4
示例 4:

输入: [1,3,5,6], 0
输出: 0
```

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:return 0  
        left, right = 0, len(nums) - 1
        while left + 1 < right:    #边界条件判断
            mid = left + (right - left) // 2
            if nums[mid] == target:return mid
            if (nums[mid] < target):
                left = mid
            else:
                right = mid
                
        if nums[left] >= target:return left
        if nums[right] >= target:return right
        return right + 1
```

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        import bisect
        return bisect.bisect_left(nums,target)
```

#### [34. 在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

你的算法时间复杂度必须是 O(log n) 级别。

如果数组中不存在目标值，返回 [-1, -1]。

```python
示例 1:

输入: nums = [5,7,7,8,8,10], target = 8
输出: [3,4]
示例 2:

输入: nums = [5,7,7,8,8,10], target = 6
输出: [-1,-1]


```

思路：通过两次二分查找，一次查找第一次出现的，一次查找最后一次出现的

时间复杂度为O(logN) + O(logN) = O(logN)

注意：寻找左边界的时候，需要先判断left指针。寻找右边界的时候，需要先判断right指针。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0 : return [-1,-1]
        left_bound,right_bound = -1,-1
        left,right = 0,len(nums) - 1
        while left + 1 < right:
            middle = left + (right - left) // 2
            if nums[middle] == target:
                right = middle
            elif nums[middle] < target:
                left = middle
            else:
                right = middle
        if nums[left] == target:
            left_bound = left
        elif nums[right] == target:
            left_bound = right
        else:
            return [-1,-1]

        left,right = 0,len(nums) - 1
        while left + 1 < right:
            middle = left + (right - left) // 2
            if nums[middle] == target:
                left = middle
            elif nums[middle] < target:
                left = middle
            else:
                right = middle

        if nums[right] == target:
            right_bound = right
        elif nums[left] == target:
            right_bound = left
        else:
            return [-1,-1]           
        return [left_bound,right_bound]   
      
#两个循环的细节，left = middle right = middle 注意      
```

```python
import bisect
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:

        left = bisect.bisect_left(nums,target)
        right = bisect.bisect_right(nums,target)

        if right <= left:return [-1,-1]

        return [left,right - 1]
```

#### 在用空字符串隔开的字符串的有序列中查找

给定一个有序的字符串序列，这个序列中的字符串用空字符隔开，请写出找到给定字符串位置的方法；

```python
def search_empty(alist, target):
    if len(alist) == 0:
        return -1
    left, right = 0, len(alist) - 1
    while left + 1 < right:
        while left + 1 < right and alist[right] == "":
            right -= 1
        if alist[right] == "":
            right -= 1
        if right < left:
            return -1
        mid = left + (right - left) // 2
        while alist[mid] == "":
            mid += 1
        if alist[mid] == target:
            return mid
        if alist[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    if alist[left] == target:
        return left
    if alist[right] == target:
        return right    
    return -1   
```

#### 在无限序列中找到某元素的第一个出现位置

数据流 l,不知道序列长度，不能使用len()来得到

思路：倍增法

```python
def search_first(alist):
    left, right = 0, 1
    while alist[right] == 0:
        left = right
        right *= 2
        if (right > len(alist)):
            right = len(alist) - 1
            break
    return left + search_range(alist[left:right+1], 1)[0]
```

#### [475. 供暖器](https://leetcode-cn.com/problems/heaters/)

冬季已经来临。 你的任务是设计一个有固定加热半径的供暖器向所有房屋供暖。

在加热器的加热半径范围内的每个房屋都可以获得供暖。

现在，给出位于一条水平线上的房屋 houses 和供暖器 heaters 的位置，请你找出并返回可以覆盖所有房屋的最小加热半径。

说明：所有供暖器都遵循你的半径标准，加热的半径也一样。

```python
示例 1:

输入: houses = [1,2,3], heaters = [2]
输出: 1
解释: 仅在位置2上有一个供暖器。如果我们将加热半径设为1，那么所有房屋就都能得到供暖。
示例 2:

输入: houses = [1,2,3,4], heaters = [1,4]
输出: 1
解释: 在位置1, 4上有两个供暖器。我们需要将加热半径设为1，这样所有房屋就都能得到供暖。
示例 3：

输入：houses = [1,5], heaters = [2]
输出：3
```

```python
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        from bisect import bisect
        heaters.sort()
        ans = 0

        for h in houses:
            hi = bisect(heaters, h)
            left = heaters[hi-1] if hi - 1 >= 0 else float('-inf')
            right = heaters[hi] if hi < len(heaters) else float('inf')
            ans = max(ans, min(h - left, right - h))

        return ans
```

补充：

```python
bisect函数：其目的在于查找该数值将会插入的位置并返回，而不会插入。
找到第一个大于等于的数字
1.首先对于每一个房子来说，要找到每一个房子的最近的供暖器的位置
2.在这些最近的供暖期的位置中找到最大值。
```

#### [240. 搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

```python
示例:

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。

给定 target = 20，返回 false。

```

思路：在矩阵中寻找只有两个走向的路径，(一个方向上减少，一个方向上增加)，例如矩阵中的18，上减右增。

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if len(matrix) == 0 : return False
        row = len(matrix) - 1
        column = len(matrix[0]) -1 

        a,b = row,0
        while a >= 0 and b <= column:
            if matrix[a][b] == target: 
                return True
            elif matrix[a][b] > target :
                a -= 1
            else:
                b += 1
        return False 
```

#### [74. 搜索二维矩阵](https://leetcode-cn.com/problems/search-a-2d-matrix/)

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。

```python
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 3
输出：true

输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,50]], target = 13
输出：false

输入：matrix = [], target = 0
输出：false
```

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        if len(matrix) == 0 : return False
        nums = []
        for i in matrix:
            nums += i
        if len(nums) == 0 : return False    
        left,right = 0,len(nums) - 1
        while left + 1 < right:
            middle = left + (right - left) // 2
            if nums[middle] == target : return True
            if nums[middle] > target:
                right = middle
            else:
                left = middle

        if nums[left] == target or nums[right] == target:return True     
        return False  
```

#### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

```python
示例 1:

输入: [1,3,4,2,2]
输出: 2
示例 2:

输入: [3,1,3,4,2]
输出: 3


```

```
说明：
不能更改原数组（假设数组是只读的）。
只能使用额外的 O(1) 的空间。
时间复杂度小于 O(n2) 。
数组中只有一个重复的数字，但它可能不止重复出现一次。
```

```python
#Solution 1 count sort
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:

        count = [0] * (len(nums))
        for i in nums:
            count[i] += 1

        for i in range(1,len(count)):
            if count[i] > 1 : return i 
            
        return -1  
```

思路：使用二分的方式，来确定重复数字的区间范围，进一步缩小。

https://leetcode-cn.com/problems/find-the-duplicate-number/solution/er-fen-fa-si-lu-ji-dai-ma-python-by-liweiwei1419/

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:

        left,right = 1,len(nums) - 1
        while left < right:
            middle = left + (right - left) // 2
            min_count = 0
            for i in nums:
                if i <= middle:
                    min_count += 1
            if min_count <= middle:
                left = middle + 1
            else:
                right = middle
        return left 
```

#### [56. 合并区间](https://leetcode-cn.com/problems/merge-intervals/)

给出一个区间的集合，请合并所有重叠的区间。

```python
示例 1:

输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
示例 2:

输入: intervals = [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
注意：输入类型已于2019年4月15日更改。 请重置默认代码定义以获取新方法签名。

提示：

intervals[i][0] <= intervals[i
```

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged
```



