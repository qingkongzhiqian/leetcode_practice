# Divide and Conquer

[TOC]

#### [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

实现 pow(x, n) ，即计算 x 的 n 次幂函数。

```
示例 1:

输入: 2.00000, 10
输出: 1024.00000
示例 2:

输入: 2.10000, 3
输出: 9.26100
示例 3:

输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
```

```python
#Solution 1
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0 : return 1.0
        elif n <= 0:
            return 1 / self.myPow(x,-n)
        elif n % 2:
            return self.myPow(x * x,n // 2) * x
        else:
            return self.myPow(x * x,n // 2)  

#Solution 2
class Solution(object):
    def myPow(self, x, n):

        if n == 0:return 1
        if n < 0:
            x = 1 / x
            n = -n

        ans = 1
        while n:
            if n % 2:
                ans *= x
            n //= 2
            x *= x
        return ans
```

#### [162. 寻找峰值](https://leetcode-cn.com/problems/find-peak-element/)

峰值元素是指其值大于左右相邻值的元素。

给定一个输入数组 nums，其中 nums[i] ≠ nums[i+1]，找到峰值元素并返回其索引。

数组可能包含多个峰值，在这种情况下，返回任何一个峰值所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞。

```python
示例 1:

输入: nums = [1,2,3,1]
输出: 2
解释: 3 是峰值元素，你的函数应该返回其索引 2。
示例 2:

输入: nums = [1,2,1,3,5,6,4]
输出: 1 或 5 
解释: 你的函数可以返回索引 1，其峰值元素为 2；
     或者返回索引 5， 其峰值元素为 6。
说明:

你的解法应该是 O(logN) 时间复杂度的。
```

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) == 0 : return 0
        left,right = 0,len(nums) - 1
        return self.findPeakElement_helper(left,right,nums)

    def findPeakElement_helper(self,left,right,nums):

        if left == right:return left
        if len(nums) == 0 : return 0
        middle = left + (right - left) // 2

        if nums[middle] > nums[middle - 1] and nums[middle] > nums[middle + 1]:
            return middle

        if nums[middle - 1] > nums[middle] and nums[middle] > nums[middle + 1]:
            return self.findPeakElement_helper(left,middle,nums)
        elif nums[middle - 1] < nums[middle] and nums[middle] < nums[middle + 1]:
            return self.findPeakElement_helper(middle + 1,right,nums)
        else:
            return self.findPeakElement_helper(middle + 1,right,nums)
```

#### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

```python
示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
示例 2:

输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4
说明:

你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。

```

```python
#Solution 1 sort
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[len(nums) - k]
```

```python
#Solution 2  bubble sort case
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        for a in range(k):
            for b in range(len(nums) - a - 1):
                if nums[b] > nums[b + 1]:
                    nums[b],nums[b + 1] = nums[b + 1],nums[b]
        return nums[-k] 
```

```python
#Solution 3 quick sort
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        target = len(nums) - k
        left = 0
        right = len(nums) - 1
        while True:
            index = self._partition(nums,left,right)
            if index == target:
                return nums[index]
            elif index < target:
                left = index + 1
            else:
                right = index - 1

    def _partition(self,nums,left,right):

        pivot = nums[left]
        j = left
        for i in range(left + 1,right + 1):
            if nums[i] < pivot:
                j += 1
                nums[i],nums[j] = nums[j],nums[i]
        nums[left],nums[j] = nums[j],nums[left]        

        return j
      
因此我们可以改进快速排序算法来解决这个问题：在分解的过程当中，我们会对子数组进行划分，如果划分得到的 qq 正好就是我们需要的下标，就直接返回 a[q]a[q]；否则，如果 qq 比目标下标小，就递归右子区间，否则递归左子区间。这样就可以把原来递归两个区间变成只递归一个区间，提高了时间效率。这就是「快速选择」算法。
      
```

```python
#Solution 4 heap sort
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        
        return heapq.nlargest(k, nums)[-1] 
```

#### [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

难度：困难

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

```python
示例 1:

输入: [7,5,6,4]
输出: 5


限制：

0 <= 数组长度 <= 50000
```

```python
#Solution 1 O(n^2)
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        count = 0
        for a in range(len(nums)):
            for b in range(a + 1,len(nums)):
                if nums[a] > nums[b]:
                    count +=1
        return count
```

```python
#Solution 2 O(nlogn)
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        
        if 0 <= len(nums) and len(nums) <= 50000:
            result,count = self.reversePairs_helper(nums)
            return count

    def reversePairs_helper(self,nums):

        if len(nums) < 2 : return nums,0
        middle = len(nums) // 2
        left,inv_left = self.reversePairs_helper(nums[:middle])
        right,inv_right = self.reversePairs_helper(nums[middle:])
        merged,count = self._merge(left,right)
        count += (inv_left + inv_right)
        return merged,count

    def _merge(self,left,right):
        result = [] 
        i,j = 0,0
        inv_count = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            elif left[i] > right[j]:
                result.append(right[j])
                j += 1
                inv_count += (len(left) - i)

        result += left[i:]
        result += right[j:]
        return result,inv_count            

```

```python
#Solution 3 bisect overall O(n^2)
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        sortns = []
        result = []
        for n in reversed(nums):
            idx = bisect.bisect_left(sortns, n)
            result.append(idx)
            sortns.insert(idx,n) #O(n)
        return sum(result)
```

https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/solution/bu-dao-10xing-dai-ma-zui-jian-dan-fang-fa-mei-you-/

#### [315. 计算右侧小于当前元素的个数](https://leetcode-cn.com/problems/count-of-smaller-numbers-after-self/)

给定一个整数数组 nums，按要求返回一个新数组 counts。数组 counts 有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。

```
示例：

输入：nums = [5,2,6,1]
输出：[2,1,1,0] 
解释：
5 的右侧有 2 个更小的元素 (2 和 1)
2 的右侧仅有 1 个更小的元素 (1)
6 的右侧有 1 个更小的元素 (1)
1 的右侧有 0 个更小的元素
```

```python
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        
        sortns = []
        result = []
        for n in reversed(nums):
            idx = bisect.bisect_left(sortns, n)
            result.append(idx)
            sortns.insert(idx,n)
        return result[::-1]
```

#### 在已排序数组中找到多余元素的索引

给定两个排好序的数组。这两个数组只有一个不同的地方：在第一个数组某个位 置上多一个元素。请找到这个元素的索引位置。

示例：

```python
Examples:

Input : {2, 4, 6, 8, 9, 10, 12}; {2, 4, 6, 8, 10, 12};

Output : 4
```

思路：

1.遍历的方式，两个循环查找(O(n^2))

2.大循环下使用二分法进行查找(O(nlogn))

3.two-pointer的方式进行(O(n))

4.直接使用二分法进行查找(O(logn))

```python
#Solution 1 O(n^2)
def find_extra(nums1,nums2):
    for a in range(len(nums1)):
        if nums1[a] not in nums2:
            return a
```

```python
#Solution 2 O(nlogn)
def find_extra(nums1,nums2):
    
    for a in range(len(nums1)):
        index = binary_searh(nums2,nums1[a])
        if index == -1:return a
        
def binary_searh(nums,target):
    left,right = 0,len(nums) - 1
    while left + 1 < right:
        middle = left + (right - left) // 2
        if nums[middle] > target:
            right = middle - 1
        elif nums[middle] < target:
            left = middle + 1
        else:
            return middle
    if nums[left] == target: return left
    if nums[right] == target: return right
    return -1
```

```python
#Solution 3 O(n)
def find_extra(nums1,nums2):
    i,j = 0,0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            i += 1
            j += 1
        else:
            return  i
```

```python
#Solution 4 O(logn)
def find_extra(nums1,nums2):
    
    index = len(nums2)
    left,right = 0,len(nums2) - 1
    
    while left <= right:
        middle = left + (right - left) // 2
        if nums1[middle] == nums2[middle]:
            left = middle + 1
        else:
            index = middle
            right = middle - 1
    return index
```

#### [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

```python
示例1:

输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

思路：

1.找出所有的子序列，对所有子序列求和，找到最大值。O(n^3)

2.找出所有的子序列的过程中，直接求和计算，保留最大的。O(n^2)

3.动态规划。O(n)

4.分治法。O(nlogn)

```python
#Solution 1
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:

        if len(nums) == 1:return nums[0]
        result = -float('inf')
        for a in range(len(nums)):
            for b in range(a + 1,len(nums)):
                total = 0
                #遍历区间内的所有子序列
                for k in range(a,b):
                    total += nums[k]
                if total > result:result = total

        return result  
```

```python
#Solution 2
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:

        if len(nums) == 1:return nums[0]
        result = -float('inf')
        for i in range(len(nums)):
            total = 0
            for j in range(i,len(nums)):
                total += nums[j]
                if total > result:
                    result = total
                    
        return result 
```

```python
#DP O(n)
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:

        global_max = -float('inf')
        local_max = 0

        for i in nums:
            local_max = max(local_max + i,i)
            global_max = max(global_max,local_max)
        return global_max 
      
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:

        dp = [0] * len(nums)
        dp[0] = nums[0]

        for i in range(1,len(nums)):
            if dp[i - 1] > 0:
                dp[i] = dp[i - 1] + nums[i]
            else:
                dp[i] = nums[i]
                
        return max(dp)      
```

```python
#Solution 4 分治法
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:

        return self.maxSubArray_helper(nums,0,len(nums) - 1)
    
    def maxSubArray_helper(self,nums,left,right):
        
        if left == right : return nums[left]
        middle = left + (right - left) // 2
        return max(
            self.maxSubArray_helper(nums,left,middle),
            self.maxSubArray_helper(nums,middle + 1,right),
            self.max_crossing(nums,left,middle,right)
        )

    def max_crossing(self,nums,left,middle,right):
        result = 0
        left_sum = -float("inf")
        for i in range(middle,left - 1,-1):
            result += nums[i]
            if result > left_sum:
                left_sum = result

        result = 0
        right_sum = -float("inf")
        for i in range(middle + 1,right + 1):
            result += nums[i]
            if result > right_sum:     
                right_sum = result

        return left_sum + right_sum   
```

#### [1470. 重新排列数组](https://leetcode-cn.com/problems/shuffle-the-array/)

给你一个数组 nums ，数组中有 2n 个元素，按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。

请你将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，返回重排后的数组。

```python
示例 1：

输入：nums = [2,5,1,3,4,7], n = 3
输出：[2,3,5,4,1,7] 
解释：由于 x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 ，所以答案为 [2,3,5,4,1,7]
示例 2：

输入：nums = [1,2,3,4,4,3,2,1], n = 4
输出：[1,4,2,3,3,2,4,1]
示例 3：

输入：nums = [1,1,2,2], n = 2
输出：[1,2,1,2
```

```python
#Solution 1 
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        
        result = []
        left,middle = 0,n
        for i in range(n):
            result.append(nums[left])
            result.append(nums[middle])
            left += 1
            middle += 1

        return result 
```

思路二：

分治法，[a1,a2,a3,a4,b1,b2,b3,b4],拆分一半后换序，[a1,a2,b1,b2,a3,a4,b3,b4],

然后接着拆分换序，[a1,b1,a2,b2,a3,b3,a4,b4]。

```python
#Solution 2 代码有问题
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:

        left,right = 0,len(nums) - 1
        self.shuffle_helper(nums,left,right)
        return nums
        
    def shuffle_helper(self,nums,left,right):

        if right - left == 1:return 
        middle  = left + (right - left) // 2
        temp = middle + 1
        mmid = (left + middle) // 2
        for i in range(mmid + 1,middle + 1):
            nums[i],nums[temp] = nums[temp],nums[i]
            temp += 1
        self.shuffle_helper(nums,left,middle)
        self.shuffle_helper(nums,middle + 1,right)
```

思路三：zip

```python
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        return sum(zip(nums[:n], nums[n:]), ())
```

