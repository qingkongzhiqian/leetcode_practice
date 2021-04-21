# Skill

[TOC]

### 1.摩尔投票法

#### [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```
示例 1:

输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
输出: 2


```

```python
from collections import Counter
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        
        count,candidate = 0,None

        for i in nums:
            if count == 0:
                count += 1
                candidate = i
            else:
                if candidate == i:
                    count += 1
                else:
                    count -= 1

        return candidate 
```

```python
from collections import Counter
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        
        count = Counter(nums)

        return [ i for i in count if count[i] > len(nums) // 2][0]
```

### 2.二分法定位区间

#### [287. 寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

```
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

### 3.三段反转法

#### 移位 II 

 写一个函数 rotate(arr[], d, n) 将大小为n的数组arr[] 移位d个单位。

思路：三段反转法，首先对0 -> k -1进行反转，然后对k -> n -1进行反转，最后将整个数组反转

```python
def reverse(arr,start,end):
    while start < end:
        arr[start],arr[end] = arr[end],arr[start]
        start += 1
        end -= 1

def rotate(arr,d):
    n = len(arr)
    reverse(arr,0,d - 1)
    reverse(arr,d,n - 1)
    reverse(arr,0,n - 1)
    return arr

arr = [1, 2, 3, 4, 5, 6, 7]
print ("aee",rotate(arr,2))

>> aee [3, 4, 5, 6, 7, 1, 2]
```

### 4.Rolling-hash

https://leetcode-cn.com/problems/implement-strstr/solution/shi-xian-strstr-by-leetcode/

#### [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)

给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

```
示例 1:

输入: haystack = "hello", needle = "ll"
输出: 2
示例 2:

输入: haystack = "aaaaa", needle = "bba"
输出: -1


```

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        L, n = len(needle), len(haystack)
        if L > n:
            return -1
        
        # base value for the rolling hash function
        a = 26
        # modulus value for the rolling hash function to avoid overflow
        modulus = 2**31
        
        # lambda-function to convert character to integer
        h_to_int = lambda i : ord(haystack[i]) - ord('a')
        needle_to_int = lambda i : ord(needle[i]) - ord('a')
        
        # compute the hash of strings haystack[:L], needle[:L]
        h = ref_h = 0
        for i in range(L):
            h = (h * a + h_to_int(i)) % modulus
            ref_h = (ref_h * a + needle_to_int(i)) % modulus
        if h == ref_h:
            return 0
              
        # const value to be used often : a**L % modulus
        aL = pow(a, L, modulus) 
        for start in range(1, n - L + 1):
            # compute rolling hash in O(1) time
            h = (h * a - h_to_int(start - 1) * aL + h_to_int(start + L - 1)) % modulus
            if h == ref_h:
                return start

        return -1
```

### 5.斐波那契数列优化



```python
import numpy as np
class Solution:
    def climbStairs(self, n: int) -> int:
            result_list = [0,1,2]
            if n <= 2:return result_list[n]
            for i in range(3,n + 1):
                result_list.append(int(np.array(self.fibonacci_matrix_tool2(i))[0][0]))
            return result_list[n]

    def fibonacci_matrix_tool1(self,n):               
        Matrix = np.matrix('1 1;1 0')
        return pow(Matrix,n)

    def fibonacci_matrix_tool2(self,n):
        Matrix = np.matrix('1 1;1 0')
        if n == 1:return Matrix
        if n % 2 == 1:
            return self.fibonacci_matrix_tool2((n - 1) / 2) ** 2 * Matrix
        else:
            return self.fibonacci_matrix_tool2(n / 2) ** 2
```

### 6.桶排序思路

#### [164. 最大间距](https://leetcode-cn.com/problems/maximum-gap/)

给定一个无序的数组，找出数组在排序之后，相邻元素之间最大的差值。

如果数组元素个数小于 2，则返回 0。时间复杂度要求 O(n)。

```
示例 1:

输入: [3,6,9,1]
输出: 3
解释: 排序后的数组是 [1,3,6,9], 其中相邻元素 (3,6) 和 (6,9) 之间都存在最大差值 3。
示例 2:

输入: [10]
输出: 0
解释: 数组元素个数小于 2，因此返回 0。

```

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        if len(nums) < 2: return 0

        mmax = max(nums)
        mmin = min(nums)
        gap = 0

        bucket_len = max(1,(mmax - mmin) // (len(nums) - 1))
        buckets = [[] for _ in range((mmax - mmin) // bucket_len + 1)]

        for i in range(len(nums)):
            index = (nums[i] - mmin) // bucket_len
            buckets[index].append(nums[i])

        pre_max = float('inf')
        result = 0
        for i in range(len(buckets)):
            if buckets[i] and pre_max != float("inf"):
                result = max(result,min(buckets[i]) - pre_max)

            if buckets[i]:
                pre_max = max(buckets[i])

        return result 
```

