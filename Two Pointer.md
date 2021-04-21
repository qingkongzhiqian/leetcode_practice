# Two Pointer

[TOC]

### Two Pointer

#### [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

```
示例 1：

输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
示例 2：

输入：nums = [10,26,30,31,47,60], target = 40
输出：[10,30] 或者 [30,10]

```

思路：双指针算法

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i,j = 0 ,len(nums) - 1
        while i < j:
            count = nums[i] + nums[j]
            if target < count:
                j -= 1
            elif target > count:
                i += 1
            else:
                return[nums[i],nums[j]]  
```

#### [15. 三数之和](https://leetcode-cn.com/problems/3sum/)

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有满足条件且不重复的三元组。

注意：答案中不可以包含重复的三元组。

```python
示例：

给定数组 nums = [-1, 0, 1, 2, -1, -4]，

满足要求的三元组集合为：
[
  [-1, 0, 1],
  [-1, -1, 2]
]

```

思路：单纯的三层for循环的时间复杂度太高，使用第一层for循环作为第一个指针，然后做法类似two pointers的方式进行。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        for a in range(len(nums) - 2):
            if a > 0 and nums[a] == nums[a - 1]:continue
            b,c = a + 1,len(nums) - 1
            while b < c:
                if nums[a] + nums[b] + nums[c] == 0:
                    result.append([nums[a],nums[b],nums[c]])
                    while b < c and nums[b] == nums[b + 1]: #防止重复的元素进入
                        b += 1
                    while b < c and nums[c] == nums[c - 1]:    
                        c -= 1
                    b += 1
                    c -= 1    

                elif nums[a] + nums[b] + nums[c] > 0:
                    c -= 1
                else:
                    b += 1

        return result
```

#### [18. 四数之和](https://leetcode-cn.com/problems/4sum/)

给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

注意：

答案中不可以包含重复的四元组。

示例：

```python
给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。

满足要求的四元组集合为：
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]

```

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:continue
            for j in range(i + 1,len(nums)):
                if j > i + 1 and nums[j] == nums[j - 1]:continue
                l = j + 1
                r = len(nums) - 1
                while l < r:
                    count = nums[i] + nums[j] + nums[l] + nums[r]
                    if count > target:
                        r -= 1
                    elif count < target:
                        l += 1
                    elif l > j + 1 and nums[l] == nums[l-1]:
                        l += 1
                    elif r < len(nums) - 1 and nums[r] == nums[r+1]:
                        r -= 1        
                    else:
                        result.append([nums[i],nums[j],nums[l],nums[r]])
                        l += 1
                        r -= 1
        return result 
```

#### [88. 合并两个有序数组](https://leetcode-cn.com/problems/merge-sorted-array/)

给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。

说明：

初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。

```python
输入：
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

输出：[1,2,2,3,5,6]
```

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        #思路：从后向前走，比较哪一个大，放在最后
        while m > 0 and n > 0:
            if nums1[m - 1] >= nums2[n - 1]:
                nums1[m + n -1] = nums1[m - 1]
                m -= 1
            else:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1

        if n > 0: #nums1 = [5,6,0,0,0] nums2 = [1,2]
            nums1[:n] = nums2[:n]
```

#### 两有序数组的最小元素差

问题描述：有两个升序数组A和B, 求|A[i]-B[j]|的最小值

```python
def printClosest(arr1,arr2):
		m = len(arr1)
  	n = len(arr2)
    diff = float('inf')
    
    p1,p2 = 0,0
    
    while p1 < m and p2 < n:
      if abs(arr1[p1] - arr2[p2]) < diff:
        diff = abs(arr1[p1] - arr2[p2])
      
      if arr1[p1] > arr2[p2]:  #当前元素小于arr1[p1],向下找一个比arr[p2]大的对比
        	p2 += 1
      else:
        	p1 += 1
		return diff          
```

#### [面试题 17.10. 主要元素](https://leetcode-cn.com/problems/find-majority-element-lcci/)

数组中占比超过一半的元素称之为主要元素。给定一个**整数**数组，找到它的主要元素。若没有，返回-1。

你有办法在时间复杂度为 O(N)，空间复杂度为 O(1) 内完成吗？

**示例 1：**

```python
输入：[1,2,5,9,5,9,5,5,5]
输出：5

输入：[3,2]
输出：-1

输入：[2,2,1,1,1,2,2]
输出：2
```

思路一：count计数

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:

        from collections import Counter
        count = Counter(nums)

        target = [i for i in count if count[i] > (len(nums) // 2)]

        return target[0] if target else -1
```

思路二：摩尔投票算法

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:

        if not nums:return -1

        count = 0
        candidate = None #候选

        for i in nums:
            if count == 0:
                count += 1
                candidate = i
            else:
                if candidate == i:
                    count += 1
                else:
                    count -= 1

        if count == 0:return -1

        identify = 0
        for num in nums:
            if num == candidate:
                identify += 1
                if identify > len(nums) // 2:
                    return candidate
        return -1            
```

#### 寻找主元素 II

问题描述：已知一个大小为n的数组，要求找出数组中出现次数超过数组长度1/3 的元素。

思路一：排序，hashTable都可以

思路二：摩尔投票算法

```python
def majority2(alist):
    n1 = n2 = None
    c1 = c2 = 0
    for num in alist:
        if n1 == num:
            c1 += 1
        elif n2 == num:
            c2 += 1
        elif c1 == 0:
            n1,c1 = num,1
        elif c2 == 0:
            n2,c2 = num,1
        else:
            c1,c2 = c1 - 1,c2 - 1
    size = len(alist)
    return [n for n in (n1,n2) if n is not None and alist.count(n) > size / 3]
```

#### [75. 颜色分类](https://leetcode-cn.com/problems/sort-colors/)

给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

```python
进阶：

你可以不使用代码库中的排序函数来解决这道题吗？
你能想出一个仅使用常数空间的一趟扫描算法吗？

示例 1：

输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
示例 2：

输入：nums = [2,0,1]
输出：[0,1,2]
示例 3：

输入：nums = [0]
输出：[0]
示例 4：

输入：nums = [1]
输出：[1]

```

思路一：单指针

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        ptr = 0
        for i in range(n):
            if nums[i] == 0:
                nums[i], nums[ptr] = nums[ptr], nums[i]
                ptr += 1
        for i in range(ptr, n):
            if nums[i] == 1:
                nums[i], nums[ptr] = nums[ptr], nums[i]
                ptr += 1
				return nums                
```

思路二：双指针

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        n = len(nums)
        p0, p2 = 0, n - 1
        i = 0
        while i <= p2:
            while i <= p2 and nums[i] == 2:
                nums[i], nums[p2] = nums[p2], nums[i]
                p2 -= 1
            if nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                p0 += 1
            i += 1

```

#### [658. 找到 K 个最接近的元素](https://leetcode-cn.com/problems/find-k-closest-elements/)

给定一个排序好的数组 arr ，两个整数 k 和 x ，从数组中找到最靠近 x（两数之差最小）的 k 个数。返回的结果必须要是按升序排好的。

整数 a 比整数 b 更接近 x 需要满足：

|a - x| < |b - x| 或者
|a - x| == |b - x| 且 a < b

```python
示例 1：

输入：arr = [1,2,3,4,5], k = 4, x = 3
输出：[1,2,3,4]
示例 2：

输入：arr = [1,2,3,4,5], k = 4, x = -1
输出：[1,2,3,4]

```

思路一：排序，按照差值进行排序

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:

        diff = [(abs(i - x),i) for i in arr]
        diff.sort()
        return sorted([i[1] for i in diff[:k]])
```

思路二：Two pointer + 二分查找

```python
import bisect

class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:

        left = right = bisect.bisect_left(arr,x)
        while right - left < k:
            if left == 0:return arr[:k]
            if right == len(arr):return arr[-k:]
            if x - arr[left - 1] <= arr[right] - x:
              left -= 1
            else:
                right += 1
        return arr[left:right]  

#直接定位出要插入的中点，      
```

#### [11. 盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

给你 n 个非负整数 a1，a2，...，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0) 。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。

说明：你不能倾斜容器。

```python
输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。

```

思路一：两次循环记录最大值

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:

        result = 0
        for i in range(len(height)):
            for j in range(i + 1,len(height)):
                result = max((j - i) * min(height[i],height[j]),result)
        return result 
```

思路二：双指针

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:

        left,right = 0,len(height) - 1
        result = 0
        count = 0
        while left < right:
            count = (right - left) * min(height[right],height[left])
            result = max(count,result)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return result
```

#### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

给定 *n* 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

```python
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
示例 2：

输入：height = [4,2,0,3,2,5]
输出：9

```

思路：我们遍历每个下标，寻找它左边和右边最高的柱子，判断是否可以接到水，将可接水的结果累加即可。

```
/*
        使用两个指针，一个 left_max ，一个 right_max

        这个双指针是怎么个使用法呢？
        首先每次循环开始，先获取 left 的左边 [0, left - 1] 最高柱子高度 和 right 右边 [right + 1, len - 1] 最高柱子高度（都不包括 left 和 right 本身）
        当 left_max < right_max 时，那么就说明对于 left 右边一定有比 left_max 更高的柱子，那么只需要判断 left 左边 最高柱子 left_max 是否比 left 柱子高就行了，如果是，那么就能装水
        当 left_max >= right_max 时，那么就说明对于 right 左边一定有比 right_max 更高或者相同高度的柱子，那么只需要判断 right 右边最高柱子 right_max 是否比 right 柱子高就行了
        其实就是保证哪边稳定有高柱子就查看哪边
        
        为什么可以隔这么远进行判断？
        比如 对于 left 柱子，如果 left_max 比 left 高，那么如果 right_max 比 left_max 高，那么就跟上面说的 left 右边一定存在比 left 高的柱子，那么 left 柱能装水，
        就算 right_max 对于 left 右边来说不是最高的柱子也无所谓，因为如果不是最高的柱子，那么同样存在另一个比 left 高的柱子，那么 left 同样也能装水，且装水量同样是 left_max - left

        当 left_max < right_max 时，那么当前柱 left 装水量就是直接 left_max - height[left];
        当 left_max >= right_max 时，那么当前柱 right 装水量就是直接， right_max - height[right]
        */
```

```python
#O(n^2)
class Solution:
    def trap(self, nums: List[int]) -> int:

        ans = 0
        for i in range(len(nums)):
            max_left,max_right = 0,0
            #寻找左边的最大值
            for j in range(0,i):
                max_left = max(max_left,nums[j])
            #寻找右边的最大值
            for j in range(i + 1,len(nums)):
                max_right = max(max_right,nums[j])
            if min(max_left,max_right) > nums[i]:
                ans += min(max_left,max_right) - nums[i]
        return ans 
```

```python
#two-pointer O(n)
class Solution:
    def trap(self, height: List[int]) -> int:
        # 边界条件
        if not height: return 0
        n = len(height)

        left,right = 0, n - 1  # 分别位于输入数组的两端
        maxleft,maxright = height[0],height[n - 1]
        ans = 0

        while left <= right:
            maxleft = max(height[left],maxleft)
            maxright = max(height[right],maxright)
            if maxleft < maxright:
                ans += maxleft - height[left]
                left += 1
            else:
                ans += maxright - height[right]
                right -= 1

        return ans
```

#### [845. 数组中的最长山脉](https://leetcode-cn.com/problems/longest-mountain-in-array/)

我们把数组 A 中符合下列属性的任意连续子数组 B 称为 “山脉”：

B.length >= 3
存在 0 < i < B.length - 1 使得 B[0] < B[1] < ... B[i-1] < B[i] > B[i+1] > ... > B[B.length - 1]
（注意：B 可以是 A 的任意子数组，包括整个数组 A。）

给出一个整数数组 A，返回最长 “山脉” 的长度。

如果不含有 “山脉” 则返回 0。

思路：枚举山顶 

维护两个数组，一个严格单调递增，一个严格单调递减

```
示例 1：

输入：[2,1,4,7,3,2,5]
输出：5
解释：最长的 “山脉” 是 [1,4,7,3,2]，长度为 5。
示例 2：

输入：[2,2,2]
输出：0
解释：不含 “山脉”。

```

```python
class Solution:
    def longestMountain(self, A: List[int]) -> int:

        if not A:return 0
        
        n = len(A)
        left = [0] * n
        for i in range(1,n):
            left[i] = left[i - 1] + 1 if A[i - 1] < A[i] else 0

        right = [0] * n
        for i in range(n - 2,-1,-1):
            right[i] = right[i + 1] + 1 if A[i + 1] < A[i] else 0

        ans = 0
        for i in range(n):
            if left[i] > 0 and right[i] > 0:
                ans = max(ans,left[i] + right[i] + 1)

        return ans 
```

#### [面试题 16.24. 数对和](https://leetcode-cn.com/problems/pairs-with-sum-lcci/)

设计一个算法，找出数组中两数之和为指定值的所有整数对。一个数只能属于一个数对。

```
示例 1:

输入: nums = [5,6,5], target = 11
输出: [[5,6]]
示例 2:

输入: nums = [5,6,5,6], target = 11
输出: [[5,6],[5,6]]


```

```python
class Solution:
    def pairSums(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        left,right = 0,len(nums) - 1
        while left < right:
            if nums[left] + nums[right] == target:
                result.append([nums[left],nums[right]])
                left += 1
                right -= 1
            elif nums[left] + nums[right] > target:
                right -= 1
            else:
                left += 1
        return result 
```

### Sliding Window

滑动窗口法属于双指针法的一种，可用于一些查找满足一定条件的连续区间的问题。

#### [26. 删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

```python
 示例 1:

给定数组 nums = [1,1,2], 

函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。 

你不需要考虑数组中超出新长度后面的元素。
示例 2:

给定 nums = [0,0,1,1,1,2,2,3,3,4],

函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。

你不需要考虑数组中超出新长度后面的元素。
```

思路:滑动窗口，与堆查找第几小的数字中记录pivot在当前的list中所在的顺序的思路类似

模板如下

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:

        if not nums:return 0
        i = 0
        for j in range(1,len(nums)):
            if nums[i] != nums[j]:
                i += 1
                nums[i],nums[j] = nums[j],nums[i]

        return i + 1 
      
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0
        K = 1
        for num in nums:
            if i < K or num != nums[i-K]:
                nums[i] = num
                i += 1
        return i       
```

#### [80. 删除排序数组中的重复项 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array-ii/)

给定一个增序排列数组 nums ，你需要在 原地 删除重复出现的元素，使得每个元素最多出现两次，返回移除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

思路：由于是排序的数组，所以其第i个位置若是与i - 2的元素相等，则i与 i-1也一定相等。

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        
        i = 0
        K = 2
        for num in nums:
            if i < K or num != nums[i-K]:
                nums[i] = num
                i += 1
        return i
```

#### [27. 移除元素](https://leetcode-cn.com/problems/remove-element/)

给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

```python
示例 1:

给定 nums = [3,2,2,3], val = 3,

函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。

你不需要考虑数组中超出新长度后面的元素。

示例 2:

给定 nums = [0,1,2,2,3,0,4,2], val = 2,

函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。

注意这五个元素可为任意顺序。

你不需要考虑数组中超出新长度后面的元素。

```

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:

        i = 0
        for j in range(len(nums)):
            if nums[j] != val:
                nums[i] = nums[j]
                i += 1
        return i  
```

#### [643. 子数组最大平均数 I](https://leetcode-cn.com/problems/maximum-average-subarray-i/)

给定 `n` 个整数，找出平均数最大且长度为 `k` 的连续子数组，并输出该最大平均数。

```python
示例 1:

输入: [1,12,-5,-6,50,3], k = 4
输出: 12.75
解释: 最大平均数 (12-5-6+50)/4 = 51/4 = 12.75

```

思路一：记录一个求和数组的列表，对应位置相减，则为该窗口内的元素的和。

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:

        sum_list = [0]
        for i in nums:
            sum_list.append(sum_list[-1] + i)

        max_sum = max(sum_list[i + k] - sum_list[i] for i in range(len(nums) - k + 1))

        return max_sum / float(k)
```

思路二：sliding windows

```python
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:

        moving_sum = 0.0
        for i in range(k):
            moving_sum += nums[i]

        result = moving_sum
        for i in range(k,len(nums)):
            moving_sum += nums[i] - nums[i - k]
            result = max(result,moving_sum)

        return result / k 
```

#### [674. 最长连续递增序列](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/)

给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。

连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

```python
示例 1：

输入：nums = [1,3,5,4,7]
输出：3
解释：最长连续递增序列是 [1,3,5], 长度为3。
尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。 
示例 2：

输入：nums = [2,2,2,2,2]
输出：1
解释：最长连续递增序列是 [2], 长度为1。

```

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:

        result,count = 0,0
        for i in range(len(nums)):
            if i == 0 or nums[i - 1] < nums[i]:
                count += 1
                result = max(result,count)
            else:
                count = 1
        return result 
```

#### [209. 长度最小的子数组](https://leetcode-cn.com/problems/minimum-size-subarray-sum/)

给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的连续子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。

**示例：**

```
输入：s = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
```

思路：滑动窗口，用于连续问题,保存窗口内的加和

```python
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:

        result = len(nums) + 1
        left = total = 0

        for right in range(len(nums)):
            total += nums[right]
            while total >= s:
                result = min(result,right - left + 1)
                total -= nums[left]
                left += 1

        return 0 if result == len(nums) + 1 else result
```

#### [713. 乘积小于K的子数组](https://leetcode-cn.com/problems/subarray-product-less-than-k/)

给定一个正整数数组 `nums`。

找出该数组内乘积小于 `k` 的连续的子数组的个数。

```
示例 1:

输入: nums = [10,5,2,6], k = 100
输出: 8
解释: 8个乘积小于100的子数组分别为: [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]。
需要注意的是 [10,5,2] 并不是乘积小于100的子数组。

```

思路：双指针

```python
'''
10, 5, 2, 6
10            Y
10, 5         Y
10, 5, 2      X
    5, 2      Y
    5, 2, 6   Y
'''

重点：(j - i + 1) 像加和那种情况下会丢掉候选项

```

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:return 0

        product = 1
        i = 0
        answer = 0
        for j,num in enumerate(nums):
            product *= num
            while product >= k:
                product /= nums[i]
                i += 1
            answer += (j - i + 1)

        return answer 
```

#### [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。

**示例:**

```python
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

```

思路一：找出每一个子窗口，分别求其最大值

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        window_max = []
        for i in range(len(nums)):
            if i + k <= len(nums):
                window_max.append(max(nums[i:i+k]))
        return  window_max
```

时间复杂度O(n*k)

#### [239. 滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

难度：困难

思路二：滑动窗口，优化到O(n)

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = collections.deque() # 本质就是单调队列
        ans = []
        for i in range(len(nums)):
            while q and nums[q[-1]] <= nums[i]: q.pop() # 维持单调性
            while q and i - q[0] >= k: q.popleft() # 移除失效元素
            q.append(i)
            if i >= k - 1: ans.append(nums[q[0]])
        return ans
```







