# Sorting

[TOC]

#### 冒泡排序(稳定)

```python
#Average case O(N^2)
def bubble_sort(nums):
    for i in range(len(nums)):
        for j in range(len(nums) - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j],nums[j + 1] = nums[j + 1],nums[j]
    return nums
```

#### 选择排序(不稳定)

思路：每次选择一个最大的，记录其index，然后每次将当前的最大的index与数组的最后的一个位置进行交换

没有best case。

```python
#max_index select sort O(n^2)
def select_sort(nums):
    for i in range(len(nums) - 1,-1,-1):
        max_index = i
        for j in range(i):
            if nums[j] > nums[max_index]:max_index = j
        nums[max_index],nums[i] = nums[i],nums[max_index]
    return nums 
```

#### 插入排序(稳定)

思路：扑克牌，每次通过，当前项目被插入列表的排序部分

比较接近排好序的时候，时间复杂度为O(n)

可以考虑添加二分查找，查找已经排好序的部分

```python
def insert_sort(nums):
    for i in range(len(nums)):
        for j in range(i):
            if nums[i] < nums[j]:
                nums[i],nums[j] = nums[j],nums[i]
    return nums 
```

```python
def insert_sort(nums):
    for i in range(1,len(nums)):
        unsort = i
        while unsort > 0 and nums[unsort - 1] > nums[unsort]:
            nums[unsort - 1],nums[unsort] = nums[unsort],nums[unsort - 1]
            unsort -= 1
    return nums 
```

#### 希尔排序(不稳定)

时间复杂度：open question 大概比O(n^2)快。

思路：插入排序的简单扩展，将原数组分组，允许相隔很远的元素交换来获得速度。（非重点）

```python
def shell_sort(nums):
    gap = len(nums)
    length = len(nums)
    while (gap > 0):
        for i in range(gap, length):
            for j in range(i, gap - 1, -gap):
                if (nums[j - gap] > nums[j]):
                    nums[j], nums[j - gap] = nums[j - gap], nums[j]
        if (gap == 2): 
            gap = 1
        else:
            gap = gap // 2
    return nums
```

#### 计数排序 **(不稳定)

适用范围：

1. 数字比较集中的 
2. 有很多的重复值

抽屉原理，时间复杂度O(n)，最快的排序方式，空间复杂度大

```python
#O(n)
def count_sort(nums):
    
    mmax,mmin = nums[0],nums[0]
    for i in range(1,len(nums)):
        if nums[i] > mmax : mmax = nums[i]
        elif nums[i] < mmin : mmin = nums[i]
    count_nums = mmax - mmin + 1
    counts = [0] * count_nums
    
    for i in range(len(nums)):
        counts[nums[i] - mmin] += 1
    
    pos = 0
    for i in range(count_nums):
        for j in range(counts[i]):
            nums[pos] = i + mmin
            pos += 1
    return nums
```

#### 归并排序**(稳定)

python中  // 为整数除法，/ 为浮点数除法

```python
def merge_sort(nums):
    
    def _merge_sort_helper(a,b):
        c = []
        while len(a) > 0 and len(b) > 0:
            if a[0] > b[0]:
                c.append(b[0])
                b.remove(b[0])
            else:
                c.append(a[0])
                a.remove(a[0])
        if len(a) == len(b) ==0:
            return c
        elif len(a) > 0:
            c += a
        else:
            c +=b
        return c    
    
    if len(nums) <= 1 : return nums            
    middle = len(nums) // 2
    a = merge_sort(nums[:middle])
    b = merge_sort(nums[middle:])
    return _merge_sort_helper(a,b)
```

#### 快速排序**(不稳定)

pivot 选择，一般为首，尾，中间 三个数 取中位数

三点中值算法，可以保证选择的pivot使得时间复杂度为O(nlogn)

```python
def quick_sort(nums):
    if len(nums) <= 1 : return nums
    pivot = nums[0]
    left = quick_sort([i for i in nums[1:] if i <= pivot])
    right = quick_sort([i for i in nums[1:] if i > pivot])
    return left + [pivot] + right
    
```

#### 堆排序(不稳定)

```python
import heapq

def heapSort(nums):
    heapq.heapify(nums)
    h = [heapq.heappop(nums) for _ in range(len(nums))]
    return h
```



