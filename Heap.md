# Heap

[TOC]

优先级别相关的，最大值，最小值的问题。

二叉树：二叉树的左右侧连接为空或者节点

完全树：除了最底层，完全对称平衡

对于任何一个node，其

Parent = (i - 1) // 2

Left_child = 2i + 1

Right_child = 2i + 2

#### [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

```
示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
示例 2:

输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
输出: 4


```

思路一：快排

```python
#时间复杂度 渐进O(n)
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
```

思路二：堆排

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k,nums)[-1]
```

思路三：排序

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
      nums.sort()
      return nums[-k]
```

#### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

给定一个非空的整数数组，返回其中出现频率前 k 高的元素。 

```
示例 1:

输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
示例 2:

输入: nums = [1], k = 1
输出: [1]

```

https://leetcode-cn.com/problems/top-k-frequent-elements/solution/leetcode347onfu-za-du-bu-fen-si-xiang-ji-dui-jie-f/

思路一：堆排

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        from collections import Counter
        import heapq
        count = Counter(nums)   
        return heapq.nlargest(k, count.keys(), key=count.get)
      
      
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    
        from collections import Counter
        counts = Counter(nums)
        heap = [(-count,item) for item,count in counts.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]      
```

思路二：直接排序

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        from collections import Counter
        lookup = Counter(nums)
        # return [num for num, freq in lookup.most_common(k)]
        return [item[0] for item in lookup.most_common(k)]
```

思路三：快排

ToDo

#### [692. 前K个高频单词](https://leetcode-cn.com/problems/top-k-frequent-words/)

给一非空的单词列表，返回前 k 个出现次数最多的单词。

返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率，按字母顺序排序。

```
示例 1：

输入: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
输出: ["i", "love"]
解析: "i" 和 "love" 为出现次数最多的两个单词，均为2次。
    注意，按字母顺序 "i" 在 "love" 之前。


示例 2：

输入: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
输出: ["the", "is", "sunny", "day"]
解析: "the", "is", "sunny" 和 "day" 是出现次数最多的四个单词，
    出现次数依次为 4, 3, 2 和 1 次。


```

思路一：排序

```python
class Solution(object):
    def topKFrequent(self, words, k):
        count = collections.Counter(words)
        candidates = list(count.keys())
        candidates.sort(key = lambda w: (-count[w], w))
        return candidates[:k]
```

解释：

解释下官方方法一中的`candidates.sort(key = lambda w: (-count[w], w))`, sorted不支持多余1个元素的输入，但是sort可以。这个意思就是优先比较val，其次比较key，排序是按照从小大到的，但是输入的是-count[w]，所以返回的索引应该是最大的。

思路二：堆排

计算每个单词的频率，然后将其添加到存储到大小为 k 的小根堆中。它将频率最小的候选项放在堆的顶部。最后，我们从堆中弹出最多 k 次，并反转结果，就可以得到前 k 个高频单词。
在 Python 中，我们使用 heapq\heapify，它可以在线性时间内将列表转换为堆，从而简化了我们的工作。

```python
class Solution(object):
    def topKFrequent(self, words, k):
        count = collections.Counter(words)
        heap = [(-count,word) for word,count in count.items()]
        heapq.heapify(heap)
        return [heapq.heappop(heap)[1] for _ in range(k)]
```

#### [263. 丑数](https://leetcode-cn.com/problems/ugly-number/)

编写一个程序判断给定的数是否为丑数。

丑数就是只包含质因数 2, 3, 5 的正整数。

```
示例 1:

输入: 6
输出: true
解释: 6 = 2 × 3
示例 2:

输入: 8
输出: true
解释: 8 = 2 × 2 × 2
示例 3:

输入: 14
输出: false 
解释: 14 不是丑数，因为它包含了另外一个质因数 7。

```

```python
class Solution:
    def isUgly(self, num: int) -> bool:
        if num <= 0:return False
        ugly_list = [2,3,5]
        for i in ugly_list:
            while num % i == 0:
                num //= i
        return num == 1
```

#### [剑指 Offer 49. 丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

**示例:**

```python
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        q2, q3, q5 = [2], [3], [5]
        ugly = 1
        for u in heapq.merge(q2, q3, q5):
            if n == 1:
                return ugly
            if u > ugly:
                ugly = u
                n -= 1
                q2 += 2 * u,
                q3 += 3 * u,
                q5 += 5 * u,
                                
```

思路二：动态规划

```python
class Solution:
    def nthUglyNumber(self, n: int) -> int:

        dp,a,b,c = [1] * n,0,0,0
        for i in range(1,n):
            n2,n3,n5 = dp[a] * 2,dp[b] * 3,dp[c] * 5
            dp[i] = min(n2,n3,n5)
            if dp[i] == n2:a += 1
            if dp[i] == n3:b += 1
            if dp[i] == n5:c += 1
        return dp[-1]    
        
```

#### [373. 查找和最小的K对数字](https://leetcode-cn.com/problems/find-k-pairs-with-smallest-sums/)

问题

给定两个以升序排列的整形数组 nums1 和 nums2, 以及一个整数 k。

定义一对值 (u,v)，其中第一个元素来自 nums1，第二个元素来自 nums2。

找到和最小的 k 对数字 (u1,v1), (u2,v2) ... (uk,vk)。

```
示例 1:

输入: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
输出: [1,2],[1,4],[1,6]
解释: 返回序列中的前 3 对数：
     [1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]

示例 2:

输入: nums1 = [1,1,2], nums2 = [1,2,3], k = 2
输出: [1,1],[1,1]
解释: 返回序列中的前 2 对数：
     [1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]
示例 3:

输入: nums1 = [1,2], nums2 = [3], k = 3 
输出: [1,3],[2,3]
解释: 也可能序列中所有的数对都被返回:[1,3],[2,3]
```

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:

        queue = []
        def push(i, j):
            if i < len(nums1) and j < len(nums2):
                heapq.heappush(queue, [nums1[i] + nums2[j], i, j])
        push(0, 0)
        pairs = []
        while queue and len(pairs) < k:
            _, i, j = heapq.heappop(queue)
            pairs.append([nums1[i], nums2[j]])
            push(i, j + 1)
            if j == 0:
                push(i + 1, 0)
        return pairs
```

```python
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        heap = []
        for num1 in nums1:
            for num2 in nums2:
                if len(heap) < k:
                    heapq.heappush(heap, (-(num1 + num2) , [num1, num2]))
                else:
                    if num1 + num2 < -heap[0][0]:
                        #heapq.heappushpop(heap, (-(num1 + num2), [num1, num2])) #ok
                        heapq.heappop(heap) #分解动作
                        heapq.heappush(heap, (-(num1 + num2), [num1, num2]))
        return [item[1] for item in heap]
```

#### [23. 合并K个升序链表](https://leetcode-cn.com/problems/merge-k-sorted-lists/)

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

```
示例 1：

输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6

```

思路一：暴力法

直接将所有元素取出来，然后排序，再重组

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists or len(lists) == 0:return None
        all_vals = []
        for l in lists:
            while l:
                all_vals.append(l.val)
                l = l.next
        all_vals.sort()
        dummy = ListNode(None)
        cur = dummy
        for i in all_vals:
            temp_node = ListNode(i)
            cur.next = temp_node
            cur = cur.next
        return dummy.next
```

思路二：优先队列

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists or len(lists) == 0:
            return None
        import heapq
        heap = []
        # 首先 for 嵌套 while 就是将所有元素都取出放入堆中
        for node in lists:
            while node:
                heapq.heappush(heap, node.val)
                node = node.next
        dummy = ListNode(None)
        cur = dummy
        # 依次将堆中的元素取出(因为是小顶堆，所以每次出来的都是目前堆中值最小的元素），然后重新构建一个列表返回
        while heap:
            temp_node = ListNode(heappop(heap))
            cur.next = temp_node
            cur = temp_node
        return dummy.next

```

思路三：分治法

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        length = len(lists)

        if length == 0:
            return None
        if length == 1:
            return lists[0]

        mid = length // 2
        return self.__merge(self.mergeKLists(lists[:mid]),self.mergeKLists(lists[mid:length]))    

    def __merge(self,node_a,node_b):

        dummy = ListNode(None)
        cursor_a,cursor_b,cursor_res = node_a,node_b,dummy
        while cursor_a and cursor_b:
            if cursor_a.val <= cursor_b.val:
                cursor_res.next = ListNode(cursor_a.val)
                cursor_a = cursor_a.next
            else:
                cursor_res.next = ListNode(cursor_b.val)
                cursor_b = cursor_b.next
            cursor_res = cursor_res.next

        if cursor_a:
            cursor_res.next = cursor_a
        if cursor_b:
            cursor_res.next = cursor_b
        return dummy.next  
```

#### [295. 数据流的中位数](https://leetcode-cn.com/problems/find-median-from-data-stream/)

难度：困难

中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。

```
例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
```

示例：

```
addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2
```

https://leetcode-cn.com/problems/find-median-from-data-stream/solution/shu-ju-liu-de-zhong-wei-shu-by-leetcode/

思路一：简单排序

```
将数字存储在可调整大小的容器中。每次需要输出中间值时，对容器进行排序并输出中间值。
```

思路二：插入排序

```
哪种算法允许将一个数字添加到已排序的数字列表中，但仍保持整个列表的排序状态？插入排序！

我们假设当前列表已经排序。当一个新的数字出现时，我们必须将它添加到列表中，同时保持列表的排序性质。这可以通过使用二分搜索找到插入传入号码的正确位置来轻松实现。
（记住，列表总是排序的）。一旦找到位置，我们需要将所有较高的元素移动一个空间，以便为传入的数字腾出空间。

当插入查询的数量较少或者中间查找查询的数量大致相同。 此方法会很好地工作。
```

思路三：两个堆

最大堆用来保存较小的元素，最小堆用来保存较大的元素。

难点：平衡这两个堆

```python
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.minheap = []
        self.maxheap = []

    def addNum(self, num: int) -> None:
        heappush(self.minheap,-heappushpop(self.maxheap,-num))
        if len(self.minheap) > len(self.maxheap):
            heappush(self.maxheap,-heappop(self.minheap))

    def findMedian(self) -> float:
        if len(self.maxheap) > len(self.minheap):
            return -self.maxheap[0]
        return (self.minheap[0] - self.maxheap[0]) / 2    
```

思路四：AVL树加双指针

ToDo

#### [502. IPO](https://leetcode-cn.com/problems/ipo/)

假设 力扣（LeetCode）即将开始其 IPO。为了以更高的价格将股票卖给风险投资公司，力扣 希望在 IPO 之前开展一些项目以增加其资本。 由于资源有限，它只能在 IPO 之前完成最多 k 个不同的项目。帮助 力扣 设计完成最多 k 个不同项目后得到最大总资本的方式。

给定若干个项目。对于每个项目 i，它都有一个纯利润 Pi，并且需要最小的资本 Ci 来启动相应的项目。最初，你有 W 资本。当你完成一个项目时，你将获得纯利润，且利润将被添加到你的总资本中。

总而言之，从给定项目中选择最多 k 个不同项目的列表，以最大化最终资本，并输出最终可获得的最多资本。

```
示例 1:

输入: k=2, W=0, Profits=[1,2,3], Capital=[0,1,1].

输出: 4

解释:
由于你的初始资本为 0，你尽可以从 0 号项目开始。
在完成后，你将获得 1 的利润，你的总资本将变为 1。
此时你可以选择开始 1 号或 2 号项目。
由于你最多可以选择两个项目，所以你需要完成 2 号项目以获得最大的资本。
因此，输出最后最大化的资本，为 0 + 1 + 3 = 4。



```


思路：贪心+堆

https://leetcode-cn.com/problems/ipo/solution/ipo-by-leetcode-3/

```python
class Solution:
    def findMaximizedCapital(self, k: int, W: int, Profits: List[int], Capital: List[int]) -> int:
        # to speed up: if all projects are available
        if W >= max(Capital):
            return W + sum(nlargest(k, Profits))
        
        n = len(Profits)
        projects = [(Capital[i], Profits[i]) for i in range(n)]
        # sort the projects :
        # the most available (= the smallest capital) is the last one
        projects.sort(key = lambda x : -x[0])
        
        available = []
        while k > 0:
            # update available projects
            while projects and projects[-1][0] <= W:
                heappush(available, -projects.pop()[1])
            # if there are available projects,
            # pick the most profitable one
            if available:
                W -= heappop(available)
            # not enough capital to start any project
            else:
                break
            k -= 1
        return W  
```

#### [1046. 最后一块石头的重量](https://leetcode-cn.com/problems/last-stone-weight/)

有一堆石头，每块石头的重量都是正整数。

每一回合，从中选出两块 最重的 石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：

如果 x == y，那么两块石头都会被完全粉碎；
如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0。

```
输入：[2,7,4,1,8,1]
输出：1
解释：
先选出 7 和 8，得到 1，所以数组转换为 [2,4,1,1,1]，
再选出 2 和 4，得到 2，所以数组转换为 [2,1,1,1]，
接着是 2 和 1，得到 1，所以数组转换为 [1,1,1]，
最后选出 1 和 1，得到 0，最终数组转换为 [1]，这就是最后剩下那块石头的重量。

```

思路一：首先排序，然后pop出stones的后两个，直到结束。排序的时间复杂度为O(nlogn) + 循环二分查找O(nlogn)

```python
import bisect
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:

        stones.sort()

        while len(stones) > 1:
            x = stones.pop()
            y = stones.pop()
            if x != y:
                tmp = x - y
                idx = bisect.bisect_left(stones,tmp)
                stones.insert(idx,tmp)

        if len(stones) == 1:return stones[0]
        return 0
```

思路二：heap Overall的时间复杂度O(nlogn)

```python
import heapq
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:

        max_stones = [i * -1 for i in stones]
        heapq.heapify(max_stones)

        while len(max_stones) > 1:
            x = heapq.heappop(max_stones) * -1
            y = heapq.heappop(max_stones) * -1

            if x != y:
                heapq.heappush(max_stones,abs(x - y) * -1)

        if len(max_stones) == 1:return max_stones[0] * -1
        return 0
```

