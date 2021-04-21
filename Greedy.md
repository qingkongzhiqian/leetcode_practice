# Greedy

[TOC]

#### 找硬币

设有n	种不同面值的硬币，各硬币的面值存于数组T〔1:n	〕中。现要 用这些面值的硬币来找钱。可以使用的各种面值的硬币面值{1，2，5 ，10，20，50，100，500，1000} 对任意钱数0≤m≤20001，设计一个用 最少硬币找钱 的方法

思路：对存在的硬币进行排序。

```python
def minCoins(v):
    available = [1,2,5,10,20,50,100,500,1000]
    result = []
    for i in available[::-1]:
        while v >= i:
            v -= i
            result.append(i)
    return result

coins = 93
print ("min coins",minCoins(coins))


1 def minCoins(nums, tar):  #除法更快一些
2     nums.sort()
3     result = []
4     for i in nums[::-1]:
5         if tar / i != 0:
6             tar = tar % i
7             result.append((i, tar//i))
8     return result
```

#### 活动问题

N个活动，每个活动的开始时间为si，结束时间是fi。如果 si ≥	fj	or	sj ≥	fi 则可以定义这两场活动不冲突。试着找到一种能够找到最多的非冲突 活动的集合（S）。也就是说无法找到集合S’ 使得|S’| > |S|。

思路：对活动的结束时间进行排序，然后比较下一个的开始时间是不是在当前的结束时间的后面。

```python
def printMaxActivities(acts):
    n = len(acts)
    sort_acts = sorted(acts,key = lambda x :x [1])
    prev = sort_acts[0]
    print ("prev",prev)
    for curr in sort_acts:
        if curr[0] >= prev[1]:
            print ("curr",curr)
            prev = curr


acts = [(0,6),(3,4),(1,2),(5,7),(8,9),(5,9)]
printMaxActivities(acts)
```

#### 最小的数字问题

如何找到给定数字总和s和位数m的最小数字

```
输入: s = 9，m = 2 

输出: 18

还有许多其他可能的数字，如45，54，90等，数字总和为9，数字位数为2. 其中最小的为18。 

输入: s = 20，m = 3 • 输出: 299
```

```python
def findSmalllest(m,s):
    if s == 0:
        if m == 1:
            print ("Smallest number is 0")
        else:
            print ("Not possible")
        return     

    if s > 9 * m:
        print ("Not possible")
        return
    
    res = [0 for i in range(m + 1)]
    s -= 1
    for i in range(m - 1,0,-1):
        if s > 9:
            res[i] = 9
            s -= 9
        else:
            res[i] = s
            s = 0
    res[0] = s + 1
    print ("Smallest number is  ",end = "")
    for i in range(m):
        print (res[i],end = "")
        
s,m = 20,3        
findSmalllest(m,s)   
```

#### 两个数字的最小和

给定一个数字数组(数值从0到9)，找出由数组数字形成的两个数字的最小可能和。给定数组的所有数 字必须用于形成两个数字。

```python
输入：[6，8，4，5，2，3] 

输出：604

最小总和由数字组成

358和246

输入：[5，3，0，7，4]

输出：82

最小总和由数字组成

35和047
```

思路：维护两个变量，一次从目标数组中取最小的两个元素，组成数字

使用的数据结构，堆

```python
堆的时间复杂度分析：
import heapq
复制copy()：O(n)
建堆heapq.heapify(heap)：O(n)
插入元素heapq.heappush()：O(nlogn)
返回根元素heap[0]:O(1)
返回并删除根元素heapq.heappop(heap):O(1)
```

```python
def min_sum(a):
    heapq.heapify(a)
    num1 = 0
    num2 = 0
    while a:
        num1 = num1 * 10 + heapq.heappop(a)
        if a:
            num2 = num2 * 10 + heapq.heappop(a)
    return num1 + num2

a = [6, 8, 4, 5, 2, 3]
print (min_sum(a))
```

#### [1167. 连接棒材的最低费用](https://leetcode-cn.com/problems/minimum-cost-to-connect-sticks/)

为了装修新房，你需要加工一些长度为正整数的棒材 。棒材以数组 sticks 的形式给出，其中 sticks[i] 是第 i 根棒材的长度。

如果要将长度分别为 x 和 y 的两根棒材连接在一起，你需要支付 x + y 的费用。 由于施工需要，你必须将所有棒材连接成一根。

返回你把所有棒材 sticks 连成一根所需要的最低费用。注意你可以任意选择棒材连接的顺序。

```python
示例 1：

输入：sticks = [2,4,3]
输出：14
解释：从 sticks = [2,4,3] 开始。
1. 连接 2 和 3 ，费用为 2 + 3 = 5 。现在 sticks = [5,4]
2. 连接 5 和 4 ，费用为 5 + 4 = 9 。现在 sticks = [9]
所有棒材已经连成一根，总费用 5 + 9 = 14
示例 2：

输入：sticks = [1,8,3,5]
输出：30
解释：从 sticks = [1,8,3,5] 开始。
1. 连接 1 和 3 ，费用为 1 + 3 = 4 。现在 sticks = [4,8,5]
2. 连接 4 和 5 ，费用为 4 + 5 = 9 。现在 sticks = [9,8]
3. 连接 9 和 8 ，费用为 9 + 8 = 17 。现在 sticks = [17]
所有棒材已经连成一根，总费用 4 + 9 + 17 = 30

```

```python
import heapq
class Solution:
    def connectSticks(self, sticks: List[int]) -> int:

        if not sticks:return sticks
        if len(sticks) == 1:return 0

        heapq.heapify(sticks)
        total = 0

        while sticks:
            first = heapq.heappop(sticks)
            second = heapq.heappop(sticks)
            local = first + second
            total += local
            if not sticks:break
            heapq.heappush(sticks,local)

        return total 
```

#### 最小平台数

根据所有到达火车站的列车的到达和离开时间，找到火车站所需的最少数量的平台，以免 列车等待。 

我们给出了代表停止列车到达和离开时间的两个数组 

```
例子： 

输入：arr [] = {9:00, 9:40, 9:50, 11:00, 15:00, 18:00} 

dep [] = {9:10, 12:00, 11:20, 11:30, 19:00, 20:00} 

输出：3 

一次最多有三班列车（时间为11:00至11:20）
```

```python
def findPlatform(arr, dep, n):
 
    arr.sort()
    dep.sort()
  
    # plat_needed indicates number of platforms needed at a time
    plat_needed = 0
    result = 0
    i = 0
    j = 0
  
    # Similar to merge in merge sort to process all events in sorted order
    while (i < n and j < n):
        if (arr[i] < dep[j]):
            plat_needed += 1
            i += 1
  
            result = max(result, plat_needed)
  
        else:
            plat_needed -= 1
            j += 1
         
    return result
    
arr = [900, 940, 950, 1100, 1500, 1800]
dep = [910, 1200, 1120, 1130, 1900, 2000]
n = len(arr)
findPlatform(arr, dep, n)    
```

#### 部分背包问题 

给定n个项目的权重和值，我们需要把这些项目放入W的背包中, 以获得背包中最大的总价值。 

在0-1背包问题中，我们不允许分解物品。我们要么拿整个项目 ，要么不拿。 

在分数背包中，我们可以打破物品以最大化背包的总价值。这个 问题，我们可以打破项目也被称为分数背包问题

思路：性价比

```python
def fracKnapsack(capacity, weights, values):
    numItems = len(values)
    valuePerWeight = sorted([[v / w, w, v] for v,w in zip(values,weights)], reverse=True)
    print(valuePerWeight)
    totalCost = 0.
    for tup in valuePerWeight:
        if capacity >= tup[1]:
            capacity -= tup[1]
            totalCost += tup[2]
        else:
            totalCost += capacity * tup[0]
            break
    return totalCost
    
    
n = 3
capacity = 50
values = [72, 100, 120]
weights = [24, 50, 30]
fracKnapsack(capacity, weights, values)    
```

#### [455. 分发饼干](https://leetcode-cn.com/problems/assign-cookies/)

假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。

对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

```python
示例 1:

输入: g = [1,2,3], s = [1,1]
输出: 1
解释: 
你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
所以你应该输出1。
示例 2:

输入: g = [1,2], s = [1,2,3]
输出: 2
解释: 
你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
你拥有的饼干数量和尺寸都足以让所有孩子满足。
所以你应该输出2.


```

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        n, m = len(g), len(s)
        i = j = count = 0

        while i < n and j < m:
            while j < m and g[i] > s[j]:
                j += 1
            if j < m:
                count += 1
            i += 1
            j += 1
        
        return count
```

#### [1505. 最多 K 次交换相邻数位后得到的最小整数](https://leetcode-cn.com/problems/minimum-possible-integer-after-at-most-k-adjacent-swaps-on-digits/)

给你一个字符串 num 和一个整数 k 。其中，num 表示一个很大的整数，字符串中的每个字符依次对应整数上的各个 数位 。

你可以交换这个整数相邻数位的数字 最多 k 次。

请你返回你能得到的最小整数，并以字符串形式返回。

```python
示例 1：
输入：num = "4321", k = 4
输出："1342"
解释：4321 通过 4 次交换相邻数位得到最小整数的步骤如上图所示。
示例 2：

输入：num = "100", k = 1
输出："010"
解释：输出可以包含前导 0 ，但输入保证不会有前导 0 。
示例 3：

输入：num = "36789", k = 1000
输出："36789"
解释：不需要做任何交换。
示例 4：

输入：num = "22", k = 22
输出："22"
示例 5：

输入：num = "9438957234785635408", k = 23
输出："0345989723478563548"

```

思路：首先在前k+1的位置找到最小的值，然后将其移动到最前面，如果移动之后还有可以移动的次数，则在第二个到k+1的区间找最小值进行移动。时间复杂度为O(n^2),运行超时。

```python
class Solution:
    def minInteger(self, num: str, k: int) -> str:

        num_list = list(num)
        for i in range(len(num_list) - 1):

            pos = i
            for j in range(i + 1,len(num_list)):
                if j - i > k:break
                if num_list[j] < num_list[pos]:
                    pos = j

            for a in range(pos,i,-1):
                num_list[a],num_list[a - 1] = num_list[a - 1],num_list[a]

            k -= pos - i    

        return ''.join(num_list) 
```

思路二：树状数组,时间复杂度为O(nlogn)

```python
class BIT:
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)
    
    @staticmethod
    def lowbit(x: int) -> int:
        return x & (-x)
    
    def update(self, x: int):
        while x <= self.n:
            self.tree[x] += 1
            x += BIT.lowbit(x)
    
    def query(self, x: int) -> int:
        ans = 0
        while x > 0:
            ans += self.tree[x]
            x -= BIT.lowbit(x)
        return ans

    def queryRange(self, x: int, y: int) -> int:
        return self.query(y) - self.query(x - 1)


class Solution:
    def minInteger(self, num: str, k: int) -> str:
        n = len(num)
        pos = [list() for _ in range(10)]
        for i in range(n - 1, -1, -1):
            pos[ord(num[i]) - ord('0')].append(i + 1)
        
        ans = ""
        bit = BIT(n)
        for i in range(1, n + 1):
            for j in range(10):
                if pos[j]:
                    behind = bit.queryRange(pos[j][-1] + 1, n)
                    dist = pos[j][-1] + behind - i
                    if dist <= k:
                        bit.update(pos[j][-1])
                        pos[j].pop()
                        ans += str(j)
                        k -= dist
                        break
        
        return ans

```

#### [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个位置。

```
示例 1:

输入: [2,3,1,1,4]
输出: true
解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
示例 2:

输入: [3,2,1,0,4]
输出: false
解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。


```

思路：直接贪心法解决

就是用一个变量记录当前能够到达的最大的索引，并逐个遍历数组中的元素去更新这个索引，遍历完成判断这个索引是否大于`数组长度 - 1`即可。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:

        mmax = 0
        for i in range(len(nums) - 1):
            if mmax < i:return False
            mmax = max(_max,nums[i] + i)

        return mmax >= len(nums) - 1        

```

