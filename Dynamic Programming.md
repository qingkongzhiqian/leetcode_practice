# Dynamic Programming

[TOC]

分治法：将一个大问题拆分成小问题，每一个小问题独立的。(top-down)

动态规划：将一个大问题拆分成小问题，小问题间有很多重复的，可以将小问题存储起来。(bottom-up)

### 1维动态规划

#### [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

注意：给定 n 是一个正整数。

```python
示例 1：

输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。

1.  1 阶 + 1 阶
2.  2 阶
    示例 2：

输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。

1.  1 阶 + 1 阶 + 1 阶
2.  1 阶 + 2 阶
3.  2 阶 + 1 阶

```

思路一：DP  O(n)

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = [0,1,2]
        if n <= 2:return dp[n]
        for i in range(3,n + 1):
            dp.append(dp[i - 1] + dp[i - 2])
        return dp[n] 
```

思路二：矩阵优化. O(logn)

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

#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

你可以认为每种硬币的数量是无限的。

```python
示例 1：

输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
示例 2：

输入：coins = [2], amount = 3
输出：-1
示例 3：

输入：coins = [1], amount = 0
输出：0
示例 4：

输入：coins = [1], amount = 1
输出：1
示例 5：

输入：coins = [1], amount = 2
输出：2
```

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        
        for coin in coins:
            for x in range(coin,amount + 1):
                dp[x] = min(dp[x],dp[x-coin] + 1)
  
        return dp[amount] if dp[amount] != float('inf') else -1 
```

#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

```
示例 1：

输入：[1,2,3,1]
输出：4
解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
示例 2：

输入：[2,7,9,3,1]
输出：12
解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。

```

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0] * 2 for i in range(n + 1)]
        for i in range(1,n + 1):
            dp[i][0] = max(dp[i - 1][0],dp[i - 1][1])
            dp[i][1] = dp[i - 1][0] + nums[i - 1]

        return max(dp[n][0],dp[n][1])  
```

优化：

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        yes,no = 0,0
        for i in nums:
            no,yes = max(yes,no),no + i

        return max(yes,no)
```

#### [213. 打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈 ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，能够偷窃到的最高金额。

```python
示例 1：

输入：nums = [2,3,2]
输出：3
解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
示例 2：

输入：nums = [1,2,3,1]
输出：4
解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
     偷窃到的最高金额 = 1 + 3 = 4 。
示例 3：

输入：nums = [0]
输出：0

```

思路：由于是环，与上一题的区别就是如果选择偷了第一个则最后一个不能偷

```python
class Solution:
    def rob(self, nums: [int]) -> int:

        if len(nums) == 0:return 0
        if len(nums) == 1:return nums[0]

        def my_rob(nums):
            cur, pre = 0, 0
            for num in nums:
                cur, pre = max(pre + num, cur), cur
            return cur
        return max(my_rob(nums[:-1]),my_rob(nums[1:]))
      
      
class Solution:
    def rob(self, nums: [int]) -> int:

        if len(nums) == 0:return 0
        if len(nums) == 1:return nums[0]
        if len(nums) == 2:return max(nums[0],nums[1])
        return max(self.__rob(nums[:-1]),self.__rob(nums[1:]))

    def __rob(self,nums):
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0],nums[1])
        for i in range(2,len(nums)):
            dp[i] = max(dp[i - 2] + nums[i],dp[i - 1])
        return dp[-1]        
```

#### [746. 使用最小花费爬楼梯](https://leetcode-cn.com/problems/min-cost-climbing-stairs/)

数组的每个索引作为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。

每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。

您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。

```python
示例 1:

输入: cost = [10, 15, 20]
输出: 15
解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
 示例 2:

输入: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
输出: 6
解释: 最低花费方式是从cost[0]开始，逐个经过那些1，跳过cost[3]，一共花费6。

```

```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:

        n = len(cost) + 1
        dp = [0] * n
        for i in range(2,n):
            dp[i] = min(dp[i - 2] + cost[i - 2],dp[i - 1] + cost[i - 1])
        return dp[n - 1] 
```

#### [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

```
示例 1:

输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
```

```python
class Solution:
    def translateNum(self, num: int) -> int:
        s = str(num)
        a = b = 1
        for i in range(2, len(s) + 1):
            a, b = (a + b if "10" <= s[i - 2:i] <= "25" else a), a
        return a

class Solution:
    def translateNum(self, num: int) -> int:
        s = str(num)
        if len(s) < 2:return 1
        dp = [0] * len(s)
        dp[0] = 1
        dp[1] = 2 if int(s[0] + s[1]) < 26 else 1
        for i in range(2,len(s)):
            dp[i] = dp[i-1] + dp[i-2] if int(s[i-1] + s[i]) < 26 and s[i-1] != '0' else dp[i-1]
        return dp[-1]      
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

#### [91. 解码方法](https://leetcode-cn.com/problems/decode-ways/)

一条包含字母 A-Z 的消息通过以下方式进行了编码：

'A' -> 1
'B' -> 2
...
'Z' -> 26
给定一个只包含数字的非空字符串，请计算解码方法的总数。

题目数据保证答案肯定是一个 32 位的整数。

```python
示例 1：

输入：s = "12"
输出：2
解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
示例 2：

输入：s = "226"
输出：3
解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
示例 3：

输入：s = "0"
输出：0
示例 4：

输入：s = "1"
输出：1
示例 5：

输入：s = "2"
输出：1

```

```python
class Solution:
    def numDecodings(self, s: str) -> int:

        if s == "" or s[0] == "0":return 0
        dp = [1,1]
        for i in range(2,len(s) + 1):
            result = 0
            if 10 <= int(s[i-2:i]) <= 26:
                result = dp[i - 2]
            if s[i - 1] != "0":
                result += dp[i - 1]
            dp.append(result)

        return dp[len(s)]   
```

#### [139. 单词拆分](https://leetcode-cn.com/problems/word-break/)

给定一个非空字符串 s 和一个包含非空单词的列表 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。

说明：

拆分时可以重复使用字典中的单词。
你可以假设字典中没有重复的单词。

```python
示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分成 "leet code"。
示例 2：

输入: s = "applepenapple", wordDict = ["apple", "pen"]
输出: true
解释: 返回 true 因为 "applepenapple" 可以被拆分成 "apple pen apple"。
     注意你可以重复使用字典中的单词。
示例 3：

输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
输出: false

```

思路：我们可以建立的是 dp[i - word.length] 和 dp[i] 的关系。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        n=len(s)
        dp=[False]*(n+1)
        dp[0]=True
        for i in range(n):
            for j in range(i+1,n+1):
                if(dp[i] and (s[i:j] in wordDict)):
                    dp[j]=True
        return dp[-1]
```

### 卡特兰数相关

https://zhuanlan.zhihu.com/p/31317307

#### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？

```
示例:

输入: 3
输出: 5
解释:
给定 n = 3, 一共有 5 种不同结构的二叉搜索树:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3


```

```python
class Solution:
    def numTrees(self, n: int) -> int:

        if n <= 2:return n
        sol = [0] * (n + 1)
        sol[0] = sol[1] = 1

        for i in range(2,n + 1):
            for left in range(0,i):
                sol[i] += sol[left] * sol[i - 1 - left]

        return sol[n]   
```

```python
class Solution:
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        G = [0]*(n+1)
        G[0], G[1] = 1, 1

        for i in range(2, n+1):
            for j in range(1, i+1):
                G[i] += G[j-1] * G[i-j]

        return G[n]

```

思路二：卡特兰数公式

```python
class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        C = 1
        for i in range(0, n):
            C = C * 2*(2*i+1)/(i+2)
        return int(C)
```

https://leetcode-cn.com/circle/article/lWYCzv/

#### [95. 不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)

给定一个整数 *n*，生成所有由 1 ... *n* 为节点所组成的 **二叉搜索树** 。

```python
输入：3
输出：
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
解释：
以上的输出对应以下 5 种不同结构的二叉搜索树：

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        def generateTrees(start, end):
            if start > end:
                return [None,]
            
            allTrees = []
            for i in range(start, end + 1):  # 枚举可行根节点
                # 获得所有可行的左子树集合
                leftTrees = generateTrees(start, i - 1)
                
                # 获得所有可行的右子树集合
                rightTrees = generateTrees(i + 1, end)
                
                # 从左子树集合中选出一棵左子树，从右子树集合中选出一棵右子树，拼接到根节点上
                for l in leftTrees:
                    for r in rightTrees:
                        currTree = TreeNode(i)
                        currTree.left = l
                        currTree.right = r
                        allTrees.append(currTree)
            
            return allTrees
        
        return generateTrees(1, n) if n else []
```

#### [53. 连续子数组的最大和](https://leetcode-cn.com/problems/maximum-subarray/)

给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

```
示例:

输入: [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        
        if len(nums) == 1:return nums[0]
        dp = [-float('inf')] * len(nums)
        dp[0] = nums[0]

        for i in range(1,len(nums)):
            dp[i] = max(dp[i - 1] + nums[i],nums[i])

        return max(dp)  
```

#### [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

给你一个整数数组 `nums` ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

思路：不同于加和的操作，需要mentain两个值，一个最大值，一个最小值。

来抵消正负的操作。

```
示例 1:

输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
示例 2:

输入: [-2,0,-1]
输出: 0
解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
```

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        
        n = len(nums)
        max_dp = [1] * (n + 1)
        min_dp = [1] * (n + 1)
        ans = float('-inf')

        for i in range(1,n + 1):
            max_dp[i] = max(max_dp[i - 1] *  nums[i - 1],
                            min_dp[i - 1] * nums[i - 1],
                            nums[i - 1]
            )

            min_dp[i] = min(max_dp[i - 1] *  nums[i - 1],
                            min_dp[i - 1] * nums[i - 1],
                            nums[i - 1])

            ans = max(ans,max_dp[i])

        return ans 
```

#### [121. 买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。

注意：你不能在买入股票前卖出股票。

```python
示例 1:

输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
示例 2:

输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        profit = 0
        min_price = float('inf')

        for i in prices:
            profit = max(profit,i - min_price)
            min_price = min(i,min_price)

        return profit    
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        if len(prices) < 2:return 0
        min_price = prices[0]
        dp = [0] * len(prices)

        for i in range(len(prices)):
            dp[i] = max(dp[i - 1],prices[i] - min_price)
            min_price = min(min_price,prices[i])

        return dp[-1] 
```

#### [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```python
示例 1:

输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
示例 2:

输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
示例 3:

输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        profit = 0
        for i in range(1,len(prices)):
            if prices[i] > prices[i - 1]:
                profit += prices[i] - prices[i - 1]
        return profit   
```

#### [714. 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/)

给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

```python
示例 1:

输入: prices = [1, 3, 2, 8, 4, 9], fee = 2
输出: 8
解释: 能够达到的最大利润:  
在此处买入 prices[0] = 1
在此处卖出 prices[3] = 8
在此处买入 prices[4] = 4
在此处卖出 prices[5] = 9
总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8.

```

思路：维护两个数组，一个是当手上没有股票的时候的收益一个是手上有股票的时候上的收益。

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        n = len(prices)
        sell, buy = 0, -prices[0]
        for i in range(1, n):
            sell, buy = max(sell, buy + prices[i] - fee), max(buy, sell - prices[i])
        return sell

        # n = len(prices)
        # dp = [[0, -prices[0]]] + [[0, 0] for _ in range(n - 1)]
        
        # for i in range(1,n):
        #     dp[i][0] = max(dp[i - 1][0],dp[i - 1][1] + prices[i] - fee)
        #     dp[i][1] = max(dp[i - 1][1],dp[i - 1][0] - prices[i])

        # return dp[n - 1][0]
```

#### [123. 买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

```python
示例 1:

输入: [3,3,5,0,0,3,1,4]
输出: 6
解释: 在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
示例 2:

输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。   
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。   
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
示例 3:

输入: [7,6,4,3,1] 
输出: 0 
解释: 在这个情况下, 没有交易完成, 所以最大利润为 0。

```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        total_max_profit = 0
        n = len(prices)
        left_profits = [0] * n
        min_price = float('inf')

        for i in range(n):
            min_price = min(min_price, prices[i])
            total_max_profit = max(total_max_profit, prices[i] - min_price)
            left_profits[i] = total_max_profit

        max_profit = 0
        max_price = float('-inf')
        for i in range(n - 1, 0, -1):
            max_price = max(max_price, prices[i])
            max_profit = max(max_profit, max_price - prices[i])
            total_max_profit = max(total_max_profit, max_profit + left_profits[i - 1])
        return total_max_profit
```

#### [188. 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

 

```python
示例 1：

输入：k = 2, prices = [2,4,1]
输出：2
解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。
示例 2：

输入：k = 2, prices = [3,2,6,5,0,3]
输出：7
解释：在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
     随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。

```

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices) < 2:return 0

        if len(prices) <= k / 2:
            maxProfit(prices)
            
        local = [0] * (k+1)
        globl = [0] * (k+1)
        
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i - 1]
            j = k
            while j > 0:
                local[j] = max(globl[j - 1], local[j] + diff)
                globl[j] = max(globl[j], local[j])
                j -= 1
                
        return globl[k]
```

#### [309. 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。

```python
示例:

输入: [1,2,3,0,2]
输出: 3 
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
```

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        if len(prices) < 2:return 0
        n = len(prices)
        sell = [0] * n
        buy  = [0] * n
        sell[0] = 0;
        buy[0] = -prices[0]
        for i in range(1, n):
            sell[i] = max(sell[i - 1], buy[i - 1] + prices[i])
            buy[i] = max(buy[i - 1], (sell[i - 2] if i > 1 else 0) - prices[i])
                
        return sell[-1]
```

股票问题的通用解法：

https://leetcode-cn.com/circle/article/qiAgHn/

#### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

```python
输入：m = 3, n = 7
输出：28

示例 2：

输入：m = 3, n = 2
输出：3
解释：
从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右
示例 3：

输入：m = 7, n = 3
输出：28
示例 4：

输入：m = 3, n = 3
输出：6

```

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:

        dp = [[1 for a in range(n)] for b in range(m)]
        for a in range(1,m):
            for b in range(1,n):
                dp[a][b] = dp[a - 1][b] + dp[a][b - 1]

        return dp[-1][-1]  
```

空间优化：aux保存的就是一行的，若是计算下一行的每一个值，只依赖于前一个和上一个，没更新的aux的当前位置就是上一个的数值，循环的就是前一个的数值

```python
def uniquePaths(m, n):
    aux = [1 for x in range(n)]
    for i in range(1, m):
        for j in range(1, n):
            aux[j] = aux[j]+aux[j-1]
    return aux[-1]
```

#### [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

```python
输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
输出：2
解释：
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右

```

思路：障碍物位置为0

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:

        m,n = len(obstacleGrid),len(obstacleGrid[0])
        dp = [1] + [0] * (n - 1)

        for a in range(m):
            for b in range(n):
                if obstacleGrid[a][b] == 1:
                    dp[b] = 0
                elif b > 0:
                    dp[b] += dp[b - 1]
        return dp[n - 1]
```

#### [剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

难度中等100收藏分享切换为英文接收动态反馈

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

**示例 1:**

```
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

```python
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:

        m,n = len(grid),len(grid[0])
        dp = [[0] * (n) for _ in range(m)]
        dp[0][0] = grid[0][0]

        for i in range(1,m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]

        for j in range(1,n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]    

        for a in range(1,m):
            for b in range(1,n):
                dp[a][b] = max(dp[a - 1][b],dp[a][b - 1]) + grid[a][b]

        return dp[-1][-1]
```

#### 在棋盘上移动

给定一个有n行n列的棋盘。每个格子上有不同的值代表该格子的价值。目标是找到获 利最大的移动路线（从第一行的某个格子到最后一行的某个格子） 

移动方法： 

• 1. 移动到下一行的之前一列（up then left） 

• 2. 移动到下一行（UP） 

• 3.移动到下一行的下一列（UP then right）

思路：向下，左下和右下移动

```python
def movingBoard(board):
    result = board
    m = len(board)
    n = len(board[0])
    for i in range(1, m):
        for j in range (0, n):
            result[i][j] = max(0 if j == 0 else result[i-1][j-1], \
                               result[i-1][j], \
                               0 if j == n-1 else result[i-1][j+1] ) \
                            + board[i][j]
    return max(result[-1])
  
  
def movingBoard2(board):
    result = board[0]
    m = len(board)
    n = len(board[0])
    for i in range(1, m):
        for j in range (0, n):
            result[j] = max(0 if j == 0 else result[j-1], \
                            result[j], \
                            0 if j == n-1 else result[j+1] ) \
                        + board[j]
    return max(result)  
```

#### [1277. 统计全为 1 的正方形子矩阵](https://leetcode-cn.com/problems/count-square-submatrices-with-all-ones/)

给你一个 m * n 的矩阵，矩阵中的元素不是 0 就是 1，请你统计并返回其中完全由 1 组成的 正方形 子矩阵的个数。 

```
示例 1：

输入：matrix =
[
  [0,1,1,1],
  [1,1,1,1],
  [0,1,1,1]
]
输出：15
解释： 
边长为 1 的正方形有 10 个。
边长为 2 的正方形有 4 个。
边长为 3 的正方形有 1 个。
正方形的总数 = 10 + 4 + 1 = 15.
示例 2：

输入：matrix = 
[
  [1,0,1],
  [1,1,0],
  [1,1,0]
]
输出：7
解释：
边长为 1 的正方形有 6 个。 
边长为 2 的正方形有 1 个。
正方形的总数 = 6 + 1 = 7.


```

```python
class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:

        m,n = len(matrix),len(matrix[0])
        dp = [[0] * n for i in range(m)]
        ans = 0
        for a in range(m):
            for b in range(n):
                if a == 0 or b == 0:
                    dp[a][b] = matrix[a][b]
                elif matrix[a][b] == 0:
                    dp[a][b] == 0
                else:
                    dp[a][b] = min(dp[a][b - 1],dp[a - 1][b],dp[a - 1][b - 1]) + 1
                ans += dp[a][b]
        return ans 
```

#### [221. 最大正方形](https://leetcode-cn.com/problems/maximal-square/)

在一个由 `'0'` 和 `'1'` 组成的二维矩阵内，找到只包含 `'1'` 的最大正方形，并返回其面积。

```python
输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
输出：4

```

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:

        if matrix == []:return 0
        m,n = len(matrix),len(matrix[0])
        dp = [[0] * n for x in range(m)]
        ans = 0
        for x in range(m):
            for y in range(n):
                dp[x][y] = int(matrix[x][y])
                if x and y and dp[x][y]:
                    dp[x][y] = min(dp[x - 1][y - 1],dp[x][y - 1],dp[x - 1][y]) + 1
                ans = max(ans,dp[x][y])
        return ans * ans  
```

以每一个小正方形的右下角作为切入点，进行动态规划

https://leetcode-cn.com/problems/maximal-square/solution/zui-da-zheng-fang-xing-by-leetcode-solution/

#### 背包问题

```python
def knapSack(W, wt, val, n):
    K = [[0 for x in range(W+1)] for x in range(n+1)]
 
    # Build table K[][] in bottom up manner
    for i in range(n+1):
        for w in range(W+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
 
    return K[n][W]
```

#### [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/)

给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。

```
示例：

输入：
A: [1,2,3,2,1]
B: [3,2,1,4,7]
输出：3
解释：
长度最长的公共子数组是 [3, 2, 1] 。
```

```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:

        m,n = len(A),len(B)
        ans = 0
        dp = [[0 for _ in range(n + 1)] for j in range(m + 1)]

        for a in range(m + 1):
            for b in range(n + 1):
                if a == 0 or b == 0:
                    dp[a][b] = 0
                elif A[a - 1] == B[b - 1]:
                    dp[a][b] = dp[a - 1][b - 1] + 1
                    ans = max(dp[a][b],ans)
                
        return ans 
```

子数组与子序列的区别就是子数组必须是连续的，而子序列未必连续

体现在转移矩阵上就是：若当前的i位置的值与j位置的值相等的情况下，Tij = Ti-1 j -1 + 1，否则Tij = 0

#### [1143. 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

https://lucifer.ren/blog/2020/07/01/LCS/

给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

若这两个字符串没有公共子序列，则返回 0。

```
示例 1:

输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace"，它的长度为 3。
示例 2:

输入：text1 = "abc", text2 = "abc"
输出：3
解释：最长公共子序列是 "abc"，它的长度为 3。
示例 3:

输入：text1 = "abc", text2 = "def"
输出：0
解释：两个字符串没有公共子序列，返回 0。


```

https://leetcode-cn.com/problems/longest-common-subsequence/solution/ni-de-yi-fu-wo-ba-liao-zui-chang-gong-gong-zi-xu-2/

思路：Tij表示text1序列的第i位置和text2的第j位置时形成的LCS

base情况：Ti0 = T0j = 0

若当前的i位置的值与j位置的值相等的情况下，Tij = Ti-1 j -1 + 1

若当前的i位置的值与j位置的值不相等的情况下,Tij = max(Ti-1 j , Ti j-1)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:

        m,n = len(text1),len(text2)
        dp = [[0 for i in range(n + 1)] for j in range(m + 1)]
        result = 0

        for a in range(m + 1):
            for b in range(n + 1):
                if a == 0 or b == 0:
                    dp[a][b] = 0
                elif text1[a - 1] == text2[b - 1]:
                    dp[a][b] = dp[a - 1][b - 1] + 1
                    result = max(result,dp[a][b])
                else:
                    dp[a][b] = max(dp[a - 1][b],dp[a][b - 1])   

        return result 
```

#### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

给你一个字符串 `s`，找到 `s` 中最长的回文子串。

```python
示例 1：

输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
示例 2：

输入：s = "cbbd"
输出："bb"
示例 3：

输入：s = "a"
输出："a"
示例 4：

输入：s = "ac"
输出："a"

```

思路：对于一个子串而言，如果它是回文串，并且长度大于 2，那么将它首尾的两个字母去除之后，它仍然是个回文串。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
      
      	 def extend(i, j, s):
            while(i >= 0 and j < len(s) and s[i] == s[j]):
                i -= 1
                j += 1
            return s[i + 1:j]
      
        n = len(s)
        if n == 0:return ""
        res = s[0]

        for i in range(n - 1):
            e1 = extend(i, i, s)
            e2 = extend(i, i + 1, s)
            if max(len(e1), len(e2)) > len(res):
                res = e1 if len(e1) > len(e2) else e2
        return res
```

#### [1035. 不相交的线](https://leetcode-cn.com/problems/uncrossed-lines/)

我们在两条独立的水平线上按给定的顺序写下 A 和 B 中的整数。

现在，我们可以绘制一些连接两个数字 A[i] 和 B[j] 的直线，只要 A[i] == B[j]，且我们绘制的直线不与任何其他连线（非水平线）相交。

以这种方法绘制线条，并返回我们可以绘制的最大连线数。

```
输入：A = [1,4,2], B = [1,2,4]
输出：2
解释：
我们可以画出两条不交叉的线，如上图所示。
我们无法画出第三条不相交的直线，因为从 A[1]=4 到 B[2]=4 的直线将与从 A[2]=2 到 B[1]=2 的直线相交。
示例 2：

输入：A = [2,5,1,2,5], B = [10,5,2,1,5,2]
输出：3
示例 3：

输入：A = [1,3,7,1,7,5], B = [1,9,2,5,1]
输出：2
```

```python
class Solution:
    def maxUncrossedLines(self, A: List[int], B: List[int]) -> int:
        m,n = len(A),len(B)
        ans = 0
        dp = [[0] * (n + 1) for i in range(m + 1)]

        for i in range(1,m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    dp[i][j] = 0
                elif A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    ans = max(dp[i][j],ans)
                else:
                    dp[i][j] = max(dp[i - 1][j],dp[i][j - 1])

        return ans  
```

#### [300. 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

```
示例 1：

输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。
示例 2：

输入：nums = [0,1,0,3,2,3]
输出：4
示例 3：

输入：nums = [7,7,7,7,7,7,7]
输出：1

```

思路一：

方法与lcs一样，重新克隆一个该数组，对其进行排序，然后运行lcs的代码。时间复杂度O(nlogn + n^2)

```python
def lengthOfLIS1(nums):
    sortNums = sorted(nums)
    n = len(nums)
    return LCS(nums, sortNums, n, n)
    
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        m = len(nums)
        nums2 = list(set(nums[:]))
        nums2.sort()

        n = len(nums2)
        dp = [[0 for a in range(n + 1)] for b in range(m + 1)]

        result = 0
        for a in range(m + 1):
            for b in range(n + 1):
                if a == 0 or b == 0:
                    dp[a][b] = 0
                elif nums[a - 1] == nums2[b - 1]:
                    dp[a][b] = dp[a - 1][b - 1] + 1
                    result = max(result,dp[a][b])
                else:
                    dp[a][b] = max(dp[a - 1][b],dp[a][b - 1])

        return result       
```

思路二：

Li代表最大递增子序列长度在第i个位置时停止，则新建一个dp数组,维护位于该位置时候的最大递增子序列的长度，则递推公式为L i= 1 + max(L j) 条件为 0 < = j < i,并且a[j] < a[i]，若j不存在就是1,时间复杂度为O(n^2)。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        if not nums:return 0
        dp = [1] * len(nums)

        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j] + 1)
        return max(dp) 
```

思路三：

维护一个dp数组，遍历数组元素，当前的元素大于dp数组的最大值的时候，说明可以形成一个递增子序列，此时像dp数组中append该元素，否则，查找该元素在dp数组中的插入位置，将其替换。时间复杂度为O(nlogn)，因为此时的查找为有序查找，二分查找即可。

https://lucifer.ren/blog/2020/06/20/LIS/

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        def search(temp,left,right,target):
            if left == right:
                return left
            mid = left + (right - left) // 2
            return search(temp, mid+1, right, target) if temp[mid]<target else search(temp, left, mid, target)
       
        dp = []
        for num in nums:
            pos = search(dp,0,len(dp),num)
            if pos >= len(dp):
                dp.append(num)
            else:
                dp[pos] = num 
        return len(dp) 
```

#### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

难度：困难

给你两个单词 `word1` 和 `word2`，请你计算出将 `word1` 转换成 `word2` 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

- 插入一个字符
- 删除一个字符
- 替换一个字符

```python
示例 1：

输入：word1 = "horse", word2 = "ros"
输出：3
解释：
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
示例 2：

输入：word1 = "intention", word2 = "execution"
输出：5
解释：
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')

```

思路：

https://leetcode-cn.com/problems/edit-distance/solution/bian-ji-ju-chi-by-leetcode-solution/

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:

        m,n = len(word1),len(word2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m +1):
            dp[i][0] = i

        for j in range(n +1):
            dp[0][j] = j

        for i in range(1,m + 1):
            for j in range(1,n + 1):
                diff = 1
                if word1[i - 1] == word2[j - 1]:
                    diff = 0
                dp[i][j] = min(dp[i - 1][j] + 1,dp[i][j - 1] + 1,dp[i - 1][j - 1] + diff) 

        return dp[m][n]


状态转移公式：
若最后一个字符相等，则min(dp[i - 1][j] + 1,dp[i][j - 1] + 1,dp[i - 1][j - 1])
若最后一个字符不等，则min(dp[i - 1][j],dp[i][j - 1],dp[i - 1][j - 1]) + 1
```

#### [435. 无重叠区间](https://leetcode-cn.com/problems/non-overlapping-intervals/)

给定一个区间的集合，找到需要移除区间的最小数量，使剩余区间互不重叠。

注意:

可以认为区间的终点总是大于它的起点。
区间 [1,2] 和 [2,3] 的边界相互“接触”，但没有相互重叠。

```
示例 1:

输入: [ [1,2], [2,3], [3,4], [1,3] ]

输出: 1

解释: 移除 [1,3] 后，剩下的区间没有重叠。
示例 2:

输入: [ [1,2], [1,2], [1,2] ]

输出: 2

解释: 你需要移除两个 [1,2] 来使剩下的区间没有重叠。
示例 3:

输入: [ [1,2], [2,3] ]

输出: 0

解释: 你不需要移除任何区间，因为它们已经是无重叠的了。
```

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        n = len(intervals)
        if n == 0: return 0
        dp = [1] * n
        ans = 1
        intervals.sort(key=lambda a: a[0])

        for i in range(len(intervals)):
            for j in range(i - 1, -1, -1):
                if intervals[i][0] >= intervals[j][1]:
                    dp[i] = max(dp[i], dp[j] + 1)
                    break # 由于是按照开始时间排序的, 因此可以剪枝
                
        return n - max(dp)
```

思路二：贪心 todo

#### [646. 最长数对链](https://leetcode-cn.com/problems/maximum-length-of-pair-chain/)

给出 n 个数对。 在每一个数对中，第一个数字总是比第二个数字小。

现在，我们定义一种跟随关系，当且仅当 b < c 时，数对(c, d) 才可以跟在 (a, b) 后面。我们用这种形式来构造一个数对链。

给定一个数对集合，找出能够形成的最长数对链的长度。你不需要用到所有的数对，你可以以任何顺序选择其中的一些数对来构造。

**示例：**

```
输入：[[1,2], [2,3], [3,4]]
输出：2
解释：最长的数对链是 [1,2] -> [3,4]
```

```python
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:

        n = len(pairs)
        if n == 0:return 0
        dp = [1] * n
        ans = 1
        pairs.sort(key = lambda x:x[0])

        for i in range(n):
            for j in range(i - 1,-1,-1):
                if pairs[i][0] > pairs[j][1]:
                    dp[i] = max(dp[i],dp[j] + 1)
                    ans = max(dp[i],ans)
                    break
        return ans
```

#### [452. 用最少数量的箭引爆气球](https://leetcode-cn.com/problems/minimum-number-of-arrows-to-burst-balloons/)

在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。

一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。

给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。

```python
示例 1：

输入：points = [[10,16],[2,8],[1,6],[7,12]]
输出：2
解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球
示例 2：

输入：points = [[1,2],[3,4],[5,6],[7,8]]
输出：4
示例 3：

输入：points = [[1,2],[2,3],[3,4],[4,5]]
输出：2
示例 4：

输入：points = [[1,2]]
输出：1
示例 5：

输入：points = [[2,3],[2,3]]
输出：1

```

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:

        n = len(points)
        if n == 0 : return 0
        dp = [1] * n
        ans = 1
        points.sort(key = lambda x:x[0])

        for i in range(n):
            for j in range(i - 1,-1,-1):
                if points[i][0] > points[j][1]:
                    dp[i] = max(dp[i],dp[j] + 1)
                    break
        return max(dp) 
```

对于LIS问题，都可以进行贪心+二分的优化

```python
class Solution:
    def lengthOfLIS(self, A: List[int]) -> int:
        d = []
        for a in A:
            i = bisect.bisect_left(d, a)
            if i < len(d):
                d[i] = a
            elif not d or d[-1] < a:
                d.append(a)
        return len(d)
```

#### [673. 最长递增子序列的个数](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/)

给定一个未排序的整数数组，找到最长递增子序列的个数。

示例 1:

输入: [1,3,5,4,7]
输出: 2
解释: 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]。
示例 2:

输入: [2,2,2,2,2]
输出: 5
解释: 最长递增子序列的长度是1，并且存在5个子序列的长度为1，因此输出5。

思路：在上面的基础上，再维护一个count数组，来记录达到当前的最长LIS的时候有多少个

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:

        n = len(nums)
        if n == 0:return 0
        dp = [1] * n
        count = [1] * n

        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    if dp[j] + 1 > dp[i]: #代表第一次遇到最长子序列
                        dp[i] = max(dp[i],dp[j] + 1)
                        count[i] = count[j]
                    elif dp[j] + 1 == dp[i]:#代表之前已经遇到过最长子序列
                        count[i] += count[j]
        res,tmp = 0,max(dp)
        for i in range(n):
            if dp[i] == tmp:
                res += count[i]       
        return res
```

难度：困难


