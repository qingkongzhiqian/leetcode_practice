# Backtracing



[TOC]

https://mp.weixin.qq.com/s/r73thpBnK1tXndFDtlsdCQ

回溯法模板：

```
result = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return

    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
```

#### [组合总和](https://leetcode-cn.com/problems/combination-sum/)

给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的数字可以无限制重复被选取。

说明：

所有数字（包括 target）都是正整数。
解集不能包含重复的组合。 

```
示例 1：

输入：candidates = [2,3,6,7], target = 7,
所求解集为：
[
  [7],
  [2,2,3]
]
示例 2：

输入：candidates = [2,3,5], target = 8,
所求解集为：
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]

```

```python
'''
回溯法，层层递减，得到符合条件的路径就加入结果集中，超出则剪纸；
主要是要注意一些细节，避免重复
'''        
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
      

        def backtrack(nums,target,path,index,size):
            if target == 0:
                result.append(path[:])
            else:
                for i in range(index,size):
                    _target = target - nums[i]
                    if _target < 0:break
                    _path = path[:]
                    _path.append(nums[i])
                    backtrack(nums,_target,_path,i,size)

        candidates.sort()
        result = []
        
        backtrack(candidates,target,[],0,len(candidates))
        return result                
                
```

#### [组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

说明：

所有数字（包括目标数）都是正整数。
解集不能包含重复的组合。 

```
示例 1:

输入: candidates = [10,1,2,7,6,1,5], target = 8,
所求解集为:
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]
示例 2:

输入: candidates = [2,5,2,1,2], target = 5,
所求解集为:
[
  [1,2,2],
  [5]
]

```

```python
'''
与39题的区别就是不能重用元素，而元素可能有重复；
不能重用好解决，回溯的index往下一个就行；
元素可能有重复，就让结果的去重麻烦一些
'''
# 如果存在重复的元素，前一个元素已经遍历了后一个元素与之后元素组合的所有可能
# 注意是i > begin 而不是 0

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        
        def backtrack(nums,target,path,index,size):

            if target == 0:
                result.append(path[:])
            else:
                for i in range(index,size):
                    
                    _target = target - nums[i]
                    if _target < 0:break
                    if i > index and nums[i] == nums[i - 1]:continue
                    _path = path[:]
                    _path.append(nums[i])
                    backtrack(nums,_target,_path,i+1,size)
                
        size = len(candidates)
        result = []
        candidates.sort()
        backtrack(candidates,target,[],0,size)    
        return result

```

#### [全排列](https://leetcode-cn.com/problems/permutations/)

给定一个 没有重复 数字的序列，返回其所有可能的全排列。

```python
示例:

输入: [1,2,3]
输出:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

```python
#Solution 1
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        import itertools
        return [i for i in itertools.permutations(nums)]

#Solution 2
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(nums,path):
            if len(nums) <= 0:
                result.append(path[:])
            else:
                for i in nums:
                    _path = path[:]
                    _path.append(i)
                    _nums = nums[:]
                    _nums.remove(i)
                    backtrack(_nums,_path)
        result = []
        backtrack(nums,[])
        return result 
```

#### [全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。 

```
示例 1：

输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]
示例 2：

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]


```

```python
#Solution 1
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        import itertools
        return list(set(itertools.permutations(nums)))
      
#Solution 2
#注意这里的i 大于0 就行

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        
        def backtrack(nums,path):
            if len(nums) <= 0:
                result.append(path[:])
            else:
                for i in range(len(nums)):
                    if i > 0 and nums[i] == nums[i - 1]:continue
                    _path = path[:]
                    _path.append(nums[i])
                    _nums = nums[:]
                    _nums.remove(nums[i])
                    backtrack(_nums,_path)

        nums.sort()
        result = []
        backtrack(nums,[])
        return result
```

#### [排列序列](https://leetcode-cn.com/problems/permutation-sequence/)

给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：

"123"
"132"
"213"
"231"
"312"
"321"
给定 n 和 k，返回第 k 个排列。

```python
示例 1：

输入：n = 3, k = 3
输出："213"
示例 2：

输入：n = 4, k = 9
输出："2314"
示例 3：

输入：n = 3, k = 1
输出："123"

```

```python
import math

class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        res = ""
        candidates = [str(i) for i in range(1, n + 1)]

        while n != 0:
            facto = math.factorial(n - 1)
            # i 表示前面被我们排除的组数，也就是k所在的组的下标
            # k // facto 是不行的， 比如在 k % facto == 0的情况下就会有问题
            i = math.ceil(k / facto) - 1
            # 我们把candidates[i]加入到结果集，然后将其弹出candidates（不能重复使用元素）
            res += candidates[i]
            candidates.pop(i)
            # k 缩小了 facto *  i
            k -= facto * i
            # 每次迭代我们实际上就处理了一个元素，n 减去 1，当n == 0 说明全部处理完成，我们退出循环
            n -= 1
        return res
```

Todo

#### [子集](https://leetcode-cn.com/problems/subsets/)

给你一个整数数组 nums ，返回该数组所有可能的子集（幂集）。解集不能包含重复的子集。

```
示例 1：

输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
示例 2：

输入：nums = [0]
输出：[[],[0]]

```

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        
        def backtrack(nums,path,index,size):
            result.append(path[:])
            for i in range(index,size):
                _path = path[:]
                _path.append(nums[i])
                backtrack(nums,_path,i + 1,size)

        result = []
        backtrack(nums,[],0,len(nums))
        return result   
            
            
```

#### [子集 II](https://leetcode-cn.com/problems/subsets-ii/)

给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

说明：解集不能包含重复的子集。

```python
示例:

输入: [1,2,2]
输出:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]

```

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for num in nums: 
            res += [ i + [num] for i in res if (i + [num]) not in res]
        return res
      
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        
        def backtrack(nums,path,index,size):
            result.append(path[:])
            for i in range(index,size):
                if i > index and nums[i] == nums[i - 1]:continue
                _path = path[:]
                _path.append(nums[i])
                backtrack(nums,_path,i + 1,size)

        result = []
        nums.sort()
        backtrack(nums,[],0,len(nums))
        return result      
```

https://leetcode-solution-leetcode-pp.gitbook.io/leetcode-solution/medium/90.subsets-ii

#### [路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

给定一个二叉树和一个目标和，找到所有从根节点到叶子节点路径总和等于给定目标和的路径。

说明: 叶子节点是指没有子节点的节点。

示例:
给定如下二叉树，以及目标和 sum = 22，

```python
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1

返回:

[
   [5,4,11,2],
   [5,8,4,5]
]
```

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:return []
        result = []
        self.__pathSum(root,sum,[],result)
        return result

    def __pathSum(self,node,s,ls,result):
        if not node.right and not node.left and node.val == s:
            ls.append(node.val)
            result.append(ls)
        if node.left:
            self.__pathSum(node.left,s-node.val,ls + [node.val],result) 
        if node.right:
            self.__pathSum(node.right,s-node.val,ls + [node.val],result)
```

#### [分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。

返回 s 所有可能的分割方案。

```
示例:

输入: "aab"
输出:
[
  ["aa","b"],
  ["a","a","b"]
]


```

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        '''
        回溯法
        '''
        res = []
        def helper(s,path):
            '''
            如果是空字符串，说明已经处理完毕
            否则逐个字符往前测试，判断是否是回文
            如果是，则处理剩余字符串，并将已经得到的列表作为参数
            '''
            if not s:
                res.append(path)
            for i in range(1,len(s) + 1):
                if s[:i] == s[:i][::-1]:
                    _path = path[:]
                    _path.append(s[:i])
                    helper(s[i:],_path)    

        helper(s,[])
        return res
```

#### [括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

数字 *n* 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

```python
示例：

输入：n = 3
输出：[
       "((()))",
       "(()())",
       "(())()",
       "()(())",
       "()()()"
     ]

```

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:

        res = []
        def dfs(l,r,s):
            if l > n or r > n:return
            if l == r == n:res.append(s)
            if l < r:return
            dfs(l + 1,r,s + '(')
            dfs(l,r + 1,s + ')')
        dfs(0,0,'')    
        return res
```

#### [电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

**示例:**

```
输入："23"
输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
```

```python
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        # 0-9
        self.d = [" "," ","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"]
        self.res = []
        self.dfs(digits, 0, "")
        return self.res

    def dfs(self, digits, index, s):
        # 递归的终止条件,用index记录每次遍历到字符串的位置
        if index == len(digits):
            self.res.append(s)
            return
        # 获取当前数字
        c = digits[index]
        # print(c, int(c))
        # 获取数字对应字母
        letters = self.d[int(c)]
        # 遍历字符串
        for l in letters:
            # 调用下一层
            self.dfs(digits, index+1, s+l)
```





