# Hash Table

[TOC]

哈希表的使用：计数，判断是否存在，是否有重复

#### [136. 只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

```python
示例 1:

输入: [2,2,1]
输出: 1
示例 2:

输入: [4,1,2,1,2]
输出: 4
```

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        from collections import Counter
        single = Counter(nums)
        return [k for k in single if single[k] == 1][0]
```

#### 统计单词数

```python
from collections import Counter
def wordCount(s):
    wordcount = Counter(s.split())
    print(wordcount)
```

#### [387. 字符串中的第一个唯一字符](https://leetcode-cn.com/problems/first-unique-character-in-a-string/)

给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

```python
示例：

s = "leetcode"
返回 0

s = "loveleetcode"
返回 2
```

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        from collections import Counter
        count = Counter(s)
        
        for i in range(len(s)):
            if count[s[i]] == 1:return i

        return -1 
```

#### [349. 两个数组的交集](https://leetcode-cn.com/problems/intersection-of-two-arrays/)

给定两个数组，编写一个函数来计算它们的交集。

```python
示例 1：

输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]
示例 2：

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]


```

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return list(set(nums1) & set(nums2))
```

#### [350. 两个数组的交集 II](https://leetcode-cn.com/problems/intersection-of-two-arrays-ii/)

给定两个数组，编写一个函数来计算它们的交集。

```
示例 1：

输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]
示例 2:

输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]
```

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:

        dict1 = dict()
        for i in nums1:
            if i not in dict1:
                dict1[i] = 1
            else:
                dict1[i] += 1
        ret = []
        for i in nums2:
            if i in dict1 and dict1[i]>0:
                ret.append(i)
                dict1[i] -= 1
        return ret  
```

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:

        dict_1 = {}
        for i in nums1:
            dict_1[i] = dict_1.get(i,0) + 1

        result = []
        for i in nums2:
            if i in dict_1 and dict_1[i] > 0:
                result.append(i)
                dict_1[i] -= 1
        return result 
```

#### [771. 宝石与石头](https://leetcode-cn.com/problems/jewels-and-stones/)

给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。

J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和"A"是不同类型的石头。

```python
示例 1:

输入: J = "aA", S = "aAAbbbb"
输出: 3
示例 2:

输入: J = "z", S = "ZZ"
输出: 0
```

```python
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        from collections import Counter
        stone = Counter(S)
        result = 0
        for i in J:
            result += stone.get(i,0)
        return result 
```

写法二：

```python
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        A = set(J) 
        return sum(i in A for i in S)
```

#### [217. 存在重复元素](https://leetcode-cn.com/problems/contains-duplicate/)

给定一个整数数组，判断是否存在重复元素。

如果任意一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不相同，则返回 false 。

```
示例 1:

输入: [1,2,3,1]
输出: true
示例 2:

输入: [1,2,3,4]
输出: false
示例 3:

输入: [1,1,1,3,3,4,3,2,4,2]
输出: true
```

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return not len(list(set(nums))) == len(nums)
```

#### [219. 存在重复元素 II](https://leetcode-cn.com/problems/contains-duplicate-ii/)

给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，使得 nums [i] = nums [j]，并且 i 和 j 的差的 绝对值 至多为 k。

```
示例 1:

输入: nums = [1,2,3,1], k = 3
输出: true
示例 2:

输入: nums = [1,0,1,1], k = 1
输出: true
示例 3:

输入: nums = [1,2,3,1,2,3], k = 2
输出: false
```

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        dic = {}
        for i,v in enumerate(nums):
            if v in dic and i - dic[v] <= k:
                return True
            dic[v] = i
        return False 
```

#### [811. 子域名访问计数](https://leetcode-cn.com/problems/subdomain-visit-count/)

一个网站域名，如"discuss.leetcode.com"，包含了多个子域名。作为顶级域名，常用的有"com"，下一级则有"leetcode.com"，最低的一级为"discuss.leetcode.com"。当我们访问域名"discuss.leetcode.com"时，也同时访问了其父域名"leetcode.com"以及顶级域名 "com"。

给定一个带访问次数和域名的组合，要求分别计算每个域名被访问的次数。其格式为访问次数+空格+地址，例如："9001 discuss.leetcode.com"。

接下来会给出一组访问次数和域名组合的列表cpdomains 。要求解析出所有域名的访问次数，输出格式和输入格式相同，不限定先后顺序。

```python
示例 1:
输入: 
["9001 discuss.leetcode.com"]
输出: 
["9001 discuss.leetcode.com", "9001 leetcode.com", "9001 com"]
说明: 
例子中仅包含一个网站域名："discuss.leetcode.com"。按照前文假设，子域名"leetcode.com"和"com"都会被访问，所以它们都被访问了9001次。
示例 2
输入: 
["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
输出: 
["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
说明: 
按照假设，会访问"google.mail.com" 900次，"yahoo.com" 50次，"intel.mail.com" 1次，"wiki.org" 5次。
而对于父域名，会访问"mail.com" 900+1 = 901次，"com" 900 + 50 + 1 = 951次，和 "org" 5 次。
```

```python
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:

        count = collections.Counter()
        for domain in cpdomains:
            num,cp = domain.split()
            sub = cp.split('.')
            for i in range(len(sub)):
                count[".".join(sub[i:])] += int(num)

        return ["{} {}".format(ct,dom) for dom,ct in count.items()]
```

#### [500. 键盘行](https://leetcode-cn.com/problems/keyboard-row/)

给定一个单词列表，只返回可以使用在键盘同一行的字母打印出来的单词。键盘如下图所示。 

```
示例：

输入: ["Hello", "Alaska", "Dad", "Peace"]
输出: ["Alaska", "Dad"]
```

```python
class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        row_one = set('qwertyuiop')
        row_two = set('asdfghjkl')
        row_three = set('zxcvbnm')

        result = []
        for word in words:
            w = set(word.lower())
            if w.issubset(row_one) or w.issubset(row_two) or w.issubset(row_three):
                result.append(word)
        return result 
```

#### [290. 单词规律](https://leetcode-cn.com/problems/word-pattern/)

给定一种规律 `pattern` 和一个字符串 `str` ，判断 `str` 是否遵循相同的规律。

这里的 **遵循** 指完全匹配，例如， `pattern` 里的每个字母和字符串 `str` 中的每个非空单词之间存在着双向连接的对应规律。

**示例1:**

```
输入: pattern = "abba", str = "dog cat cat dog"
输出: true
```

**示例 2:**

```
输入:pattern = "abba", str = "dog cat cat fish"
输出: false
```

**示例 3:**

```
输入: pattern = "aaaa", str = "dog cat cat dog"
输出: false
```

**示例 4:**

```
输入: pattern = "abba", str = "dog dog dog dog"
输出: false
```

思路：对于示例4，需要在哈希表中存储key-value的同时，也要存储value-key，这样才能保证一一对应关系。

```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        p = pattern
        t = s.split()
        return len(set(zip(p, t))) == len(set(p)) == len(set(t)) and len(p) == len(t)
```

```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        word2ch = dict()
        ch2word = dict()
        words = s.split()
        if len(pattern) != len(words):
            return False
        
        for ch, word in zip(pattern, words):
            if (word in word2ch and word2ch[word] != ch) or (ch in ch2word and ch2word[ch] != word):
                return False
            word2ch[word] = ch
            ch2word[ch] = word
    
        return True
```

#### [49. 字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

```
示例:

输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]

```

思路：对每一个子串排序，存在hash表中

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

        dic = collections.defaultdict(list)

        for s in strs:
            key = "".join(sorted(s))
            dic[key].append(s)

        return list(dic.values())  
```

