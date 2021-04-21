# String

[TOC]

## 字符串题型  

- 基于字符计数的问题  
- 同字母异序  
- 回文  
- 二进制字符串  
- 子序列  
- 模式搜索  
- 其他

### 基于字符计数的问题  

#### 偶数子串的数量  

给定一串0到9的数字，任务是计算在将整数转换为偶数时的子串的数量。  

```
Input : str = "1234". 

Output : 6 

“2”, “4”, “12”, “34”, “234”, “1234是6个子字符串，它们是偶数。
```

思路：通过观察发现，只要当前的位置为偶数，则其与其前面的数组都可以组成偶数,所以index + 1就是当前的元素可以组成的偶数的数量。

```python
def evenNum(s):
    count = 0
    for i in range(len(s)):
        if int(s[i]) % 2 == 0:
            count += i + 1
    return count
print (evenNum("1234"))
```

#### [821. 字符的最短距离](https://leetcode-cn.com/problems/shortest-distance-to-a-character/)

给定一个字符串 S 和一个字符 C。返回一个代表字符串 S 中每个字符到字符串 S 中的字符 C 的最短距离的数组。

```
示例：

输入：S = "loveleetcode", C = 'e'
输出：[3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]
```

```python
class Solution:
    def shortestToChar(self, S: str, C: str) -> List[int]:

        prev = float('-inf')
        ans = []
        for i, x in enumerate(S):
            if x == C: prev = i
            ans.append(i - prev)

        prev = float('inf')
        for i in range(len(S) - 1, -1, -1):
            if S[i] == C: prev = i
            ans[i] = min(ans[i], prev - i)

        return ans
```

### 计数问题 II  

#### [551. 学生出勤记录 I](https://leetcode-cn.com/problems/student-attendance-record-i/)

给你一个代表学生出勤记录的字符串。 该记录只包含以下三个字符： 

 'A' : 缺席.  

'L' : 迟到.  

'P' : 出席.  

如果学生的出勤记录不包含多于一个“A”（缺席）或超过两个连续的“L”（迟 到），则可以获得奖励。  

你需要根据他的出勤记录来返回学生是否可以得到奖励。

思路：不包含多于一个A，则只需要统计s中A出现的次数，超过两个连续的L，则可以判断LLL in s

```python
def checkRecord(s):
    return not (s.count('A') > 1 or 'LLL' in s)
```

### 计数问题 III  

#### 对具有相同首尾字符的子字符进行计数  

给出一个字符串S，我们需要找到所有连续的子字符串开始和结束的字符都相同的 计数。  

```
Input : S = "abcab"  

Output : 7  

"abcab" 有15个子字符串  在上面的子串中，有7个子串满足要求：a，abca，b，bcab，c，a和b。.
```

思路：two-pointer 两个指针，如果两个指针的值相同，则count加1

```python
def countSub(s):
    result = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            if (s[i] == s[j]):
                result += 1
    return result
```

#### 字符串中最大连续重复字符 

 给定一个字符串，其任务是在字符串中查找最大连续重复字符。

```python
def maxRepeating(s):
    n = len(s)
    count = 0
    result = s[0]
    local = 1
    
    for i in range(n):
        if (i < n - 1 and s[i] == s[i+1]):
            local += 1
        else:
            if (local > count):
                count = local
                result = s[i]
            local = 1
    return result
```

#### 排序数组中删除重复 

```python
def removeDuplicates(A):
    if not A:return 0
    
    new_tail = 0
    for i in range(1,len(A)):
        if A[i] != A[new_tail]:
            new_tail += 1
            A[new_tail] = A[i]
    for j in range(new_tail + 1,len(A)):
        A[j] = 'X'
        
    print ("A",A)    
    return new_tail + 1        

A = [1,1,2]
print (removeDuplicates(A))
```

### 同字母异序

#### [242. 有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/)

给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

```
示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false

```

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):return False
        return sorted(s) == sorted(t)
      
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:

        from collections import Counter
        return Counter(s) == Counter(t)
      
```

#### [438. 找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。

字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。

说明：

字母异位词指字母相同，但排列不同的字符串。
不考虑答案输出的顺序。

```python
示例 1:

输入:
s: "cbaebabacd" p: "abc"

输出:
[0, 6]

解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的字母异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的字母异位词。
 示例 2:

输入:
s: "abab" p: "ab"

输出:
[0, 1, 2]

解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的字母异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的字母异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的字母异位词。


```

思路：滑动窗口，先将元素放入count中，然后滑动插入，删除

```python
from collections import Counter
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:

        result = []
        p_counter = Counter(p)
        s_counter = Counter(s[:len(p) - 1])

        i,j = 0,len(p) - 1
        while i < len(s) and j < len(s):
            s_counter[s[j]] += 1
            if p_counter == s_counter:
                result.append(i)
            s_counter[s[i]] -= 1
            if s_counter[s[i]] == 0:
                del s_counter[s[i]]
            i += 1
            j += 1    

        return result 
```

#### 查找同字母异序词的映射  

给定两个列表A和B，B是A的一个同字母组。这意味着B是通过随机化A中元素的顺 序而创建的。  

我们希望找到一个从A到B的索引映射P。映射P [i] = j意味着列表A中的第i个元 素出现在B的索引为j的位置。  

这些列表A和B可能包含重复项。 如果有多个答案，则输出它们中的任意一个。  

```
例如，给定  

A = [12, 28, 46, 32, 50]  

B = [50, 12, 32, 46, 28]  

应当返回[1, 4, 3, 2, 0]  

P[0] = 1因为A的第0个元素出现在B[1]处，并且P[1] = 4，因为A的第1个元素出 现在B [4]处，依此类推。
```

```python
def anagramMappings1(A, B):
    answer = []
    for a in A:
        for i,b in enumerate(B):
            if a == b:
                answer.append(i)
                break
    return answer

def anagramMappings2(A, B):
    return [B.index(a) for a in A]
    
def anagramMappings3(A, B):
    d = {}
    for i,b in enumerate(B):
        d[b] = i
    return [d[a] for a in A]    
```

### 回文

#### 移位  

给定两个字符串s1和s2，写一段程序说明s2是否是s1 的移位。

思路：判断是否是其的移位，技巧性的，直接将s1载拼接一个s1,然后判断s2 in s1s1。

```python
def areRotations(s1,s2):
    size_1 = len(s1)
    size_2 = len(s2)
    
    if size_1 != size_2:return 0
    
    temp = s1 + s1
    return temp.count(s2) > 0

string1 = "AACD"
string2 = "ACDA"
print (areRotations(string1, string2))
```

#### 移位 II 

 写一个函数 rotate(arr[], d, n) 将大小为n的数组arr[] 移位d个单位。

思路：三段反转法，首先对0 - k -1进行反转，然后对k - n -1进行反转，最后将整个数组反转

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
```

#### [9. 回文数](https://leetcode-cn.com/problems/palindrome-number/)

判断一个整数是否是回文数。回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

```
示例 1:

输入: 121
输出: true
示例 2:

输入: -121
输出: false
解释: 从左向右读, 为 -121 。 从右向左读, 为 121- 。因此它不是一个回文数。
示例 3:

输入: 10
输出: false
解释: 从右向左读, 为 01 。因此它不是一个回文数。


```

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        _ = str(x)
        return _ == _[::-1]
      
def isPalindrome(s):
    for i in range(len(s) // 2):
        if s[i] != s[- 1 - i]:
            return False

    return True  
  
#不允许转成str时候的方式
def isPalindrome(x):
    if x < 0:
        return False

    ranger = 1
    while x // ranger >= 10:
        ranger *= 10
    print(ranger)
    while x:
        left = x // ranger
        right = x % 10
        if left != right:
            return False

        x = (x % ranger) // 10
        ranger //= 100

    return True
```

#### 移位回文  

检查给定的字符串是否是一个回文字符串的移位。

思路：和上述一致，s1+s1,然后判断窗口为k的是s1+s1中是否存在回文

```python
def isRotationOfPalindrome(s):
 
    # If string itself is palindrome
    if isPalindrome(s):
        return True
 
    # Now try all rotations one by one
    n = len(s)
    for i in range(len(s) - 1):
        s1 = s[i+1:n]
        s2 = s[0:i+1]
 
        # Check if this rotation is palindrome
        s1 += s2
        if isPalindrome(s1):
            return True
 
    return False

def isRotationOfPalindrome(s):
    n = len(s)
    s = s + s
    for i in range(n):
        if isPalindrome(s[i : i + n]):
            return True
    return False
```

#### 重排回文  

给定一个字符串，检查字符串中的各字符是否可以构成一个回文字符串

思路：对于回文串来讲，其中奇数只可能存在一个，其余都是偶数

```python
from collections import Counter
def canRearrage(s):
    odd = 0
    counter = Counter(s)
    for key in counter.keys():
        if counter[key] % 2 == 1:
            odd += 1
        if odd > 1:
            return False
    return True
    
print(canRearrage("baaaa"))
print(canRearrage("ababaa"))
print(canRearrage("ababcbaab"))
print(canRearrage("ababcbaa"))    
```

#### [409. 最长回文串](https://leetcode-cn.com/problems/longest-palindrome/)

给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。

在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。

```python
注意:
假设字符串的长度不会超过 1010。

示例 1:

输入:
"abccccdd"

输出:
7

解释:
我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。


```

思路：统计找出其中的所有偶数，奇数只是随机取一个

```python
from collections import Counter
class Solution:
    def longestPalindrome(self, s: str) -> int:
        answer = 0
        counter = Counter(s)
        for key in counter.keys():
            v = counter[key]
            answer += v // 2 * 2
            if answer % 2 == 0 and v % 2 == 1:
                answer +=  1
        return answer  
```

### 子串

#### [3. 无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

给定一个字符串，请你找出其中不含有重复字符的 **最长子串** 的长度。

```
示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

```

思路一：滑动窗口,一步一步移动窗口

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        usedChar = set()
        max_length = 0
        i = j = 0
        while i < len(s) and j < len(s):
            if s[j] not in usedChar:
                usedChar.add(s[j])
                j += 1
                max_length = max(max_length,j - i)
            else:
                usedChar.remove(s[i])
                i += 1
        return max_length  
```

思路二：滑动窗口，判断重复出现，直接跳过中间的元素

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        start = max_length = 0
        userChar = {}

        for i,c in enumerate(s):
            if c in userChar and start <= userChar[c]:
                start = userChar[c] + 1
            else:
                max_length = max(max_length,i - start + 1)

            userChar[c] = i    
        return max_length
```

#### [76. 最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

难度：困难

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。

```python
示例 1：

输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
示例 2：

输入：s = "a", t = "a"
输出："a"

```

思路一：使用两个指针，其中 end 用于「延伸」现有窗口，start 用于「收缩」现有窗口。在任意时刻，只有一个指针运动，而另一个保持静止。我们在 s 上使用双指针，通过移动 end指针不断扩张窗口，当窗口包含 t 全部所需的字符后，如果能收缩，我们就收缩窗口直到得到最小窗口。

```python
class Solution:
    def minWindow(self, s: 'str', t: 'str') -> 'str':
        from collections import Counter
        t = Counter(t)
        lookup = Counter()
        start = 0
        end = 0
        min_len = float("inf")
        res = ""
        while end < len(s):
            lookup[s[end]] += 1
            end += 1
            while all(map(lambda x: lookup[x] >= t[x], t.keys())):
                if end - start < min_len:
                    res = s[start:end]
                    min_len = end - start
                lookup[s[start]] -= 1
                start += 1
        return res

```

思路二：

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:

        if len(t) > len(s):return ""
        n = len(t)
        count = n
        ct = collections.Counter(t)
        left = right = 0
        min_length = float("inf")
        not_found = 1
        ansleft = ansright = 0

        for i in range(len(s)):
            if ct[s[i]] > 0:
                count -= 1
            ct[s[i]] -= 1

            while count == 0:
                right = i
                not_found = 0
                if right - left < min_length:
                    min_length = right - left
                    ansleft = left
                    ansright = right

                if ct[s[left]] == 0:
                    count += 1
                ct[s[left]] += 1
                left += 1

        if not_found == 1:
            return ""      

        return s[ansleft:ansright + 1]                  
```

### 子序列

#### 最长子序列  

给定一个字符串‘s’和一个整数k，找到其他字符串‘t’，使得‘t’是给定字 符串‘s’的最大子序列，同时‘t’的每个字符在字符串s中必须至少出现k次。  

```
Input: s = "baaabaacba“, k = 3 

Output : baaabaaba
```

```python
import collections

def longestSub(s, k):
    result = list()
    c = collections.Counter(s)
    for i in s:
        if (c[i] >= k):
            result.append(i)
    return "".join(result)
```

#### 检查子序列

给定两个字符串str1和str2。 确定str1是否是str2的子序列。 子序列是可以通 过删除一些元素而不改变其余元素的顺序从另一个序列派生的序列。  

```
Input: str1 = "AXY", str2 = "ADXCPY"  Output: True (str1 是 str2的子序列) 

 Input: str1 = "AXY", str2 = "YADXCP"  Output: False (str1 不是str2的子序列)
```

```python
def isSubSequence(string1, string2, m, n):
    # Base Cases
    if m == 0:    return True
    if n == 0:    return False
 
    # If last characters of two strings are matching
    if string1[m-1] == string2[n-1]:
        return isSubSequence(string1, string2, m-1, n-1)
 
    # If last characters are not matching
    return isSubSequence(string1, string2, m, n-1)
```

```python
def isSubSequence(str1, str2):
    m = len(str1)
    n = len(str2)
    j = 0   # Index of str1
    i = 0   # Index of str2
    while j < m and i < n:
        if str1[j] == str2[i]:  
            j = j + 1
        i = i + 1
         
    return j == m
```

#### 通过删除给定字符串的字符得到字典中最长的单词

给一个字典和一个字符串‘str’，找到字典中最长的字符串，它可以通过删除给 定的‘str’中的一些字符来形成。  

```
Input: dict = {"ale", "apple", "monkey", "plea"} ,  

str = "abpcplea"  

Output : apple
```

```python
def findLongestString(words, s):
    result = ""
    length = 0
    for w in words:
        if length < len(w) and isSubSequence(w, s):
            result = w
            length = len(w)
    return result
```

#### 找出所有子列元素之和的加和

给定一列n个整数.。找出所有子列元素之和的加和

思路：求所有可能存在的子数组的和，可以通过找到所有的子数组然后再求和的方式进行。太慢O(2^n)

数学方式，考察每一个元素的出现与不出现，其出现的次数为2^(n - 1)

```python
def sumSub(arr):
    ans = sum(arr)
    return ans * pow(2, len(arr) - 1)
    
arr = [5, 6, 8]
print(sumSub(arr))  
```

#### [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

```
示例 1:

输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。


```

思路一：brust force

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        if not s:return 0
        
        count = 1
        for a in range(len(s)):
            repeat = set()
            repeat.add(s[a])
            for b in range(a + 1,len(s)):
                if s[b] not in repeat:
                    repeat.add(s[b])
                else:
                    break    
            count = max(count,len(repeat))

        return count
```

思路二：双指针+哈希表

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic, res, i = {}, 0, -1
        for j in range(len(s)):
            if s[j] in dic:
                i = max(dic[s[j]], i) # 更新左指针 i
            dic[s[j]] = j # 哈希表记录
            res = max(res, j - i) # 更新结果
        return res
```

思路三：dp

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = {}
        res = tmp = 0
        for j in range(len(s)):
            i = dic.get(s[j], -1) # 获取索引 i
            dic[s[j]] = j # 更新哈希表
            tmp = tmp + 1 if tmp < j - i else j - i # dp[j - 1] -> dp[j]
            res = max(res, tmp) # max(dp[j - 1], dp[j])
        return res
```

### 模式搜索

思路：rolling-hash

#### [28. 实现 strStr()](https://leetcode-cn.com/problems/implement-strstr/)

实现 strStr() 函数。

给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

```
示例 1:

输入: haystack = "hello", needle = "ll"
输出: 2
示例 2:

输入: haystack = "aaaaa", needle = "bba"
输出: -1

```

思路一：

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:

        l,n = len(needle),len(haystack)

        for start in range(n - l + 1):
            if haystack[start:start + l] == needle:
                return start
        return -1
```

思路二：rolling-hash

https://leetcode-cn.com/problems/implement-strstr/solution/shi-xian-strstr-by-leetcode/

时间复杂度：渐进O(N)

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

思路三：

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:

        if needle not in haystack:
            return -1
        else:
            return haystack.index(needle)
```

#### 敏感词  

给定的句子作为输入，用星号‘*’检查替换的单词

```python
def censor(text, word):
    word_list = text.split()
    result = ''
    stars = '*' * len(word)
    count = 0
    index = 0;
    for i in word_list:
        if i == word:
            word_list[index] = stars
        index += 1
 
    # join the words
    result =' '.join(word_list)
    return result
  
word = "Barcelona"

text = "It wasn't any place close to a vintage performance but Barcelona eventually overcame \
Deportivo La Coruna and secured the La Liga title for the 25th time. It is the 9th La \
Liga win for Barcelona in the last 14 seasons and the 7th in the last 10. In the last \
ten, only Real Madrid twice and Atletico Madrid have broken Barcelona run of Liga success."

censor(text, word)  
```

#### 用C替换所有出现的字符串AB  

给定一个可能包含一个“AB”的字符串str。 将str中的所有“AB”替换为“C

```python
def translate(st) :
    l = len(st)
     
    if (l < 2) :
        return
 
    i = 0 # Index in modified string
    j = 0 # Index in original string
 
    while (j < l - 1) :
        # Replace occurrence of "AB" with "C"
        if (st[j] == 'A' and st[j + 1] == 'B') :
             
            # Increment j by 2
            j += 2
            st[i] = 'C'
            i += 1
            continue
         
        st[i] = st[j]
        i += 1
        j += 1
 
    if (j == l - 1) :
        st[i] = st[j]
        i += 1
 
    # add a null character to
    # terminate string
    return i
    
    
st = list("helloABworldABGfGAAAB")
length = translate(st)
for i in range(length):
    print(st[i])    
```

#### 数出“1（0+）1”模式的发生次数

给定一个字母数字字符串，找出给定字符串中出现模式1（0+）1的次数。 这里， （0+）表示存在连续0的非空序列。

```python
def patternCount(s):
    last = s[0]
    i = 1
    counter = 0
    while (i < len(s)):
         
        # We found 0 and last character was '1',
        # state change
        if (s[i] == '0' and last == '1'):
            while (s[i] == '0' and i < len(s)):
                i += 1
                if (i == len(s)):
                    return counter
                # After the stream of 0's, we got a '1',
                # counter incremented
                if (s[i] == '1'): 
                    counter += 1
         
        # Last character stored 
        last = s[i]
        i += 1
     
    return counter
  
s = "100001abc1010100"
print(patternCount(s))  
```
