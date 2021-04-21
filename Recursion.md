# Recursion

[TOC]

## 递归的基本思想

1. 可以把解决的问题转化为一个子问题，而这个字问题的解决方法仍与原来的解决方法相同，只是问题的规模变小。
2. 原问题可以通过子问题解决而组合解决。
3. 编码小心，很难调试，出错会导致进入无线递归。

python递归的max deepth 1000左右

递归的本质为数学 归纳法，需要写出base 条件和公式。

一般解决的是P问题，涉及时间复杂度为2^n或者n!的大部分为NP问题，常用方法为递归求解。

#### 简单求和

求1…n的和

```python
def simplysum(n):
    if n == 0 : return 0
    return n + simplysum(n - 1)
```

#### 阶乘

```python
def factorial_recursive(n):
    if n == 1:return 1
    return n * factorial_recursive(n - 1)
```

#### 斐波那契数列

```python
def fibonacci1(n):
    a, b = 0, 1
    for i in range(1, n+1):
        a, b = b, a + b
    return a    
    
def fibonacci2(n):
    if (n <= 2): 
        return 1
    return fibonacci2(n-1) + fibonacci2(n-2)

def fibonacci3(n):
    if (n <= 1): 
        return (n,0)
    (a, b) = fibonacci3(n-1)
    return (a+b, a)
```

#### 打印尺子

打印如下内容：

1

1 2 1

1 2 1 3 1 2 1

1 2 1 3 1 2 1 4 1 2 1 3 1 2 1

```python
def ruler_bad(n):
    if n == 1 : return "1"
    return ruler_bad(n - 1) + str(n) + ruler_bad(n - 1)
  
def ruler2(n):
    result = ""
    for i in range(1, n+1):
        result = result + str(i) + " " + result
    return result  
```

#### 数学表达式

给定两个整数 a	≤	b,	编写一个程序,通过加1和乘以2的方式，将a变换成b

例如：

23 = ((5 * 2 + 1) * 2 + 1)

113 = ((((11 + 1) + 1) + 1) * 2 * 2 * 2 + 1)

```python
def intSeq(a, b):
    if (a == b):
        return str(a)
    if (b % 2 == 1):
        return "(" + intSeq(a, b-1) + " + 1)"
    if (b < a * 2):
        return "(" + intSeq(a, b-1) + " + 1)"
    return intSeq(a, b/2) + " * 2"
```

#### 汉诺塔

```python
def hanoi(n, start, end, by):
    if (n==1):
        print("Move from " + start + " to " + end)
    else:
        hanoi(n-1, start, by, end)
        hanoi(1, start, end, by)
        hanoi(n-1, by, end, start)
        
n = 3
hanoi(n, "START", "END", "BY")        
```

#### 格雷码

![截屏2020-11-13 上午11.11.33](/Users/yangning/Desktop/ownspace/数据结构与算法/万门/images/截屏2020-11-13 上午11.11.33.png)

在尺子的基础上加入了move的enter与exit

```python
def moves_ins(n, forward):
    if n == 0: return
    moves_ins(n-1, True)
    print("enter ", n) if forward else print("exit  ", n)
    moves_ins(n-1, False) 
```

#### Permutation of Size K

akes two parameters n and k, and prints out all P(n, k) = n! / (n-k)! permutations that contain exactly k of the n elements. when k = 2 and n = 4

ab ac ad ba bc bd ca cb cd da db dc

```python
def permSizeK(result, nums, k):
    if k == 0:
        print(result)
    for i in range(len(nums)):
        permSizeK(result+str(nums[i]), nums[0:i] + nums[i+1:], k - 1)
```

#### Letter Case Permutation

Enumerate all uppercase/lowercase permutation for any letter specified in input

For example,

word = “medium-one”

Rule = “io”

solutions = [“medium-one”, “medIum-one”, “medium-One”, “medIum-One”]

```python
def permLetter(word, rule):
    rule = rule.lower()
    for c in rule:
        keys.add(c)
    permHelper(word, rule, 0, "")
    
def permHelper(word, rule, index, prefix):
    length = len(word)
    for i in range(index, length):
        c = word[i]
        if (c in keys):
            permHelper(word, rule, i + 1, prefix + c)
            c = c.upper()
            permHelper(word, rule, i + 1, prefix + c)
        else:
            prefix += c
    if (len(prefix) == len(word)):
        results.add(prefix) 
```
