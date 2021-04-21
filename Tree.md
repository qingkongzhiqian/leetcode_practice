# Tree

[TOC]

二叉树可以通过存储一个节点的数据加两个子指针来实现

### 二叉树

二叉树ADT扩展了树ADT，即它继承了树ADT的所有方法

```python
class Node:
    def __init__(self,item,left = None,right = None):
        self._item = item
        self._left = left
        self._right = right
        
class BinarySearchTree:
    def __init__(self,root = None):
        self._root = root
        
    def add(self,value):
        self._root = self.__add(self._root,value)
    
    def __add(self,node,value):
        if node is None:
            return Node(value)
        if value == node._item:
            pass
        else:
            if value < node._item:
                node._left = self.__add(node._left,value)
            else:
                node._right = self.__add(node._right,value)
        return node        
    
    def get(self,key):
        return self.__get(self._root,key)
    
    def __get(self,node,key):
        if node is None:
            return None
        if key == node._item:
            return node._item
        if key < node._item:
            return self.__get(node._left,key)
        else:
            return self.__get(node._right,key)
            
    def remove(self,key):
        self._root = self.__remove(self._root,key)
    
    def __remove(self,node,key):
        if node is None:return None
        if key < node._item: #case 1:no child #case 2:one child
            node._left = self.__remove(node._left,key)
        elif key > node._item:
            node._right = self.__remove(node._right,key)
        else:
            if node._left is None:
                node = node._right
            elif node._right is None:
                node = node._left
            else:
                node._item = self.__get_max(node._left)
                node._left = self.__remove(node._left,node._item)
        return node                

    def get_max(self):
        return self.__get_max(self._root)
    
    def __get_max(self,node):
        if node is None:return None
        while node._right:
            node = node._right
        return node._item
    
    def print_inorder(self):
        self.__print_inorder(self._root)
    
    def __print_inorder(self,node):
        if node is None:return
        self.__print_inorder(node._left)
        print ("inorder node",node._item)
        self.__print_inorder(node._right)
    
    def print_preorder(self):
        self.__print_preorder(self._root)
    
    def __print_preorder(self,node):
        if node is None:return 
        print ("preorder node",node._item)
        self.__print_preorder(node._left)
        self.__print_preorder(node._right)
        
    def print_postorder(self):
        self.__print_postorder(self._root)
        
    def __print_postorder(self,node):
        if node is None:return 
        self.__print_postorder(node._left)
        self.__print_postorder(node._right)
        print ("postorder node",node._item)
        
bst = BinarySearchTree()
numbers = [6, 4, 8, 7, 9, 2, 1, 3, 5, 13, 11, 10, 12]
for i in numbers:
    bst.add(i)

bst.print_inorder()
bst.print_postorder()
bst.print_preorder()
        

```

### 第一类问题：递归实现

#### [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

给定一个二叉树，找出其最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 [3,9,20,null,null,15,7]，

```
  3
   / \
  9  20
    /  \
   15   7
```

返回它的最大深度 3 。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        return self.__maxDepth(root)

    def __maxDepth(self,node):
        if node is None:return 0
        left_depth = self.__maxDepth(node.left)
        right_depth = self.__maxDepth(node.right)
        return max(left_depth,right_depth) + 1
```

#### [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。

示例 1：

输入：root = [3,9,20,null,null,15,7]
输出：true

思路一：

```
从顶至底（暴力法）
此方法容易想到，但会产生大量重复计算，时间复杂度较高。

思路是构造一个获取当前节点最大深度的方法 depth(root) ，通过比较此子树的左右子树的最大高度差abs(depth(root.left) - depth(root.right))，来判断此子树是否是二叉平衡树。若树的所有子树都平衡时，此树才平衡。

算法流程：
isBalanced(root) ：判断树 root 是否平衡

特例处理： 若树根节点 root 为空，则直接返回 truetrue ；
返回值： 所有子树都需要满足平衡树性质，因此以下三者使用与逻辑 \&\&&& 连接；
abs(self.depth(root.left) - self.depth(root.right)) <= 1 ：判断 当前子树 是否是平衡树；
self.isBalanced(root.left) ： 先序遍历递归，判断 当前子树的左子树 是否是平衡树；
self.isBalanced(root.right) ： 先序遍历递归，判断 当前子树的右子树 是否是平衡树；
depth(root) ： 计算树 root 的最大高度

终止条件： 当 root 为空，即越过叶子节点，则返回高度 00 ；
返回值： 返回左 / 右子树的最大高度加 11 。
复杂度分析：
时间复杂度 O(Nlog_2 N)O(Nlog 2 N)： 最差情况下， isBalanced(root) 遍历树所有节点，占用 O(N)O(N) ；判断每个节点的最大高度 depth(root) 需要遍历 各子树的所有节点 ，子树的节点数的复杂度为 O(log_2 N)O(log 2N) 。
空间复杂度 O(N)O(N)： 最差情况下（树退化为链表时），系统递归需要使用 O(N)O(N) 的栈空间。
```

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if not root: return True
        return abs(self.__depth(root.left) - self.__depth(root.right)) <= 1 and \
            self.isBalanced(root.left) and self.isBalanced(root.right)

    def __depth(self, root):
        if not root: return 0
        return max(self.__depth(root.left), self.__depth(root.right)) + 1
```

```python
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if not root:return True
        if abs(self.__max_depth(root.left,0) - self.__max_depth(root.right,0)) > 1:return False
        return self.isBalanced(root.left) and self.isBalanced(root.right) 


    def __max_depth(self,root,depth):
        if root is None:return 0
        left = self.__max_depth(root.left,depth + 1)
        right = self.__max_depth(root.right,depth + 1)
        return max(left,right) + 1
```

#### [270. 最接近的二叉搜索树值](https://leetcode-cn.com/problems/closest-binary-search-tree-value/)

给定一个不为空的二叉搜索树和一个目标值 target，请在该二叉搜索树中找到最接近目标值 target 的数值。

注意：

给定的目标值 target 是一个浮点数
题目保证在该二叉搜索树中只会存在一个最接近目标值的数
示例：

```
输入: root = [4,2,5,1,3]，目标值 target = 3.714286

    4
   / \
  2   5
 / \
1   3

输出: 4
```

```python
class Solution:
    def closestValue(self, root: TreeNode, target: float) -> int:
        closest = root.val
        while root:
            closest = min(root.val, closest, key = lambda x: abs(target - x)) #注意这行
            root = root.left if target < root.val else root.right
        return closest
```

#### [面试题 04.05. 合法二叉搜索树](https://leetcode-cn.com/problems/legal-binary-search-tree-lcci/)

实现一个函数，检查一棵二叉树是否为二叉搜索树。

```python

示例 1:
输入:
    2
   / \
  1   3
输出: true

示例 2:
输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。
```

正确思路：

判断每一个节点的取值范围，对于根节点，其值为负无穷到正无穷，对于一个节点的左子树，其最大值不超过当前节点，最小值为父节点的最小值，对于一个节点的右子树，其最小值大于当前节点，最大值不超过父节点的最大值。

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        return self.__isValidBST(root,float('inf'),-float('inf'))

    def __isValidBST(self,node,maxval,minval):
        if node is None:return True
        if node.val <= minval or node.val >= maxval:
            return False
        return self.__isValidBST(node.left,node.val,minval) and self.__isValidBST(node.right,maxval,node.val)    

```

思路：中序遍历来判断数组严格递增

```python
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:

        if root is None:return True

        result = []
        WHITE,GRAY = 1,0
        stack = [(WHITE,root)]

        while stack:
            color,node = stack.pop()
            if node is None:continue
            if color == WHITE:
                stack.append((WHITE,node.right))
                stack.append((GRAY,node))
                stack.append((WHITE,node.left))
            else:
                if len(result) == 0:
                    result.append(node.val)   
                else:
                    if node.val > result[-1]:
                        result.append(node.val)   
                    else:
                        return False

        return True  
```

#### [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

```python
例如输入：

   4
  /  \
 2   7
 / \  / \
1  3 6  9

镜像输出：

   4
  /  \
 7   2
 / \  / \
9  6 3  1 
```

示例 1：

输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]

思路:递归调用

```python
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        return self.__mirrorTree(root)


    def __mirrorTree(self,node):

        if node is None:return 
        self.__mirrorTree(node.left)
        self.__mirrorTree(node.right)

        node.left,node.right = node.right,node.left
        return node
```

#### [100. 相同的树](https://leetcode-cn.com/problems/same-tree/)

给定两个二叉树，编写一个函数来检验它们是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

**示例 1:**

```
输入:       1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

输出: true

```

**示例 2:**

```
输入:      1          1
          /           \
         2             2

        [1,2],     [1,null,2]

输出: false

```

```python
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        return self.__isSameTree(p,q)

    def __isSameTree(self,p,q):
        if p is None and q is None:
            return True
        if p is not None and q is not None:
            return p.val == q.val and self.__isSameTree(p.left,q.left) and self.__isSameTree(p.right,q.right)
        else:
            return False 
```

#### [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

```
1
  / \
 2  2
 / \ / \
3  4 4  3

但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

  1
  / \
 2  2
  \  \
  3   3
```

```
示例 1：

输入：root = [1,2,2,3,4,4,3]
输出：true
示例 2：

输入：root = [1,2,2,null,3,null,3]
输出：false
```

```python
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root is None : return True
        return self.__isSymmetric(root.left,root.right)

    def __isSymmetric(self,left,right):
        if left is None and right is None:
            return True
        if left is not None and right is not None:
            return left.val == right.val and self.__isSymmetric(left.left,right.right)and self.__isSymmetric(left.right,right.left)
        else:
            return False
```

### 第二类问题：DFS，BFS

#### [700. 二叉搜索树中的搜索](https://leetcode-cn.com/problems/search-in-a-binary-search-tree/)

给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。

例如，

```python
给定二叉搜索树:

        4
       / \
      2   7
     / \
    1   3

和值: 2

你应该返回如下子树:
     2     
     / \   
    1   3
```

Case 一：recursion

```python
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        return self.__searchBST(root,val)

    def __searchBST(self,node,val):    

        if node is None:return None
        if node.val == val:return node
        elif node.val < val:  #注意些return的递归调用
            return self.__searchBST(node.right,val)
        else:
            return self.__searchBST(node.left,val) 
```

Case 二：iterative

```python
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        node = root
        while node is not None:
            if val == node.val:
                return node
            if val < node.val:
                node = node.left
            else:
                node = node.right
        return None
```

#### [701. 二叉搜索树中的插入操作](https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/)

给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。

注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 。

case 一：recursion

```python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        root = self.__insertIntoBST(root,val)
        return root

    def __insertIntoBST(self,node,val):
        if node is None:return TreeNode(val)
        if node.val == val:pass
        else:
            if val < node.val:
                node.left = self.__insertIntoBST(node.left,val)
            else:
                node.right = self.__insertIntoBST(node.right,val)
        return node 
```

Case 二：iterative

```python
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        
        if root is None:
            root = TreeNode(val)
            return root

        current = root
        parent = None
        while True:
            parent = current
            if val == current.val:return
            if val < current.val:
                current = current.left
                if current is None:
                    parent.left = TreeNode(val)
                    return root
            else:
                current = current.right
                if current is None:
                    parent.right = TreeNode(val)
                    return root
```

#### [450. 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/)

给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

首先找到需要删除的节点；
如果找到了，删除它。
说明： 要求算法时间复杂度为 O(h)，h 为树的高度。

**示例:**

```python
root = [5,3,6,2,4,null,7]
key = 3

    5
   / \
  3   6
 / \   \
2   4   7

给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。

一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。

    5
   / \
  4   6
 /     \
2       7

另一个正确答案是 [5,2,6,null,4,null,7]。

    5
   / \
  2   6
   \   \
    4   7

```

思路：

BST删除节点的思路（Hibbard删除）

分为三个情况

case 1：该节点没有任何的children节点---直接删除即可

Case 2:  该节点有一个children节点---用children节点直接替换

case 3: 该节点有两个children节点— 找到待删除节点的左子树的最大值（或者右子树的最小值）进行替换，然后再将替换前的节点删除

```python
class Solution:
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        root = self.__deleteNode(root,key)
        return root

    def __deleteNode(self,node,key):
        if node is None:return node
        if key < node.val:
            node.left = self.__deleteNode(node.left,key)
        elif key > node.val:
            node.right = self.__deleteNode(node.right,key)
        else:
            if node.left is None:
                node = node.right
            elif node.right is None:
                node = node.left
            else:
                node.val = self.__get_max(node.left)
                node.left = self.__deleteNode(node.left,node.val)
        return node        

    def __get_max(self,node):

        if node is None:return None
        while node.right is not None:
            node = node.right
        return node.val 
```

#### [剑指 Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

例如，给出

```
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
```

返回如下的二叉树：

```python
    3
   / \
  9  20
    /  \
   15   7
```

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        return self.__buildTree(preorder,inorder)

    def __buildTree(self,preorder,inorder):
        if inorder:
            ind = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[ind])
            root.left = self.__buildTree(preorder, inorder[0:ind])
            root.right = self.__buildTree(preorder, inorder[ind+1:])
            return root
```

#### 重建二叉树 II

输入某二叉树的后序遍历和中序遍历的结果，请重建该二叉树。假设输入的后序遍历和中序遍历的结果中都不含重复的数字。

例如，给出

```
中序遍历 preorder = [9,3,15,20,7]
后序遍历 inorder = [9,15,7,20,3]
```

```python
class Solution:
    def buildTree(self,inorder: List[int],postorder: List[int],) -> TreeNode:
        return self.__buildTree(inorder,postorder)

    def __buildTree(self,preorder,inorder):
        if not inorder or not preorder:return None
        root = Node(postorder.pop())
        inorderIndex = inorder.index(root._item)
        root.right = buildTree(inorder[inorderIndex+1:], postorder)
        root.left = buildTree(inorder[:inorderIndex], postorder)
        return root
```

#### [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

思路：这不就是从根节点开始，到叶子节点结束的所有路径**搜索出来**，挑选出和为目标值的路径么？这里的开始点是根节点， 结束点是叶子节点，目标就是路径。

```
示例:
给定如下二叉树，以及目标和 sum = 22，

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

        def backtrack(nodes,path,target):
            # 空节点
            if not nodes: return
            # 叶子节点
            if nodes and not nodes.left and not nodes.right:
                if target == nodes.val:
                    result.append((path + [nodes.val]).copy())
                return
            # 选择
            path.append(nodes.val)
            # 递归左右子树
            backtrack(nodes.left, path,target - nodes.val)
            backtrack(nodes.right, path,target - nodes.val)
            # 撤销选择
            path.pop(-1)
        result = []
        # 入口，路径，目标值全部传进去，其中路径和path都是扩展的参数
        backtrack(root,[],sum)
        return result

```

#### [1372. 二叉树中的最长交错路径](https://leetcode-cn.com/problems/longest-zigzag-path-in-a-binary-tree/)

给你一棵以 root 为根的二叉树，二叉树中的交错路径定义如下：

选择二叉树中 任意 节点和一个方向（左或者右）。
如果前进方向为右，那么移动到当前节点的的右子节点，否则移动到它的左子节点。
改变前进方向：左变右或者右变左。
重复第二步和第三步，直到你在树中无法继续移动。
交错路径的长度定义为：访问过的节点数目 - 1（单个节点的路径长度为 0 ）。

请你返回给定树中最长 交错路径 的长度。

```
示例 1：



输入：root = [1,null,1,1,1,null,null,1,1,null,1,null,null,null,1,null,1]
输出：3
解释：蓝色节点为树中最长交错路径（右 -> 左 -> 右）。
```

思路：这不就是从任意节点**开始**，到任意节点**结束**的所有交错**路径**全部**搜索出来**，挑选出最长的么？这里的开始点是树中的任意节点，结束点也是任意节点，目标就是最长的交错路径。

对于入口是任意节点的题目，我们都可以方便地使用**双递归**来完成

对于这种交错类的题目，一个好用的技巧是使用 -1 和 1 来记录方向，这样我们就可以通过乘以 -1 得到另外一个方向。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    @lru_cache(None)
    def dfs(self, root, dir):
        if not root:
            return 0
        if dir == -1:
            return int(root.left != None) + self.dfs(root.left, dir * -1)
        return int(root.right != None) + self.dfs(root.right, dir * -1)

    def longestZigZag(self, root: TreeNode) -> int:
        if not root:
            return 0
        return max(self.dfs(root, 1), self.dfs(root, -1), self.longestZigZag(root.left), self.longestZigZag(root.right))
```

#### [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1。

```
给定有序数组: [-10,-3,0,5,9],

一个可能的答案是：[0,-3,9,-10,null,5]，它可以表示下面这个高度平衡二叉搜索树：

      0
     / \
   -3   9
   /   /
 -10  5

```

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        return self.__sortedArrayToBST(nums)

    def __sortedArrayToBST(self,nums):

        if not nums:return 
        middle = len(nums) // 2
        root = TreeNode(nums[middle])
        root.left = self.__sortedArrayToBST(nums[:middle])
        root.right = self.__sortedArrayToBST(nums[middle + 1:])
        return root
```

#### [109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/)

给定一个单链表，其中的元素按升序排序，将其转换为高度平衡的二叉搜索树。

本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。

**示例:**

```python
给定的有序链表： [-10, -3, 0, 5, 9],

一个可能的答案是：[0, -3, 9, -10, null, 5], 它可以表示下面这个高度平衡二叉搜索树：

      0
     / \
   -3   9
   /   /
 -10  5

```

```python
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        
        if not head:return head
        pre,slow,fast = None,head,head

        while fast and fast.next:
            fast = fast.next.next
            pre = slow
            slow = slow.next

        if pre:
            pre.next = None

        node = TreeNode(slow.val)
        if slow == fast:
            return node   

        node.left = self.sortedListToBST(head)
        node.right = self.sortedListToBST(slow.next)  
        return node 
```

#### [1382. 将二叉搜索树变平衡](https://leetcode-cn.com/problems/balance-a-binary-search-tree/)

给你一棵二叉搜索树，请你返回一棵 **平衡后** 的二叉搜索树，新生成的树应该与原来的树有着相同的节点值。

如果一棵二叉搜索树中，每个节点的两棵子树高度差不超过 1 ，我们就称这棵二叉搜索树是 **平衡的** 。

如果有多种构造方法，请你返回任意一种。

```
输入：root = [1,null,2,null,3,null,4,null,null]
输出：[2,1,3,null,null,null,4]
解释：这不是唯一的正确答案，[3,1,4,null,2,null,null] 也是一个可行的构造方案。
```

 思路：中序遍历得到有序的数组，然后将有序数组转化为二叉搜索树

```python
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:

        nums = self.inorder(root)
        def dfs(start,end):
            if start == end:return TreeNode(nums[start])
            if start > end:return None
            mid = (start + end) // 2
            root = TreeNode(nums[mid])
            root.left = dfs(start,mid - 1)
            root.right = dfs(mid + 1,end)
            return root

        return dfs(0,len(nums) - 1)

    def inorder(self,node):
        if node is None:return []
        return self.inorder(node.left) + [node.val] + self.inorder(node.right)

```

#### [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。

说明: 叶子节点是指没有子节点的节点。

示例: 
给定如下二叉树，以及目标和 sum = 22，

```python
5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1

```

```python
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        return self.__hasPathSum(root,sum)

    def __hasPathSum(self,node,s):    
        if node is None:return False

        if not node.left and not node.right and node.val == s:
            return True

        s -= node.val
        return self.__hasPathSum(node.left,s) or self.__hasPathSum(node.right,s)  
```

#### [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

Ps：图论相关

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


```

返回:

```python
[
   [5,4,11,2],
   [5,8,4,5]
]
```

```python
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

#### [437. 路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

给定一个二叉树，它的每个结点都存放着一个整数值。

找出路径和等于给定数值的路径总数。

路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。

二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。

**示例：**

```python
root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。和等于 8 的路径有:

1.  5 -> 3
2.  5 -> 2 -> 1
3.  -3 -> 11

```

#### [面试题 04.12. 求和路径](https://leetcode-cn.com/problems/paths-with-sum-lcci/)

给定一棵二叉树，其中每个节点都含有一个整数数值(该值或正或负)。设计一个算法，打印节点数值总和等于某个给定值的所有路径的数量。注意，路径不一定非得从二叉树的根节点或叶节点开始或结束，但是其方向必须向下(只能从父节点指向子节点方向)。

示例:
给定如下二叉树，以及目标和 sum = 22，

```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1

```

```python
3
解释：和为 22 的路径有：[5,4,11,2], [5,8,4,5], [4,11,7]

```

求解：

binary search tree IV 最后一个题

#### [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]

```python
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。

```

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        return self.__lowestCommonAncestor(root,p,q)

    def __lowestCommonAncestor(self,node,p,q):    
        while node:
            if node.val > p.val and node.val > q.val:
                node = node.left
            elif node.val < p.val and node.val < q.val:
                node = node.right
            else:
                return node 
```

#### [129. 求根到叶子节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。

例如，从根到叶子节点路径 1->2->3 代表数字 123。

计算从根到叶子节点生成的所有数字之和。

说明: 叶子节点是指没有子节点的节点。

```python
示例 1:

输入: [1,2,3]
    1
   / \
  2   3
输出: 25
解释:
从根到叶子节点路径 1->2 代表数字 12.
从根到叶子节点路径 1->3 代表数字 13.
因此，数字总和 = 12 + 13 = 25.
示例 2:

输入: [4,9,0,5,1]
    4
   / \
  9   0
 / \
5   1
输出: 1026
解释:
从根到叶子节点路径 4->9->5 代表数字 495.
从根到叶子节点路径 4->9->1 代表数字 491.
从根到叶子节点路径 4->0 代表数字 40.
因此，数字总和 = 495 + 491 + 40 = 1026.

```

```python
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        
        def dfs(root,prevTotal):
            if not root:return 0
            total = prevTotal * 10 + root.val
            if not root.left and not root.right:
                return total
            else:
                return dfs(root.left,total) + dfs(root.right,total)
        return dfs(root,0)  
```


