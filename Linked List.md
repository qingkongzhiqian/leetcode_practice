# Linked List

[TOC]

### LinkedList实现

```python

class Node:
    def __init__(self,value = None,next = None):
        self.value = value
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = Node()
        self.tail = None
        self.length = 0
        
    def add_last(self,value):
        new_node = Node(value,None)
        node = self.head
        while node.next:
            node = node.next
        node.next = new_node
        self.length += 1
    
    def add_first(self,value):
        node = Node(value,None)
        node.next = self.head.next
        self.head.next = node
        self.length += 1
        
    def add(self,index,value):
        if index < 0 or index > self.length:
            raise Error("Index is out of bound")
        if not self.head.next:
            raise Error("LinkedList is empty")
        new_node = Node(value)
        node = self.head
        for i in range(index):
            node = node.next
        new_node.next = node.next
        node.next = new_node
        self.length += 1

    def get_first(self):
        if not self.head.next:
            raise Error("LinkedList is empty")
        return self.head.next

    def get_last(self):
        if not self.head.next:
            raise Error("LinkedList is empty")
        node = self.head
        while node.next:
            node = node.next
        return node
    
    def get(self,index):
        if index < 0 or index >= self.length:
            raise Error("Index is out of bound")
        if not self.head.next:
            raise Error("LinkedList is empty")
        node = self.head.next
        for i in range(index):
            node = node.next
        return node
    
    def remove_first(self):
        if not self.head.next:
            raise Error("LinkedList is empty")
        value = self.head.next
        self.head.next = self.head.next.next
        self.length -= 1
        return value
    
    def remove_last(self):
        if not self.head.next:
            raise Error("LinkedList is empty")
        node = self.head.next
        while node.next.next:
            node = node.next
        value = node.next    
        node.next = None  
        return value
    
    def remove(self,index):
        if index < 0 or index >= self.length:
            raise Error("Index is out of bound")
        if not self.head.next:
            raise Error("LinkedList is empty")
        node = self.head
        for i in range(index):
            node = node.next
        value = node.next
        node.next = node.next.next
        self.length -= 1
        return value
    
    def printlist(self):
        node = self.head.next
        while node:
            print ("node",node.value)
            node = node.next
            

ll = LinkedList()
mm = LinkedList()
for i in range(1,10):
    ll.add_last(i)
    
for i in range(100,110):
    mm.add_last(i)
            
ll.printlist()
mm.printlist()
            
```

#### [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

注意：此题对比原题有改动

```python
示例 1:

输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
示例 2:

输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
```

思路1：找到待删除元素的pre节点和next节点，然后pre.next = node.next。

```python

class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:

        if head.val == val:return head.next  #要删除的是头结点
        node = head
        while node.next.val != val:
            node = node.next
        node.next = node.next.next 
        return head 

```

思路2：找到待删除元素的下一个节点，将当前node的value替换为node.next的value,然后node.next = node.next.next（偷梁换柱）

```python
def delete_node(node):
    print(node.value)
    node.value = node.next.value
    node.next = node.next.next
```

#### [876. 链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

给定一个头结点为 head 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

思路：

two pointer 一个fast,一个slow,当fast到达最后的时候slow就是中间点了。

```
示例 1：

输入：[1,2,3,4,5]
输出：此列表中的结点 3 (序列化形式：[3,4,5])
返回的结点值为 3 。 (测评系统对该结点序列化表述是 [3,4,5])。
注意，我们返回了一个 ListNode 类型的对象 ans，这样：
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, 以及 ans.next.next.next = NULL.
示例 2：

输入：[1,2,3,4,5,6]
输出：此列表中的结点 4 (序列化形式：[4,5,6])
由于该列表有两个中间结点，值分别为 3 和 4，我们返回第二个结点。
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:

        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next

        if fast.next:
            slow = slow.next

        return slow
```

#### [面试题 02.08. 环路检测](https://leetcode-cn.com/problems/linked-list-cycle-lcci/)

给定一个链表，如果它是有环链表，实现一个算法返回环路的开头节点。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

```python
示例 1：

输入：head = [3,2,0,-4], pos = 1
输出：tail connects to node index 1
解释：链表中有一个环，其尾部连接到第二个节点。
示例 2：

输入：head = [1,2], pos = 0
输出：tail connects to node index 0
解释：链表中有一个环，其尾部连接到第一个节点。
示例 3：

输入：head = [1], pos = -1
输出：no cycle
解释：链表中没有环。
```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

#判断是否有环的code
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:

        slow = fast = head

        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next

            if fast == slow:
                return slow

        return -1  
```

思路：

1. 设置一个快指针为F，每次走过两个节点。
2. 设置一个慢指针为S，每次走过一个节点。
3. 两个指针从链表头同时开始走。
4. 如果F最终指向null，则说明链表没有环。
5. 如果最终两个指针相遇，则说明链表有环，此时将F放置回链表头，同时只让F每次走过一个节点。
6. 当两个指针相遇时，所在位置即是环起点。

证明：

设:

disSlow = L_1 + L_2

disFast = L_1 + L_2 + nC

知:

2(L1 + L2) = L1 + L2 + nC

(因为速度是两倍)

所以:

L1 + L2 = nC

又已知:

L3+ L2 = nC

故:

L1=L3

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:

        slow = fast = head

        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next

            if fast == slow:
                fast = head
                break

        if fast is None or fast.next is None:
            return None

        while fast != slow:
            fast = fast.next
            slow = slow.next

        return slow                  

```

#### [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。 

```python
示例：

给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.


```

```python
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:

        fast = slow = head
        
        while k > 0:
            fast = fast.next
            k -= 1

        while fast is not None and slow is not None:
            fast = fast.next
            slow = slow.next

        return slow   
```

#### 分半：给定一个列表，把它分成两个列表，一个是前半部分，一个是 后半部分。

```python
def split(head):
    if (head is None):
        return
    slow = head
    fast = head
    front_last_node = slow
    while (fast is not None):
        front_last_node = slow
        slow = slow.next
        fast = fast.next.next if fast.next is not None else None
    front_last_node.next = None
    front = head
    back = slow
    return (front, back)
```

#### [19. 删除链表的倒数第N个节点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

示例：

给定一个链表: 1->2->3->4->5, 和 n = 2.

当删除了倒数第二个节点后，链表变为 1->2->3->5.
说明：

给定的 n 保证是有效的。

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:

        fast = slow = head

        while n > 0:
            fast = fast.next
            n -= 1

        if fast is None: #注意不要丢掉
            return slow.next

        while fast.next is not None:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next
        return head
```

#### [21. 合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

```python
示例：

输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

```python
#Solution 1 iterative
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next 
```

```python
#Solution 2 recursive
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:

        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2   
```

#### [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

编写一个程序，找到两个单链表相交的起始节点。

输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

思路一：

首先找到每一个链表的长度，较长的链表先前移动几步，然后两个一起向前走，直到A与B相等。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:

        len_A = len_B = 0
        tmpA,tmpB = headA,headB
        while tmpA:
            tmpA = tmpA.next
            len_A +=1
        while tmpB:
            tmpB = tmpB.next
            len_B +=1
        tmpA,tmpB = headA,headB
        if len_A > len_B:
            for i in range(len_A - len_B):
                tmpA = tmpA.next 
        else:
            for i in range(len_B - len_A):
                tmpB = tmpB.next 
        while tmpA != tmpB:
            tmpA = tmpA.next
            tmpB = tmpB.next
        return tmpA  
```

思路二：

由于链表有相交的部分，所以相交的部分的长度是相等的。差距为不相交的部分的。对于A链表来说，其先从前向后走，走到最末尾，将其送至B链表的头部，对于B链表来说，其先从前向后走，走到最末尾，将其送至A链表的头部，则两个链表一起向前走，值相等的点即为相交的点。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:

        curA,curB = headA,headB

        while curA != curB:
            curA = curA.next if curA else headB
            curB = curB.next if curB else headA

        return curA 
        
```

#### [148. 排序链表](https://leetcode-cn.com/problems/sort-list/)

给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。

进阶：

你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:

        if head is None or head.next is None:return head
        middle = self._get_middle(head)
        right = middle.next
        middle.next = None
        return self._merge(self.sortList(head),self.sortList(right))

    def _merge(self,left,right):
        
        dummy = dummyHead = ListNode(0)
        while left and right:
            if left.val < right.val:
                dummyHead.next = left
                left = left.next
            else:
                dummyHead.next = right
                right = right.next    
            dummyHead = dummyHead.next #不能忘了

        if left:
            dummyHead.next = left
        if right:
            dummyHead.next = right

        return dummy.next            

    def _get_middle(self,head):

        if not head:return head
        fast = slow = head    

        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
            
        return slow    
```

#### [86. 分隔链表](https://leetcode-cn.com/problems/partition-list/)

给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于 x 的节点都在大于或等于 x 的节点之前。

你应当保留两个分区中每个节点的初始相对位置。

```
示例:

输入: head = 1->4->3->2->5->2, x = 3
输出: 1->2->2->4->3->5
```

```python
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:

        left_head = ListNode(None)
        right_head = ListNode(None)

        left,right = left_head,right_head
        while head:
            if head.val < x:
                left.next = head
                left = left.next
            else:
                right.next = head
                right = right.next
            head = head.next

        right.next = None
        left.next = right_head.next
        return left_head.next 
```

#### [剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

```
示例:

输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

```python
#Solution 1 iterative
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:

        prev = None
        current = head
        nxt = None
        
        while current is not None:
            nxt = current.next
            current.next = prev
            prev = current
            current = nxt

        return prev   
#step1:找到next节点
#step2:将当前节点指向前一个节点
#step3:prev往前走一个
#step4:current往前走一个
```

```python
#Solution 2 recursion
class Solution:

    def reverseList(self, head: ListNode) -> ListNode:

        if head is None or head.next is None : return head

        prev = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return prev
```

思路：https://zhuanlan.zhihu.com/p/147171393

#### [92. 反转链表 II](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

反转从位置 *m* 到 *n* 的链表。请使用一趟扫描完成反转。

**说明:**
1 ≤ *m* ≤ *n* ≤ 链表长度。

**示例:**

```python
输入: 1->2->3->4->5->NULL, m = 2, n = 4
输出: 1->4->3->2->5->NULL
```

```python
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:

        dummy = ListNode(0)
        dummy.next = head
        prev = dummy

        for i in range(m - 1):
            prev = prev.next

        result = None
        current = prev.next
        for i in range(n- m + 1):
            nxt = current.next
            current.next = result
            result = current
            current = nxt

        prev.next.next = current
        prev.next = result
        return dummy.next   
```

#### [24. 两两交换链表中的节点](https://leetcode-cn.com/problems/swap-nodes-in-pairs/)

给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

**你不能只是单纯的改变节点内部的值**，而是需要实际的进行节点交换。

```python
输入：head = [1,2,3,4]
输出：[2,1,4,3]

输入：head = []
输出：[]

输入：head = [1]
输出：[1]
```

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = cur = ListNode(0)
        dummy.next = head
        
        while cur.next and cur.next.next:
            p1 = cur.next
            p2 = cur.next.next
            cur.next  = p2
            p1.next = p2.next
            p2.next = p1
            cur = cur.next.next
        return dummy.next
```

https://leetcode-cn.com/problems/swap-nodes-in-pairs/solution/liang-liang-jiao-huan-lian-biao-zhong-de-jie-di-91/

#### [25. K 个一组翻转链表](https://leetcode-cn.com/problems/reverse-nodes-in-k-group/)

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

```python
示例：

给你这个链表：1->2->3->4->5

当 k = 2 时，应当返回: 2->1->4->3->5

当 k = 3 时，应当返回: 3->2->1->4->5
```

```python
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:

        if head is None or k < 2:return head
        next_head = head
        for i in range(k - 1):
            next_head = next_head.next
            if next_head is None:
                return head

        ret = next_head
        current = head

        while next_head:
            tail = current
            prev = None
            for i in range(k):
                if next_head:
                    next_head = next_head.next
                nxt = current.next
                current.next = prev
                prev = current
                current = nxt
            tail.next = next_head or current
            
        return ret 
```

https://leetcode-cn.com/problems/reverse-nodes-in-k-group/solution/k-ge-yi-zu-fan-zhuan-lian-biao-by-leetcode-solutio/

#### [面试题 02.06. 回文链表](https://leetcode-cn.com/problems/palindrome-linked-list-lcci/)

编写一个函数，检查输入的链表是否是回文的。

```python
示例 1：

输入： 1->2
输出： false 
示例 2：

输入： 1->2->2->1
输出： true 
```

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
      
        rev = None
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            rev,rev.next,slow = slow,rev,slow.next
        if fast:
            slow = slow.next
        while rev and rev.val == slow.val:
            slow = slow.next
            rev = rev.next

        return not rev  

class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        vals = []
        current_node = head
        while current_node is not None:
            vals.append(current_node.val)
            current_node = current_node.next
        return vals == vals[::-1]      
```

#### [83. 删除排序链表中的重复元素](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

```python

示例 1:

输入: 1->1->2
输出: 1->2
示例 2:

输入: 1->1->2->3->3
输出: 1->2->3
```

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:

        if head is None:return head
        node = head
        while node.next:
            if node.val == node.next.val:
                node.next = node.next.next
            else:
                node = node.next
        return head 
```

#### [82. 删除排序链表中的重复元素 II](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。

```python

示例 1:

输入: 1->2->3->3->4->4->5
输出: 1->2->5
示例 2:

输入: 1->1->1->2->3
输出: 2->3
```

```python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:

        dummy = prev = ListNode(0)
        dummy.next = head

        while head and head.next:
            if head.val == head.next.val:
                while head and head.next and head.val == head.next.val:
                    head = head.next
                head = head.next
                prev.next = head
            else:
                prev = prev.next
                head = head.next
        return dummy.next 
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
示例 2：

输入：lists = []
输出：[]
示例 3：

输入：lists = [[]]
输出：[]


```

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        
        if not lists or len(lists) == 0:return None
        import heapq
        heap = []
        for node in lists:
            while node:
                heapq.heappush(heap,node.val)
                node = node.next

        dummy = ListNode(None)
        current = dummy
        while heap:
            temp_node = ListNode(heappop(heap))
            current.next = temp_node
            current = current.next
        return dummy.next
```

#### [2. 两数相加](https://leetcode-cn.com/problems/add-two-numbers/)

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
示例 2：

输入：l1 = [0], l2 = [0]
输出：[0]
示例 3：

输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]

```

思路：设立一个表示进位的变量 carried，建立一个新链表，把输入的两个链表从头往后同时处理，每两个相加，将结果加上 carried 后的值作为一个新节点到新链表后面，并更新 carried 值即可。

```python

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:

        dummy = current = ListNode()
        carry = 0

        while l1 or l2 or carry != 0:
            sum = carry
            if l1:
                sum += l1.val
                l1 = l1.next
            if l2:
                sum += l2.val
                l2 = l2.next

            if sum <= 9:
                tmp = ListNode(val = sum)
                carry = 0
            else:
                tmp = ListNode(val = sum % 10)
                carry = sum // 10
                
            current.next = tmp
            current = current.next

        return dummy.next 
```

