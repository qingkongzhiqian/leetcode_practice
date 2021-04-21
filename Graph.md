# Graph

[TOC]

#### 单源最短路径 Dijkstra's

```python
def dijkstra(G, source, destination):
    print('''Dijkstra's shortest path''')
    # Set the distance for the source node to zero 
    source.setDistance(0)
    # Put tuple pair into the priority queue
    unvisitedQueue = [(v.getDistance(), v) for v in G]
    heapq.heapify(unvisitedQueue)

    while len(unvisitedQueue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisitedQueue)
        current = uv[1]
        current.setVisited()

        # for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            newDist = current.getDistance() + current.getWeight(next)
            
            if newDist < next.getDistance():
                next.setDistance(newDist)
                next.setPrevious(current)
                print('Updated : current = %s next = %s newDist = %s' \
                        % (current.getVertexID(), next.getVertexID(), next.getDistance()))
            else:
                print('Not updated : current = %s next = %s newDist = %s' \
                        % (current.getVertexID(), next.getVertexID(), next.getDistance()))
        # Rebuild heap
        # 1. Pop every item
        while len(unvisitedQueue):
            heapq.heappop(unvisitedQueue)
        # 2. Put all vertices not visited into the queue
        unvisitedQueue = [(v.getDistance(), v) for v in G if not v.visited]
        heapq.heapify(unvisitedQueue)
```

#### 迷宫

由空地和墙组成的迷宫中有一个球。球可以向上下左右四个方向滚动，给定球的起始位置，目的地和迷宫，判断球能否到达目的地。

迷宫由一个0和1的二维数组表示。 1表示墙壁，0表示空地。你可以假定迷宫的边缘都是墙壁。起始位置和目的地的坐标通过行号和列号给出。

**示例 1:**

```python
输入 1: 迷宫由以下二维数组表示

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

输入 2: 起始位置坐标 (rowStart, colStart) = (0, 4)
输入 3: 目的地坐标 (rowDest, colDest) = (4, 4)

输出: true

解析: 一个可能的路径是 : 左 -> 下 -> 左 -> 下 -> 右 -> 下 -> 右。

```

```python
#DFS recursion
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:

        visited = [[False] * len(maze[0]) for i in range(len(maze))]
        return self.__hasPath(maze,start,destination,visited)

    def __hasPath(self,maze,start,destination,visited):

        if maze[start[0]][start[1]] == 1:
            return False
    
        if visited[start[0]][start[1]]:
            return False
        if start[0] == destination[0] and start[1] == destination[1]:
            return True
        
        visited[start[0]][start[1]] = True

        if (start[1] < len(maze[0]) - 1):
            r = (start[0], start[1] + 1)
            if (self.__hasPath(maze, r, destination, visited)):
                return True
        
        if (start[1] > 0):
            l = (start[0], start[1] - 1)
            if (self.__hasPath(maze, l, destination, visited)):
                return True
            
        if (start[0] > 0):
            u = (start[0] - 1, start[1])
            if (self.__hasPath(maze, u, destination, visited)):
                return True
            
        if (start[0] < len(maze[0]) - 1):
            d = (start[0] + 1, start[1])
            if (self.__hasPath(maze, d, destination, visited)):
                return True
                
        return False
```

```python
#DFS iterative
def dfsIterative(matrix, start, dest):
    visited = [[False] * len(matrix[0]) for i in range(len(matrix))]
    stack = []
    stack.append(start)
    visited[start[0]][start[1]] = True
    
    idxs = [[0,1], [0,-1], [-1,0], [1,0]]
    
    while len(stack) != 0:
        curr = stack.pop() # vertex
        if (curr[0] == dest[0] and curr[1] == dest[1]):
            return True

        for idx in idxs:
            x = curr[0] + idx[0]
            y = curr[1] + idx[1]
            
            if (x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0])):
                continue
            
            if (matrix[x][y] == 1):
                continue
                
            if (visited[x][y] == True):
                continue
            visited[x][y] = True
            stack.append((x, y))
            
    return False
```

```python
#BFS
from collections import deque
def bfs(matrix, start, dest):
    visited = [[False] * len(matrix[0]) for i in range(len(matrix))]
    queue = deque()
    queue.append(start)
    visited[start[0]][start[1]] = True
    
    idxs = [[0,1], [0,-1], [-1,0], [1,0]]
    
    while len(queue) != 0:
        curr = queue.popleft() # vertex
        if (curr[0] == dest[0] and curr[1] == dest[1]):
            return True

        for idx in idxs:
            x = curr[0] + idx[0]
            y = curr[1] + idx[1]
            
            if (x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0])):
                continue
            
            if (matrix[x][y] == 1):
                continue
                
            if (visited[x][y] == True):
                continue
            visited[x][y] = True
            queue.append((x, y))
            
    return False
```

#### [490. 迷宫](https://leetcode-cn.com/problems/the-maze/)

由空地和墙组成的迷宫中有一个球。球可以向上下左右四个方向滚动，但在遇到墙壁前不会停止滚动。当球停下时，可以选择下一个方向。

给定球的起始位置，目的地和迷宫，判断球能否在目的地停下。

迷宫由一个0和1的二维数组表示。 1表示墙壁，0表示空地。你可以假定迷宫的边缘都是墙壁。起始位置和目的地的坐标通过行号和列号给出。

```python
输入 1: 迷宫由以下二维数组表示

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

输入 2: 起始位置坐标 (rowStart, colStart) = (0, 4)
输入 3: 目的地坐标 (rowDest, colDest) = (4, 4)

输出: true

解析: 一个可能的路径是 : 左 -> 下 -> 左 -> 下 -> 右 -> 下 -> 右。


```

思路：与上一题不同的地方在于，其不是每一步都能直接开始找其neighbor，而是必须遇到墙之后才可以，则找neighbor的地方需要修改

```python
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:

        visited = [[False] * len(maze[0]) for i in range(len(maze))]
        return self.__hasPath(maze,start,destination,visited)

    def __hasPath(self,maze,start,destination,visited):

        if maze[start[0]][start[1]] == 1:
            return False
        if visited[start[0]][start[1]]:
            return False
        if start[0] == destination[0] and start[1] == destination[1]:
            return True
        
        visited[start[0]][start[1]] = True

        r = start[1] + 1
        l = start[1] - 1
        u = start[0] - 1
        d = start[0] + 1

        while r < len(maze[0]) and maze[start[0]][r] == 0:
            r += 1
        tmp = (start[0],r - 1)
        if self.__hasPath(maze,tmp,destination,visited):
            return True

        while l >= 0 and maze[start[0]][l] == 0:
            l -= 1
        tmp = (start[0],l + 1)
        if self.__hasPath(maze,tmp,destination,visited):
            return True

        while u >= 0 and maze[u][start[1]] == 0:
            u -= 1
        tmp = (u + 1,start[1])
        if self.__hasPath(maze,tmp,destination,visited):
            return True    

        while d < len(maze) and maze[d][start[1]] == 0:
            d += 1
        tmp = (d - 1,start[1])
        if self.__hasPath(maze,tmp,destination,visited):
            return True  

        return False        
```

```python
#DFS Iterative
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:

        if not maze:return False
        visited = [[False] * len(maze[0]) for i in range(len(maze))]
        stack = []
        visited[start[0]][start[1]] = True
        stack.append(start)
        idxs = [[1,0],[-1,0],[0,1],[0,-1]]

        while stack:
            current = stack.pop()
            if current[0] == destination[0] and current[1] == destination[1]:return True

            for dx,dy in idxs:
                x,y = current[0],current[1]
                while 0 <= x + dx < len(maze) and 0 <= y + dy < len(maze[0]) and (maze[x+dx][y+dy] == 0):
                    x += dx
                    y += dy

                if maze[x][y] == 1:continue
                if visited[x][y]:continue

                visited[x][y] = True
                stack.append((x,y))

        return False  
```

#### [505. 迷宫 II](https://leetcode-cn.com/problems/the-maze-ii/)

由空地和墙组成的迷宫中有一个球。球可以向上下左右四个方向滚动，但在遇到墙壁前不会停止滚动。当球停下时，可以选择下一个方向。

给定球的起始位置，目的地和迷宫，找出让球停在目的地的最短距离。距离的定义是球从起始位置（不包括）到目的地（包括）经过的空地个数。如果球无法停在目的地，返回 -1。

迷宫由一个0和1的二维数组表示。 1表示墙壁，0表示空地。你可以假定迷宫的边缘都是墙壁。起始位置和目的地的坐标通过行号和列号给出。

**示例 1:**

```python
输入 1: 迷宫由以下二维数组表示

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

输入 2: 起始位置坐标 (rowStart, colStart) = (0, 4)
输入 3: 目的地坐标 (rowDest, colDest) = (4, 4)

输出: 12

```

思路：Dijkstra算法，单源最短路径

```python
import heapq

class Solution(object):
    def shortestDistance(self, maze, start, destination):
        
        direction = [(0,1),(0,-1),(1,0),(-1,0)]
        m,n = len(maze),len(maze[0])
        start,destination = tuple(start),tuple(destination)
        stack = [(0,start)]
        visited = set()
        
        while stack:
            path,cur = heapq.heappop(stack)
            if cur == destination:
                return path
            visited.add(cur)
            for dx,dy in direction:
                x,y = cur
                length = 0
                while x+dx >= 0 and x+dx < m and y+dy >= 0 and y+dy < n and maze[x+dx][y+dy] == 0:
                    x,y = x+dx,y+dy
                    length += 1
                if (x,y) not in visited:
                    heapq.heappush(stack,(path+length,(x,y)))
        
        return -1

```

#### [499. 迷宫 III](https://leetcode-cn.com/problems/the-maze-iii/)

由空地和墙组成的迷宫中有一个球。球可以向上（u）下（d）左（l）右（r）四个方向滚动，但在遇到墙壁前不会停止滚动。当球停下时，可以选择下一个方向。迷宫中还有一个洞，当球运动经过洞时，就会掉进洞里。

给定球的起始位置，目的地和迷宫，找出让球以最短距离掉进洞里的路径。 距离的定义是球从起始位置（不包括）到目的地（包括）经过的空地个数。通过'u', 'd', 'l' 和 'r'输出球的移动方向。 由于可能有多条最短路径， 请输出字典序最小的路径。如果球无法进入洞，输出"impossible"。

迷宫由一个0和1的二维数组表示。 1表示墙壁，0表示空地。你可以假定迷宫的边缘都是墙壁。起始位置和目的地的坐标通过行号和列号给出。

```python
输入 1: 迷宫由以下二维数组表示

0 0 0 0 0
1 1 0 0 1
0 0 0 0 0
0 1 0 0 1
0 1 0 0 0

输入 2: 球的初始位置 (rowBall, colBall) = (4, 3)
输入 3: 洞的位置 (rowHole, colHole) = (0, 1)

输出: "lul"

解析: 有两条让球进洞的最短路径。
第一条路径是 左 -> 上 -> 左, 记为 "lul".
第二条路径是 上 -> 左, 记为 'ul'.
两条路径都具有最短距离6, 但'l' < 'u'，故第一条路径字典序更小。因此输出"lul"。

```

```python
class Solution:
    def findShortestWay(self, maze: List[List[int]], ball: List[int], hole: List[int]) -> str:
        import heapq
        dirs = {'u' : (-1, 0), 'r' : (0, 1), 'l' : (0, -1), 'd': (1, 0)}
        ball,hole = tuple(ball),tuple(hole)

        def neighbors(maze, node):
            for dir, vec in dirs.items():
                cur_node, dist = list(node), 0
                while 0 <= cur_node[0]+vec[0] < len(maze) and \
                    0 <= cur_node[1]+vec[1] < len(maze[0]) and \
                    not maze[cur_node[0]+vec[0]][cur_node[1]+vec[1]]:
                    cur_node[0] += vec[0]
                    cur_node[1] += vec[1]
                    dist += 1
                    if tuple(cur_node) == hole:
                        break
                yield tuple(cur_node), dir, dist

        
        heap = [(0, '', ball)]
        visited = set()
        
        while heap:
            dist, path, node = heapq.heappop(heap)
            if node in visited: continue
            if node == hole: return path
            visited.add(node)
            for neighbor, dir, neighbor_dist in neighbors(maze, node):
                heapq.heappush(heap, (dist+neighbor_dist, path+dir, neighbor))

        return "impossible"
```

#### [733. 图像渲染](https://leetcode-cn.com/problems/flood-fill/)

有一幅以二维整数数组表示的图画，每一个整数表示该图画的像素值大小，数值在 0 到 65535 之间。

给你一个坐标 (sr, sc) 表示图像渲染开始的像素值（行 ，列）和一个新的颜色值 newColor，让你重新上色这幅图像。

为了完成上色工作，从初始坐标开始，记录初始坐标的上下左右四个方向上像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应四个方向上像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为新的颜色值。

最后返回经过上色渲染后的图像。

```python
输入: 
image = [[1,1,1],[1,1,0],[1,0,1]]
sr = 1, sc = 1, newColor = 2
输出: [[2,2,2],[2,2,0],[2,0,1]]
解析: 
在图像的正中间，(坐标(sr,sc)=(1,1)),
在路径上所有符合条件的像素点的颜色都被更改成2。
注意，右下角的像素没有更改为2，
因为它不是在上下左右四个方向上与初始点相连的像素点。

```

```python
class Solution:
    def floodFill(self,image, sr, sc, newColor):
        rows, cols, orig_color = len(image), len(image[0]), image[sr][sc]

        def traverse(row, col):
            if (not (0 <= row < rows and 0 <= col < cols)) or image[row][col] != orig_color:
                return
            image[row][col] = newColor
            [traverse(row + x, col + y) for (x, y) in ((0, 1), (1, 0), (0, -1), (-1, 0))]
            
        if orig_color != newColor:
            traverse(sr, sc)
        return image
```

#### [547. 朋友圈](https://leetcode-cn.com/problems/friend-circles/)

班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，那么我们可以认为 A 也是 C 的朋友。所谓的朋友圈，是指所有朋友的集合。

给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。如果M[i][j] = 1，表示已知第 i 个和 j 个学生互为朋友关系，否则为不知道。你必须输出所有学生中的已知的朋友圈总数。

```python
输入：
[[1,1,0],
 [1,1,0],
 [0,0,1]]
输出：2 
解释：已知学生 0 和学生 1 互为朋友，他们在一个朋友圈。
第2个学生自己在一个朋友圈。所以返回 2 。

```

```python
输入：
[[1,1,0],
 [1,1,1],
 [0,1,1]]
输出：1
解释：已知学生 0 和学生 1 互为朋友，学生 1 和学生 2 互为朋友，所以学生 0 和学生 2 也是朋友，所以他们三个在一个朋友圈，返回 1 。

```

```python
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        circle = 0
        n = len(M)

        for i in range(n):
            if M[i][i] != 1:continue
            friends = [i]
            while friends:
                f = friends.pop()
                if M[f][f] == 0:continue
                M[f][f] = 0
                for j in range(n):
                    if M[f][j] == 1 and M[j][j] == 1:
                        friends.append(j)
            circle += 1
        return circle 
```

```python
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:

        def dfs(node):
            visited.add(node)
            for friend in range(len(M)):
                if M[node][friend] and friend not in visited:
                    dfs(friend)

        circle = 0
        visited = set()
        for node in range(len(M)):
            if node not in visited:
                dfs(node)
                circle += 1 

        return circle 
```

#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

```python
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1

输入：grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
输出：3

```

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:

        if not grid:return 0
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    self.__dfs(grid,i,j)
                    count += 1

        return count

    def __dfs(self,grid,i,j):                
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':return
        grid[i][j] = '#'
        self.__dfs(grid, i + 1, j)
        self.__dfs(grid, i - 1, j)
        self.__dfs(grid, i, j + 1)
        self.__dfs(grid, i, j - 1)
        

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == '1':  # 发现陆地
                    count += 1  # 结果加1
                    grid[row][col] = '0'  # 将其转为 ‘0’ 代表已经访问过
                    # 对发现的陆地进行扩张即执行 BFS，将与其相邻的陆地都标记为已访问
                    # 下面还是经典的 BFS 模板
                    land_positions = collections.deque()
                    land_positions.append([row, col])
                    while len(land_positions) > 0:
                        x, y = land_positions.popleft()
                        for new_x, new_y in [[x, y + 1], [x, y - 1], [x + 1, y], [x - 1, y]]:  # 进行四个方向的扩张
                            # 判断有效性
                            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and grid[new_x][new_y] == '1':
                                grid[new_x][new_y] = '0'  # 因为可由 BFS 访问到，代表同属一块岛，将其置 ‘0’ 代表已访问过
                                land_positions.append([new_x, new_y])
        return count
```

#### [695. 岛屿的最大面积](https://leetcode-cn.com/problems/max-area-of-island/)

给定一个包含了一些 0 和 1 的非空二维数组 grid 。

一个 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在水平或者竖直方向上相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

找到给定的二维数组中最大的岛屿面积。(如果没有岛屿，则返回面积为 0 。)

```python
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]

对于上面这个给定矩阵应返回 6。注意答案不应该是 11 ，因为岛屿只能包含水平或垂直的四个方向的 1 。

[[0,0,0,0,0,0,0,0]]
对于上面这个给定的矩阵, 返回 0。


```

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:

        m,n = len(grid),len(grid[0])

        def dfs(i,j):
            if  0 <= i < m and 0 <= j < n and grid[i][j]:
                grid[i][j] = 0
                return 1 + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i + 1, j) + dfs(i, j - 1)
            return 0

        result = 0
        for x in range(m):
            for y in range(n):
                if grid[x][y]:
                    result = max(result,dfs(x,y))
                    
        return result 
```

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:

        m,n = len(grid),len(grid[0])
        idxs = [[1,0],[-1,0],[0,1],[0,-1]]
        stack = []
        num = 0
        maxnum = 0

        for a in range(m):
            for b in range(n):
                if grid[a][b] == 1:
                    grid[a][b] = 0
                    stack.append((a,b))
                    num += 1

                    while stack:
                        x,y = stack.pop()
                        count = 1
                        for dx,dy in idxs:
                            if 0 <= x + dx < m and 0 <= y + dy < n and grid[x + dx][y + dy] == 1:
                                num += 1
                                grid[x+dx][y+dy] = 0
                                stack.append((x+dx,y+dy))
                    maxnum = max(maxnum,num)   
                    num = 0   
        return maxnum  
```

#### [690. 员工的重要性](https://leetcode-cn.com/problems/employee-importance/)

给定一个保存员工信息的数据结构，它包含了员工唯一的id，重要度 和 直系下属的id。

比如，员工1是员工2的领导，员工2是员工3的领导。他们相应的重要度为15, 10, 5。那么员工1的数据结构是[1, 15, [2]]，员工2的数据结构是[2, 10, [3]]，员工3的数据结构是[3, 5, []]。注意虽然员工3也是员工1的一个下属，但是由于并不是直系下属，因此没有体现在员工1的数据结构中。

现在输入一个公司的所有员工信息，以及单个员工id，返回这个员工和他所有下属的重要度之和。

```python
输入: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
输出: 11
解释:
员工1自身的重要度是5，他有两个直系下属2和3，而且2和3的重要度均为3。因此员工1的总重要度是 5 + 3 + 3 = 11。

```

```python
"""
# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""

class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        value = 0
        hash_table = {emp.id : emp for emp in employees}
        stack = [hash_table[id]]
        while stack:
            current = stack.pop()
            for sub in current.subordinates:
                stack.append(hash_table[sub])
            value += current.importance
        return value 
```

#### [785. 判断二分图](https://leetcode-cn.com/problems/is-graph-bipartite/)

给定一个无向图graph，当这个图为二分图时返回true。

如果我们能将一个图的节点集合分割成两个独立的子集A和B，并使图中的每一条边的两个节点一个来自A集合，一个来自B集合，我们就将这个图称为二分图。

graph将会以邻接表方式给出，graph[i]表示图中与节点i相连的所有节点。每个节点都是一个在0到graph.length-1之间的整数。这图中没有自环和平行边： graph[i] 中不存在i，并且graph[i]中没有重复的值。

https://leetcode-cn.com/problems/is-graph-bipartite/solution/pan-duan-er-fen-tu-by-leetcode-solution/

```python
示例 1:
输入: [[1,3], [0,2], [1,3], [0,2]]
输出: true
解释: 
无向图如下:
0----1
|    |
|    |
3----2
我们可以将节点分成两组: {0, 2} 和 {1, 3}。

```

```python

示例 2:
输入: [[1,2,3], [0,2], [0,1,3], [0,2]]
输出: false
解释: 
无向图如下:
0----1
| \  |
|  \ |
3----2
我们不能将节点分割成两个独立的子集。

```

```python
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        
        def dfs(pos):
            for i in graph[pos]:
                if i in color:
                    if color[i] == color[pos]:return False
                else:
                    color[i] = color[pos] ^ 1
                    if not dfs(i):return False
            return True            

        color = {}

        for i in range(len(graph)):
            if i not in color:color[i] = 0
            if not dfs(i):return False
        return True
```

#### [417. 太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)

给定一个 m x n 的非负整数矩阵来表示一片大陆上各个单元格的高度。“太平洋”处于大陆的左边界和上边界，而“大西洋”处于大陆的右边界和下边界。

规定水流只能按照上、下、左、右四个方向流动，且只能从高到低或者在同等高度上流动。

请找出那些水流既可以流动到“太平洋”，又能流动到“大西洋”的陆地单元的坐标。

提示：

输出坐标的顺序不重要
m 和 n 都小于150


示例：

```
给定下面的 5x5 矩阵:

  太平洋 ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * 大西洋

返回:

[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (上图中带括号的单元).

```

```python
from collections import deque

class Solution:
    def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
        if not matrix: return []
        m, n = len(matrix), len(matrix[0])
        def bfs(reachable_ocean):
            q = deque(reachable_ocean)
            while q:
                (i, j) = q.popleft()
                for (di, dj) in [(0,1), (0, -1), (1, 0), (-1, 0)]:
                    if 0 <= di+i < m and 0 <= dj+j < n and (di+i, dj+j) not in reachable_ocean \
                        and matrix[di+i][dj+j] >= matrix[i][j]:
                        q.append( (di+i,dj+j) )
                        reachable_ocean.add( (di+i, dj+j) )
            return reachable_ocean         
        pacific  =set ( [ (i, 0) for i in range(m)]   + [(0, j) for j  in range(1, n)]) 
        atlantic =set ( [ (i, n-1) for i in range(m)] + [(m-1, j) for j in range(n-1)]) 
        return list( bfs(pacific) & bfs(atlantic) )
```

#### [329. 矩阵中的最长递增路径](https://leetcode-cn.com/problems/longest-increasing-path-in-a-matrix/)

给定一个整数矩阵，找出最长递增路径的长度。

对于每个单元格，你可以往上，下，左，右四个方向移动。 你不能在对角线方向上移动或移动到边界外（即不允许环绕）。

**示例 1:**

```
输入: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
输出: 4 
解释: 最长递增路径为 [1, 2, 6, 9]。

```

**示例 2:**

```
输入: nums = 
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
] 
输出: 4 
解释: 最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。

```

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:


        def dfs(i,j,matrix,cache,m,n):
            if cache[i][j] != -1:
                return cache[i][j]
            res = 1
            for direction in directions:
                x,y = i + direction[0],j+direction[1]
                if x < 0 or x >= m or y < 0 or y >= n or matrix[x][y] <= matrix[i][j]:continue
                length = 1 + dfs(x,y,matrix,cache,m,n)
                res = max(length,res)

            cache[i][j] = res
            return res

        if not matrix:return 0
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        m = len(matrix)
        n = len(matrix[0])

        cache = [[-1 for _ in range(n)] for _ in range(m)]
        res = 0
        for i in range(m):
            for j in range(n):
                cur_len = dfs(i,j,matrix,cache,m,n)
                res = max(res,cur_len)
        return res
    
```

#### [542. 01 矩阵](https://leetcode-cn.com/problems/01-matrix/)

给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

```
示例 1：

输入：
[[0,0,0],
 [0,1,0],
 [0,0,0]]

输出：
[[0,0,0],
 [0,1,0],
 [0,0,0]]
示例 2：

输入：
[[0,0,0],
 [0,1,0],
 [1,1,1]]

输出：
[[0,0,0],
 [0,1,0],
 [1,2,1]]
```

```python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:

        def DP(i, j, m, n, dp):
            if i > 0: dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
            if j > 0: dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)
            if i < m - 1: dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1)
            if j < n - 1: dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1)
            
        if not matrix: return [[]]
        m, n = len(matrix), len(matrix[0])
        dp = [[0x7fffffff if matrix[i][j] != 0 else 0 for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                DP(i, j, m, n, dp)

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                DP(i, j, m, n, dp)

        return dp
```

#### [79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

给定一个二维网格和一个单词，找出该单词是否存在于网格中。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```python
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

给定 word = "ABCCED", 返回 true
给定 word = "SEE", 返回 true
给定 word = "ABCB", 返回 false

```

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def dfs(i,j,index):
            
            if board[i][j] != word[index]:return False
            if index == len(word) - 1:return True
            visited.add((i,j))
            result = False

            for dx,dy in directions:
                x,y = i + dx,j + dy
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and (x,y) not in visited:
                    if dfs(x,y,index + 1):
                        result = True
                        break

            visited.remove((i,j))
            return result            
                    
        m,n = len(board),len(board[0])
        visited = set()
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    if dfs(i,j,0):
                        return True
        return False
```

