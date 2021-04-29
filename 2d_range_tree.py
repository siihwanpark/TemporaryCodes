class node():
    def __init__(self, val, left, right, region, ytree=None):
        self.val = val
        self.left = left
        self.right = right
        self.region = region
        self.ytree = ytree
    
    #def __repr__(self):
    #    return "val : {}, dir : {}, left : {}, right : {}".format(self.val, self.dir, self.left, self.right) 

class leaf():
    def __init__(self, point, ytree):
        self.point = point
        self.ytree = ytree

    #def __repr__(self):
    #    return "point : {}".format(self.point)

import copy

def BuildXTree(S, region = [-1e10, 1e10]):
    if len(S) == 1:
        return leaf(S[0], [S[0][1]])
    
    S = sorted(S, key = lambda x : x[0])

    # median of X coordinates of all points in S
    idx_mid = len(S)//2 - 1
    x = S[idx_mid][0] 

    left_region = copy.deepcopy(region)
    if left_region[1] > x:
        left_region[1] = x

    right_region = copy.deepcopy(region)
    if right_region[0] < x:
        right_region[0] = x

    root = node(x, BuildXTree(S[:idx_mid+1], left_region), BuildXTree(S[idx_mid+1:], right_region), region)
    root.ytree = MergeYTree(root.left.ytree, root.right.ytree)

    return root

def MergeYTree(t1, t2):
    """
    t1 : first Y tree which contains y-coordinate of points as a list
    t2 : second Y tree which contains y-coordinate of points as a list
    """

    return sorted(t1 + t2, key=lambda x : x)

def query(t, r):
    if isinstance(t, leaf):
        if t.point[0] >= r[0][0] and t.point[0] <= r[0][1] \
            and t.point[1] >= r[1][0] and t.point[1] <= r[1][1]:
            return 1
        else:
            return 0
    else:
        if isInside(t.region, r[0]):
            return query_Y(t.ytree, r[1])
        
        elif isIntersect(t.region, r[0]):
            return query(t.left, r) + query(t.right, r)
        
        else:
            return 0

def query_Y(t, r):
    """
    t : y-tree; [y1, y2, y3, ..., yN] and this list is sorted
    r : region; [a, b] with a <= b

    return the # of yi's that contained in r
    (using binary search)
    """
    
    import bisect
    leftmost_idx = bisect.bisect_left(t, r[0])
    rightmost_idx = bisect.bisect_right(t, r[1])

    return rightmost_idx - leftmost_idx

def isInside(r1, r2):
    """
    r1 : [a,b] , a <= b
    r2 : [c,d] , c <= d

    return true if r1 is contained in r2
    """

    return r1[0] >= r2[0] and r1[1] <= r2[1]

def isIntersect(r1, r2):
    """
    r1 : [a,b] , a <= b
    r2 : [c,d] , c <= d

    return true if r1 intersects with r2
    """

    return r2[0] <= r1[1] or r2[1] <= r1[0]

import bisect

if __name__ == "__main__":
    # sample 1
    #points = [[-2,1], [0,1], [2,1], [-2,-1], [0,-1], [2,-1], [4,1]]
    points = [[0, 0], [0, 0], [0, 0]]

    root = BuildXTree(points)
    sol = query(root, [[0, 0], [0, 0]])
    print(sol)

	#sol = Solution(points)
	#print(sol.query([[-1,2], [0,2]]))       # 4
	#print(sol.query([[1,4], [0,-1]]))       # 3
