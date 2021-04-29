class node():
    def __init__(self, val, dir, left, right, region):
        self.val = val
        self.dir = dir
        self.left = left
        self.right = right
        self.region = region
    
    def __repr__(self):
        return "val : {}, dir : {}, left : {}, right : {}".format(self.val, self.dir, self.left, self.right) 

class leaf():
    def __init__(self, point):
        self.point = point

    def __repr__(self):
        return "point : {}".format(self.point)

import copy

def BuildTree(S, d, region=[[-1e10, 1e10],[-1e10, 1e10]]):
    #print("region : {}".format(region))
    if len(S) == 1:
        return leaf(S[0])
    S = sorted(S, key = lambda x : x[d])
    #print(S)
    idx_mid = len(S)//2 - 1
    x = S[idx_mid]

    left_region = copy.deepcopy(region)
    if left_region[d][1] > x[d]:
        left_region[d][1] = x[d]
    
    right_region = copy.deepcopy(region)
    if right_region[d][0] < x[d]:
        right_region[d][0] = x[d]

    return node(x[d], d, BuildTree(S[:idx_mid+1], 1-d, left_region), BuildTree(S[idx_mid+1:], 1-d, right_region), region)

def query(t, r):
    if isinstance(t, leaf):
        if t.point[0] >= r[0][0] and t.point[0] <= r[0][1] \
            and t.point[1] >= r[1][0] and t.point[1] <= r[1][1]:
            return [t.point]
        else:
            return []
    else:
        if isInside(t.region, r):
            return reportTree(t)
        
        elif isIntersect(t.region, r):
            return query(t.left, r) + query(t.right, r)
        
        else:
            return []

def reportTree(t):
    if isinstance(t, leaf):
        return [t.point]
    else:
        return reportTree(t.left) + reportTree(t.right)

def isInside(r1, r2):
    return r1[0][0] >= r2[0][0] and r1[0][1] <= r2[0][1] and r1[1][0] >= r2[1][0] and r1[1][1] <= r2[1][1]

def isIntersect(r1, r2):
    r1_top_right = [r1[0][1], r1[1][1]]
    r1_bot_left = [r1[0][0], r1[1][0]]
    r2_top_right = [r2[0][1], r2[1][1]]
    r2_bot_left = [r2[0][0], r2[1][0]]

    if (r1_bot_left[0] > r2_top_right[0]) or (r1_top_right[0] < r2_bot_left[0]) \
       or (r1_top_right[1] < r2_bot_left[1]) or (r1_bot_left[1] > r2_top_right[1]):
       return False
    else:
        return True

if __name__ == "__main__":
    # sample 1
    #points = [[-2,1], [0,1], [2,1], [-2,-1], [0,-1], [2,-1], [4,1]]
    points = [[0, 0], [0, 0], [0, 0]]
    root = BuildTree(points, 0)
    sol = query(root, [[0, 0], [0, 0]])
    print(len(sol))

	#sol = Solution(points)
	#print(sol.query([[-1,2], [0,2]]))       # 4
	#print(sol.query([[1,4], [0,-1]]))       # 3
