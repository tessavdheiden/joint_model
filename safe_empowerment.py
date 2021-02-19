import numpy as np
from itertools import product
from functools import reduce
import matplotlib.pyplot as plt
from collections import defaultdict

DIMS=(H,W)=(10,10)
N_STEP=5
WALLS=[["N", [1, i]] for i in range(3, W)]
WALLS+=[["W", [i, 1]] for i in range(3, H)]
DANGS=[[i, j] for (i,j) in product(range(6,8),range(7,9))]

actions={"N":np.array([1,0]), # UP
        "S":np.array([-1,0]), # DOWN
        "E":np.array([0,1]), # RIGHT
        "W":np.array([0,-1]),# LEFT
        "_" : np.array([0, 0])}
op={"N":"S","S":"N","W":"E","E":"W"}
adj=defaultdict(dict)
for i in range(H):
    for j in range(W):
        adj[i][j]=list(actions.keys())
for a,rc in WALLS:
    adj[rc[0]][rc[1]].remove(a)
    rc_=rc+actions[a]
    adj[rc_[0]][rc_[1]].remove(op[a])

gs1=lambda r,c:r>=0 and r<H and c>=0 and c<W
gs2=lambda r,c:[r,c] not in DANGS
ga=lambda a,r,c:a in adj[r][c]

s_to_rc=lambda s: np.array([s//W,s%W])
rc_to_s=lambda rc: rc[1]+W*rc[0]

class Env(object):
    n_a=len(actions)
    n_s=H*W
    w=[]
    def step(self,s,a):
        rc=s_to_rc(s)
        if not ga(a,rc[0],rc[1])or not gs2(rc[0],rc[1]): return s
        rc_=rc+actions[a]
        s_=rc_to_s(rc_)
        if not gs1(rc_[0],rc_[1]) or not gs2(rc_[0],rc_[1]):return s
        return s_

def compute(env):
    # T[s',a,s] is the probability of landing in s' given action a is taken in state s.
    n_s, n_a = env.n_s, env.n_a
    T = np.zeros([n_s,n_a,n_s])
    for s in range(n_s):
        for i, a in enumerate(actions.keys()):
            s_=env.step(s, a)
            T[s_, i, s] += 1.
    return T

def render(Tn):
    E = np.zeros(DIMS)
    for r in range(H):
        for c in range(W):
            s = rc_to_s([r, c])
            E[r, c] = np.log2(np.linalg.matrix_rank(Tn[:, :, s]))
    plt.pcolor(E)
    plt.colorbar()
    for a,w in WALLS:
        r,c=w
        if a=="N": plt.plot([c,c+1],[r+1,r+1],c='w')
        if a=="W": plt.plot([c,c],[r,r+1],c='w')
        if a=="E": plt.plot([c+1,c+1],[r,r+1],c='w')
    for d in DANGS:
        r,c=d
        plt.plot([c, c + 1], [r, r+1], c='r')
        plt.plot([c, c + 1], [r+1, r], c='r')
    plt.savefig("E.png")

env=Env()
T=compute(env)
ns_a = list(product(range(env.n_a), repeat=N_STEP))
Tn = np.zeros([env.n_s, env.n_a**N_STEP, env.n_s])
for i, an in enumerate(ns_a):
    Tn[:, i, :] = reduce(lambda x, y: np.dot(y, x), map(lambda a: T[:, a, :], an))
render(Tn)




