import networkx as nx
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities
import numpy as np
import matplotlib.pyplot as plt
import math as m
import random

p = 0.1
n1 = 40
n2 = 40
gamma = 0.02
delta = 0.02
epsilon = 0.001
t = 0
T = 200

def density_return(mat): # input matrix, output edge density
    ans, ans1, ans2, ans3 = 0, 0, 0, 0
    for i in range(1,n1):
        for j in range(i):
            ans += mat[i,j]
            ans1 += mat[i,j]
    for i in range(n1,n1+n2):
        for j in range(n1):
            ans += mat[i,j]
            ans2 += mat[i,j]
    for i in range(n1,n1+n2):
        for j in range(n1,i):
            ans += mat[i,j]
            ans3 += mat[i,j]
    return ans/(0.5*(n1+n2)*(n1+n2-1)), ans1/(0.5*n1*(n1-1)), ans2/(n1*n2), ans3/(0.5*n2*(n2-1))

def new_adj(adj,gamma=gamma,delta=delta,epsilon=epsilon): #input is Ak, gamma, delta, epsilon. Output it Ak+1. Each entry is given by randomising parameters
    adj_squared = np.matmul(adj,adj)
    new_adj = np.zeros((n1+n2,n1+n2))
    for i in range(1,n1+n2):
        for j in range(i):
            rand1 = random.uniform(0,1)
            rand2 = random.uniform(0,1)
            if (rand1<gamma) and (rand2<delta+epsilon*adj_squared[i,j]):
                new_adj[i,j], new_adj[j,i] = 1-adj[i,j], 1-adj[j,i]
            elif (rand1>=gamma) and (rand2<delta+epsilon*adj_squared[i,j]):
                new_adj[i,j], new_adj[j,i] = 1, 1
            elif (rand1<gamma) and (rand2>=delta+epsilon*adj_squared[i,j]):
                new_adj[i,j], new_adj[j,i] = 0, 0
            else:
                new_adj[i,j], new_adj[j,i] = adj[i,j], adj[j,i]
    return new_adj

def new_p(p,gamma=gamma,delta=delta,epsilon=epsilon):
    return p*(1-gamma) + (1-p)*(delta+(n1+n2-2)*epsilon*(p**2))

stoch_density = []
mfa_density = []

S1 = nx.erdos_renyi_graph(n1,2*p)
S2 = nx.erdos_renyi_graph(n2,2*p)
adj = np.zeros((n1+n2,n1+n2))
adj[:n1,:n1] = nx.to_numpy_array(S1)
adj[n1:,n1:] = nx.to_numpy_array(S2)

while t<T:
    stoch_density.append(density_return(adj)[0])
    mfa_density.append(p)
    adj = new_adj(adj)
    p = new_p(p)
    t += 1

plt.plot([x for x in range(T)],stoch_density,label="stochastic")
plt.plot([x for x in range(T)],mfa_density,label="mean-field")
plt.xlabel("Time")
plt.ylabel("Edge density")
plt.legend()
plt.show()
