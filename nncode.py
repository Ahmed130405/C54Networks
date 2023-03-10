import networkx as nx
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities
import numpy as np
import matplotlib.pyplot as plt
import math as m
import random

p = 0.1
n=100
beta = 0.016 # birth rate
delta = 0.17 # death rate
epsilon = 0.0032
t = 0
T = 200

def roots():
    return np.roots([-1*epsilon*(2*n-2), epsilon*(2*n-2), -1*(beta+delta), beta])

def density_return(mat): # input matrix, output edge density
    ans, ans1, ans2, ans3 = 0, 0, 0, 0
    nn=int(len(mat)/2)
    for i in range(1,nn):
        for j in range(i):
            ans += mat[i,j]
            ans1 += mat[i,j]
    for i in range(nn,nn+nn):
        for j in range(nn):
            ans += mat[i,j]
            ans2 += mat[i,j]
    for i in range(nn,nn+nn):
        for j in range(nn,i):
            ans += mat[i,j]
            ans3 += mat[i,j]
    return ans/(0.5*(nn+nn)*(nn+nn-1)), ans1/(0.5*nn*(nn-1)), ans2/(nn*nn), ans3/(0.5*nn*(nn-1))

def new_adj(adj,beta=beta,delta=delta,epsilon=epsilon): #input is Ak, beta, delta, epsilon. Output it Ak+1. Each entry is given by randomising parameters
    adj_squared = np.matmul(adj,adj)
    nn=int(len(adj)/2)
    new_adj = np.zeros((2*nn, 2*nn))
    for i in range(1,nn+nn):
        for j in range(i):
            rand1 = random.uniform(0,1)
            rand2 = random.uniform(0,1)
            if (rand1<delta) and (rand2<beta+epsilon*adj_squared[i,j]):
                new_adj[i,j], new_adj[j,i] = 1-adj[i,j], 1-adj[j,i]
            elif (rand1>=delta) and (rand2<beta+epsilon*adj_squared[i,j]):
                new_adj[i,j], new_adj[j,i] = 1, 1
            elif (rand1<delta) and (rand2>=beta+epsilon*adj_squared[i,j]):
                new_adj[i,j], new_adj[j,i] = 0, 0
            else:
                new_adj[i,j], new_adj[j,i] = adj[i,j], adj[j,i]
    return new_adj

def new_p(p,beta=beta,delta=delta,epsilon=epsilon):
    return p*(1-delta) + (1-p)*(beta+(n+n-2)*epsilon*(p**2))


def erdos_albert(nn, pp):
    ee=nx.erdos_renyi_graph(nn,pp)
    aa=nx.barabasi_albert_graph(nn, int(pp*(nn-1)/2))
    matrix=np.zeros((2*nn,2*nn))
    matrix[:nn,:nn]=nx.to_numpy_array(ee)
    matrix[nn:,nn:]=nx.to_numpy_array(aa)
    return matrix

def erdos_erdos(nn, pp):
    m1=nx.erdos_renyi_graph(nn, pp)
    m2=nx.erdos_renyi_graph(nn, pp)
    matrix=np.zeros((2*nn,2*nn))
    matrix[:nn,:nn]=nx.to_numpy_array(m1)
    matrix[nn:,nn:]=nx.to_numpy_array(m2)
    return matrix

def albert_albert(nn, pp):
    m1=nx.barabasi_albert_graph(nn, int(pp*(nn-1)/2))
    m2=nx.barabasi_albert_graph(nn, int(pp*(nn-1)/2))
    matrix=np.zeros((2*nn,2*nn))
    matrix[:nn,:nn]=nx.to_numpy_array(m1)
    matrix[nn:,nn:]=nx.to_numpy_array(m2)
    return matrix
    
stoch_density = []
mfa_density = []

'''S1 = nx.erdos_renyi_graph(n,2*p)
S2 = nx.erdos_renyi_graph(n,2*p)
adj = np.zeros((2*n,2*n))
adj[:n,:n] = nx.to_numpy_array(S1)
adj[n:,n:] = nx.to_numpy_array(S2)'''


for i in range(5):
    stoch_density = []
    t=0
    mm=albert_albert(101, 0.5)
    while t<T:
        stoch_density.append(density_return(mm)[0])
        mfa_density.append(p)
        mm = new_adj(mm)
        p = new_p(p)
        t += 1
    plt.plot([x for x in range(T)],stoch_density, 'k')


'''for i in range(5):
    stoch_density = []
    t=0
    mm=erdos_erdos(100, 0.5)
    while t<T:
        stoch_density.append(density_return(mm)[0])
        mfa_density.append(p)
        mm = new_adj(mm)
        p = new_p(p)
        t += 1
    plt.plot([x for x in range(T)],stoch_density, 'k')'''

#plt.plot([x for x in range(T)],stoch_density,label="stochastic")
#plt.plot([x for x in range(T)],mfa_density,label="mean-field")
plt.xlabel("Time")
plt.ylabel("Edge density")
#plt.legend()
plt.show()
