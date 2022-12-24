import networkx as nx
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities
import numpy as np
import matplotlib.pyplot as plt
import math
import random



# adj_mat = np.zeros((n1 + n2, n1 + n2))
# adj_mat[n1:, n1:] = 0.5*nx.to_numpy_array(nx.erdos_renyi_graph(n1, 1))
# adj_mat[:n1, :n1] = 0.5*nx.to_numpy_array(nx.erdos_renyi_graph(n2, 1))
# for i in range(10):
#     adj_mat = (1-0.5)*adj_mat + np.multiply(
#         (nx.to_numpy_array(nx.erdos_renyi_graph(n1+n2, 1)) - adj_mat),
#         0.5*nx.to_numpy_array(nx.erdos_renyi_graph(n1+n2, 1)) + 0.01*np.matmul(adj_mat, adj_mat))
#     print(adj_mat)


def new_f_calc(adj_mat):
    out_f = 0.0
    all_f = 0.0
    for a in range(n1):
        for c in range(a, n1+n2):
            if adj_mat[a, c] == 1:
                all_f += 1
                if c >= n1:
                    out_f += 1

    if all_f == 0:
        return 0
    return out_f/all_f

def density_return(mat):
    ans1, ans2, ans3 = 0, 0, 0
    for i1 in range(1, n1):
        for j1 in range(i1):
            ans1 += mat[i1, j1]
    for k1 in range(n1, n1+n2):
        for l1 in range(n1):
            ans2 += mat[k1, l1]
    for s in range(n1, n1+n2):
        for r in range(n1, s):
            ans3 += mat[s, r]
    return ans1/(0.5*n1*(n1-1)), ans2/(n1*n2), ans3/(0.5*n2*(n2-1))
# d = 0.26
# b = 0.03
# e = 0.00455
# f = 0.00311


p = 0.8
n1 = 100
n2 = 100
S1 = nx.erdos_renyi_graph(n1, p)
S2 = nx.erdos_renyi_graph(n2, p)

adj_mat = np.zeros((n1 + n2, n1 + n2))
adj_mat[:n1, :n1] = nx.to_numpy_array(S1)
adj_mat[n1:, n1:] = nx.to_numpy_array(S2)

# d = 0.11
# b = 0.007
# e = 0.0022
d = 0.647
b = 0.156
e = 0.00088
f = 0.00132


t = 0
T = 100
friendship_props = []
#edge_densities = {1: [], 2: [], 3: []}

while t < T:
    # graph = nx.from_numpy_matrix(adj_mat)
    # friendship_props.append(new_f_calc(adj_mat))
    # edge_densities.append(nx.density(graph))
    t += 1
    print(t)
    adj_mat_squared = np.matmul(adj_mat, adj_mat)
    ans1, ans2, ans3 = density_return(adj_mat)
    edge_densities[1].append(ans1)
    edge_densities[2].append(ans2)
    edge_densities[3].append(ans3)
    for i in range(1, n1):
        for j in range(i):
            rand_float = random.random()
            if adj_mat[i, j] == 0:
                if rand_float < b1 + e1*adj_mat_squared[i, j]:
                    adj_mat[i, j], adj_mat[j, i] = 1, 1
            else:
                if rand_float < d1 - f1*adj_mat_squared[i, j]:
                    adj_mat[i, j], adj_mat[j, i] = 0, 0
    for k in range(n1, n1+n2):
        for l in range(n1):
            rand_float = random.random()
            if adj_mat[k, l] == 0:
                if rand_float < b2 + e2*adj_mat_squared[k, l]:
                    adj_mat[k, l], adj_mat[l, k] = 1, 1
            else:
                if rand_float < d2 - f2*adj_mat_squared[k, l]:
                    adj_mat[k, l], adj_mat[l, k] = 0, 0
    for m in range(n1, n1+n2):
        for n in range(n1, m):
            rand_float = random.random()
            if adj_mat[m, n] == 0:
                if rand_float < b3 + e3*adj_mat_squared[m, n]:
                    adj_mat[m, n], adj_mat[n, m] = 1, 1
            else:
                if rand_float < d3 - f3*adj_mat_squared[m, n]:
                    adj_mat[m, n], adj_mat[n, m] = 0, 0

detm1 = [p]
for x in range(1, T):
    inp = detm1[x-1]
    val = (1-d1 + f1*(n1 + n2 - 2)*math.pow(inp, 2))*inp + (1-inp)*(b1 + e1*(n1 + n2 - 2)*math.pow(inp, 2))
    detm1.append(val)
     #this needs changing in the advanced case
detm2 = [0]
for x in range(1, T):
    inp = detm2[x-1]
    val = (1-d2 + f2*(n1 + n2 - 2)*math.pow(inp, 2))*inp + (1-inp)*(b2 + e2*(n1 + n2 - 2)*math.pow(inp, 2))
    detm2.append(val)
detm3 = [p]
for x in range(1, T):
    inp = detm3[x-1]
    val = (1-d3 + f3*(n1 + n2 - 2)*math.pow(inp, 2))*inp + (1-inp)*(b3 + e3*(n1 + n2 - 2)*math.pow(inp, 2))
    detm3.append(val)



G = nx.from_numpy_array(adj_mat)
GG = greedy_modularity_communities(G)
print(f"{len(GG)} communities in City 1:")
for community in GG:
    print(sorted(community))


plt.plot([x for x in range(T)], detm1, label="mean-field 1-1")
plt.plot([x for x in range(T)], detm2, label="mean-field 1-2")
plt.plot([x for x in range(T)], detm3, label="mean-field 2-2")
plt.plot([x for x in range(T)], edge_densities[1], label="School 1-1 stochastic")
plt.plot([x for x in range(T)], edge_densities[2], label="School 1-2 stochastic")
plt.plot([x for x in range(T)], edge_densities[3], label="School 2-2 stochastic")


#plt.plot([x for x in range(T)], friendship_props, label="friends")
plt.xlabel("Time")
plt.ylabel("Edge density")
plt.legend()
plt.show()







