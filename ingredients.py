# %%
import jax.numpy as jnp
import networkx as nx
import numpy as np
import pandas as pd
from tinygp import kernels

# %% spatial adjacency matrix
edges = pd.read_csv("data/hungary_county_edges.csv")
G = nx.from_pandas_edgelist(edges, source="name_1", target="name_2")
G.remove_edges_from(nx.selfloop_edges(G))
A = nx.to_numpy_array(G)

# %% continuous time series
df = pd.read_csv("data/hungary_chickenpox.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

t = jnp.array(df.Date.index.values, dtype=float)

# %%
Y = jnp.array(df.iloc[:, 1:])  # N1 x N2 column major
y = jnp.ravel(Y, order="F")

# %% time covariance
k_t = kernels.Matern52(10)
K_t = k_t(t, t)

# %% spatial graph covariance
# k_s = 10 * kernels.Matern52(1)

# distance 1 for neighbours
# distance 0 for diagonal
# distance Inf (* 0) for not neighbours
# could try geodesic distance

r = jnp.array(A) / 1  # distance / lengthscale
arg = np.sqrt(5) * r
K_s = (1 + arg + jnp.square(arg) / 3) * jnp.exp(-arg)
K_s = A * K_s  # set to zero
K_s += jnp.eye(K_s.shape[-1])  # perfectly correlated diag
K_s *= 10  # scale

# %%
K_st = jnp.kron(K_s, K_t)

# %%
jnp.linalg.solve(jnp.kron(K_s, K_t), jnp.eye(K_st.shape[0])) @ y
jnp.ravel(
    jnp.linalg.solve(K_t, jnp.eye(K_t.shape[0]))
    @ Y
    @ jnp.linalg.solve(K_s, jnp.eye(K_s.shape[0])).T,
    order="F",
)  # flaxman 2015 eq 8 does not work

# %%
jnp.kron(jnp.linalg.inv(K_s), jnp.linalg.inv(K_t))

# %%
K1 = np.random.rand(3, 3)  # N1 x N1
K2 = np.random.rand(2, 2)  # N2 x N1

Y = np.random.rand(2, 3)  # N2 x N1
y = np.ravel(Y, order="F")  # N1*N2 x 1

# %%
np.kron(K1, K2) @ y

# %%
np.ravel(K2 @ Y @ K1.T, order="F")
