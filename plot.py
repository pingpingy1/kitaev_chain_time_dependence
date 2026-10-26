import matplotlib.pyplot as plt
from sim import *

mu = 0.5
t_hop = 1.0
N = 8
A = 0.1
omega = 0.8
T = 100.0

times, leakage = simulate_kitaev(
    mu=mu,
    t_hop=t_hop,
    N=N,
    A=A,
    omega=omega,
    T=T,
    num_times=501,
    parity="even",
)

plt.plot(times, leakage)
plt.xlabel("Time")
plt.ylabel("Instantaneous ground-state leakage")
plt.title("Driven Kitaev chain")
plt.show()
