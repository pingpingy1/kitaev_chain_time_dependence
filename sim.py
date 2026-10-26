import numpy as np

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.integrate import solve_ivp


def annihilation_operator(N, j):
    """Fermionic annihilation operator c_j, with sites indexed from 0."""
    D = 1 << N
    c = lil_matrix((D, D), dtype=np.complex128)

    for state in range(D):
        if (state >> j) & 1:
            target = state ^ (1 << j)

            # Jordan-Wigner sign from occupied sites to the left.
            lower_bits = state & ((1 << j) - 1)
            sign = (-1) ** lower_bits.bit_count()

            c[target, state] = sign

    return c.tocsr()


def build_kitaev_chain(N, mu, t_hop, delta=None):
    """
    Build the static Kitaev Hamiltonian and total number operator.

    Uses hbar = 1 and open boundary conditions.
    """
    if delta is None:
        delta = t_hop

    if N < 2:
        raise ValueError("N must be at least 2.")

    D = 1 << N
    identity = csr_matrix(np.eye(D, dtype=np.complex128))

    c = [annihilation_operator(N, j) for j in range(N)]
    cd = [op.getH() for op in c]

    number_op = csr_matrix((D, D), dtype=np.complex128)
    H = csr_matrix((D, D), dtype=np.complex128)

    # On-site chemical-potential term.
    for j in range(N):
        nj = cd[j] @ c[j]
        number_op += nj
        H += -mu * (nj - 0.5 * identity)

    # Hopping and superconducting pairing.
    for j in range(N - 1):
        H += -t_hop * (
            cd[j] @ c[j + 1] + cd[j + 1] @ c[j]
        )

        H += delta * (
            c[j] @ c[j + 1]
            + cd[j + 1] @ cd[j]
        )

    return H.tocsr(), number_op.tocsr()


def simulate_kitaev(
    mu,
    t_hop,
    N,
    A,
    omega,
    T,
    num_times=401,
    parity="even",
    delta=None,
    rtol=1e-9,
    atol=1e-11,
):
    """
    Simulate a Kitaev chain with
        mu(t) = mu + A*cos(omega*t).

    Returns:
        times: sample times
        leakage: 1 - fidelity with the instantaneous ground state
                  in the chosen fermion-parity sector.

    Units: hbar = 1.
    """
    if A < 0:
        raise ValueError("A must be nonnegative.")
    if omega < 0:
        raise ValueError("omega must be nonnegative.")
    if T <= 0 or num_times < 2:
        raise ValueError("T must be positive and num_times >= 2.")

    if delta is None:
        delta = t_hop

    # Build the full Hamiltonian.
    H0, number_op = build_kitaev_chain(
        N, mu, t_hop, delta
    )

    # Restrict to a fixed fermion-parity sector.
    D = 1 << N

    if parity == "even":
        basis = np.array([
            s for s in range(D)
            if s.bit_count() % 2 == 0
        ])
    elif parity == "odd":
        basis = np.array([
            s for s in range(D)
            if s.bit_count() % 2 == 1
        ])
    else:
        raise ValueError("parity must be 'even' or 'odd'.")

    H0 = H0[basis, :][:, basis].tocsr()
    number_op = number_op[basis, :][:, basis].tocsr()

    # H(t) = H0 - A*cos(omega*t)*number_op.
    def H_of_time(time):
        return H0 - A * np.cos(omega * time) * number_op

    # Initialize in the instantaneous ground state at t = 0.
    _, initial_vectors = eigsh(
        H_of_time(0.0), k=1, which="SA"
    )
    psi0 = initial_vectors[:, 0]
    psi0 /= np.linalg.norm(psi0)

    # Time-dependent Schrödinger equation, hbar = 1:
    # d|psi>/dt = -i H(t)|psi>.
    def rhs(time, psi):
        return -1j * (H_of_time(time) @ psi)

    times = np.linspace(0.0, T, num_times)

    solution = solve_ivp(
        rhs,
        (0.0, T),
        psi0,
        t_eval=times,
        rtol=rtol,
        atol=atol,
        method="DOP853",
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    leakage = np.empty(len(times))

    for k, time in enumerate(times):
        psi = solution.y[:, k]
        psi /= np.linalg.norm(psi)

        # Instantaneous ground state in the same parity sector.
        _, vectors = eigsh(
            H_of_time(time), k=1, which="SA"
        )
        ground = vectors[:, 0]

        fidelity = abs(np.vdot(ground, psi)) ** 2
        leakage[k] = np.clip(1.0 - fidelity, 0.0, 1.0)

    return times, leakage
