"""
Tugas Kelompok Berbasis Kasus 02 - N-Queen (8x8)
Perbandingan DFS-Backtracking (MRV + Forward Checking + LCV)
dengan Local Search (Simulated Annealing & Hill Climbing random restart).

Representasi state: state[col] = baris ratu pada kolom col.
"""
import math
import random
import time

N = 8
# Initial state dari soal: (6,0) (5,1) (3,2) (3,3) (1,4) (3,5) (7,6) (3,7)
INITIAL = [6, 5, 3, 3, 1, 3, 7, 3]


def attacks(c1, r1, c2, r2):
    return r1 == r2 or abs(r1 - r2) == abs(c1 - c2)


def count_conflicts(state):
    """h(n) = jumlah pasangan ratu yang saling menyerang."""
    h = 0
    for i in range(N):
        for j in range(i + 1, N):
            if attacks(i, state[i], j, state[j]):
                h += 1
    return h


def moved(state):
    return sum(1 for c in range(N) if state[c] != INITIAL[c])


def print_board(state, title):
    print(title)
    print("    " + " ".join(str(c) for c in range(N)))
    for r in range(N):
        row = ["Q" if state[c] == r else "." for c in range(N)]
        print(f" {r}  " + " ".join(row))
    print(f"    state = {state}, h = {count_conflicts(state)}\n")


# ---------------------------------------------------------------
# 1. DFS-Backtracking dengan MRV + Forward Checking + LCV
# ---------------------------------------------------------------
def backtracking():
    stats = {"nodes": 0, "backtracks": 0}
    domains = {c: list(range(N)) for c in range(N)}

    def lcv_order(col, assign, doms):
        # nilai awal ratu dicoba lebih dulu, sisanya diurutkan LCV
        def eliminated(r):
            return sum(1 for c in doms if c not in assign and c != col
                       for r2 in doms[c] if attacks(col, r, c, r2))
        vals = sorted(doms[col], key=lambda r: (r != INITIAL[col], eliminated(r)))
        return vals

    def solve(assign, doms):
        if len(assign) == N:
            return dict(assign)
        # MRV: kolom dengan domain tersisa paling sedikit
        col = min((c for c in range(N) if c not in assign),
                  key=lambda c: (len(doms[c]), c))
        for r in lcv_order(col, assign, doms):
            stats["nodes"] += 1
            # Forward checking
            new_doms = {}
            ok = True
            for c in range(N):
                if c in assign or c == col:
                    new_doms[c] = doms[c] if c != col else [r]
                    continue
                new_doms[c] = [r2 for r2 in doms[c] if not attacks(col, r, c, r2)]
                if not new_doms[c]:
                    ok = False
                    break
            if ok:
                assign[col] = r
                res = solve(assign, new_doms)
                if res:
                    return res
                del assign[col]
            stats["backtracks"] += 1
        return None

    t0 = time.perf_counter()
    res = solve({}, domains)
    dt = time.perf_counter() - t0
    return [res[c] for c in range(N)], stats, dt


def all_solutions():
    sols = []

    def place(col, st):
        if col == N:
            sols.append(st[:])
            return
        for r in range(N):
            if all(not attacks(c, st[c], col, r) for c in range(col)):
                st.append(r)
                place(col + 1, st)
                st.pop()
    place(0, [])
    return sols


# ---------------------------------------------------------------
# 2. Local Search
# ---------------------------------------------------------------
def simulated_annealing(start, seed, T0=10.0, alpha=0.99, max_iter=20000):
    rng = random.Random(seed)
    cur = start[:]
    h = count_conflicts(cur)
    T = T0
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        if h == 0:
            return cur, it - 1, time.perf_counter() - t0
        col = rng.randrange(N)
        row = rng.randrange(N - 1)
        if row >= cur[col]:
            row += 1
        nxt = cur[:]
        nxt[col] = row
        hn = count_conflicts(nxt)
        delta = hn - h
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur, h = nxt, hn
        T = max(T * alpha, 1e-3)
    return cur, max_iter, time.perf_counter() - t0


def hill_climbing_restart(start, seed, max_restarts=1000):
    rng = random.Random(seed)
    cur = start[:]
    steps = restarts = 0
    t0 = time.perf_counter()
    while restarts <= max_restarts:
        h = count_conflicts(cur)
        while True:
            best_h, best = h, []
            for c in range(N):
                for r in range(N):
                    if r == cur[c]:
                        continue
                    nxt = cur[:]
                    nxt[c] = r
                    hn = count_conflicts(nxt)
                    if hn < best_h:
                        best_h, best = hn, [nxt]
                    elif hn == best_h and best:
                        best.append(nxt)
            if best_h >= h:
                break
            cur, h = rng.choice(best), best_h
            steps += 1
        if h == 0:
            return cur, steps, restarts, time.perf_counter() - t0
        restarts += 1
        cur = [rng.randrange(N) for _ in range(N)]
    return cur, steps, restarts, time.perf_counter() - t0


def main():
    print("=" * 60)
    print("N-QUEEN 8x8 - Initial state dari soal")
    print("=" * 60)
    print_board(INITIAL, "Initial state:")

    sols = all_solutions()
    min_move = min(moved(s) for s in sols)
    best = [s for s in sols if moved(s) == min_move]
    print(f"Total solusi valid 8-Queen: {len(sols)}")
    print(f"Perpindahan ratu minimum (optimal): {min_move}, contoh: {best}\n")

    print("=" * 60)
    print("1. DFS-BACKTRACKING (MRV + Forward Checking + LCV)")
    print("=" * 60)
    sol, st, dt = backtracking()
    print_board(sol, "Solusi Backtracking:")
    print(f"Node dicoba: {st['nodes']}, backtrack: {st['backtracks']}, "
          f"ratu dipindah: {moved(sol)}, waktu: {dt*1000:.3f} ms\n")

    print("=" * 60)
    print("2a. SIMULATED ANNEALING (start dari initial state, seed=1)")
    print("=" * 60)
    sa, it, dt_sa = simulated_annealing(INITIAL, seed=1)
    print_board(sa, "Solusi Simulated Annealing:")
    print(f"Iterasi: {it}, ratu dipindah: {moved(sa)}, waktu: {dt_sa*1000:.3f} ms\n")

    print("=" * 60)
    print("2b. HILL CLIMBING steepest-ascent + random restart (seed=1)")
    print("=" * 60)
    hc, steps, rs, dt_hc = hill_climbing_restart(INITIAL, seed=1)
    print_board(hc, "Solusi Hill Climbing:")
    print(f"Langkah: {steps}, restart: {rs}, ratu dipindah: {moved(hc)}, "
          f"waktu: {dt_hc*1000:.3f} ms\n")

    print("=" * 60)
    print("3. EKSPERIMEN 30 KALI PERCOBAAN (seed 1..30)")
    print("=" * 60)
    runs = 30
    bt_t = []
    for _ in range(runs):
        _, _, d = backtracking()
        bt_t.append(d)
    sa_ok, sa_it, sa_t, sa_mv = 0, [], [], []
    hc_ok, hc_rs, hc_t, hc_mv = 0, [], [], []
    for s in range(1, runs + 1):
        r, i, d = simulated_annealing(INITIAL, seed=s)
        if count_conflicts(r) == 0:
            sa_ok += 1
            sa_mv.append(moved(r))
        sa_it.append(i)
        sa_t.append(d)
        r, _, k, d = hill_climbing_restart(INITIAL, seed=s)
        if count_conflicts(r) == 0:
            hc_ok += 1
            hc_mv.append(moved(r))
        hc_rs.append(k)
        hc_t.append(d)
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"{'Metode':<22}{'Sukses':>8}{'Rata2 waktu (ms)':>18}{'Rata2 dipindah':>16}")
    print(f"{'Backtracking':<22}{runs:>5}/{runs}{avg(bt_t)*1000:>18.3f}{moved(sol):>16.2f}")
    print(f"{'Simulated Annealing':<22}{sa_ok:>5}/{runs}{avg(sa_t)*1000:>18.3f}{avg(sa_mv):>16.2f}")
    print(f"{'Hill Climbing+Restart':<22}{hc_ok:>5}/{runs}{avg(hc_t)*1000:>18.3f}{avg(hc_mv):>16.2f}")
    print(f"\nSA rata-rata iterasi: {avg(sa_it):.1f} (min {min(sa_it)}, max {max(sa_it)})")
    print(f"HC rata-rata restart: {avg(hc_rs):.1f} (min {min(hc_rs)}, max {max(hc_rs)})")


if __name__ == "__main__":
    main()
