# ==============================================================================
# GPU-ONLY (CuPy) EVOLUTION STRATEGY FOR CSD FIR FILTER SYNTHESIS
# Includes:
#   - CSDIndividual with get_response()
#   - GPU batch fitness via CuPy (no NumPy fallback)
#   - FIR analytic spectrum computation H = sum b_k e^{-jw k}
# ==============================================================================

import random
import cupy as cp


# ========================= CSD Individual =====================================
class CSDIndividual:
    def __init__(self, wordlength, order, n_digits, genome=None):
        self.wordlength = wordlength
        self.order = order
        self.n_digits = n_digits

        if genome is None:
            self.genome = []
        else:
            if len(genome) != wordlength * order:
                raise ValueError("Genome length mismatch")
            self.genome = genome

    def get_coefficients(self):
        wl = self.wordlength
        return [self.genome[i:i+wl] for i in range(0, len(self.genome), wl)]

    def get_real_coefficients(self):
        """Vectorized CSD decoding (CPU → small)"""
        G = cp.array(self.get_coefficients(), dtype=cp.float64)   # [order, wordlength]
        weights = cp.power(2.0, -cp.arange(self.wordlength, dtype=cp.float64))
        return G @ weights  # [order]

    # ======================== RESTORED GPU get_response() ======================
    def get_response(self, worN=2048):
        """
        GPU computation identical to scipy.signal.freqz for FIR a=[1].
        Returns (w, H) on GPU.
        """
        coeff = self.get_real_coefficients()        # [order]
        order = coeff.shape[0]

        w = cp.linspace(0.0, cp.pi, worN, dtype=cp.float64)  # [N]
        k = cp.arange(order, dtype=cp.float64)               # [order]

        # Construct exponentials: (N, order)
        E = cp.exp(-1j * w[:, None] * k[None, :])

        # Response H(w) = Σ_k b[k] * exp(-jw k)
        H = E @ coeff

        return w, H  # both are CuPy arrays


# ============================ Initialization ==================================
def init(wordlength, order, n_digits):
    ind = CSDIndividual(wordlength, order, n_digits)
    ind.genome = [
        x for _ in range(ind.order)
        for x in (lambda k, s: [s if j == k else 0 for j in range(ind.wordlength)])(
            random.randrange(ind.wordlength),
            random.choice([-1, 1])
        )
    ]
    return ind

def init_pop(n_pop, wordlength, order, n_digits):
    return [init(wordlength, order, n_digits) for _ in range(n_pop)]


# ============================ Mutation (unchanged semantics) ===================
def mut(individual, mutation_rate=None):
    if mutation_rate is None:
        mutation_rate = 1 / individual.wordlength

    def ls_nonzero(coeff):
        for i in range(len(coeff)-1, -1, -1):
            if coeff[i] != 0:
                return i
        return None

    mutated = []

    for coeff in individual.get_coefficients():
        curr_d = sum(1 for x in coeff if x != 0)
        m = coeff[:]

        for i in range(len(m)):
            if random.random() < mutation_rate:

                old = m[i]
                choices = [-1, 0, 1]
                choices.remove(old)
                choice = random.choice(choices)

                if old != 0:
                    m[i] = choice
                    if choice == 0:
                        curr_d -= 1
                    continue

                if i == 0:
                    if m[1] != 0:
                        m[1] = 0; curr_d -= 1
                    elif curr_d == individual.n_digits:
                        idx = ls_nonzero(m)
                        if idx is not None:
                            m[idx] = 0; curr_d -= 1
                    m[0] = choice; curr_d += 1
                    continue

                if i == len(m)-1:
                    if m[i-1] != 0:
                        m[i-1] = 0; curr_d -= 1
                    elif curr_d == individual.n_digits:
                        idx = ls_nonzero(m)
                        if idx is not None:
                            m[idx] = 0; curr_d -= 1
                    m[i] = choice; curr_d += 1
                    continue

                L = m[i-1]
                R = m[i+1]

                if L != 0 and R != 0:
                    m[i-1] = m[i+1] = 0
                    curr_d -= 2
                    m[i] = choice; curr_d += 1
                    continue

                if L != 0:
                    m[i-1] = 0; curr_d -= 1
                    m[i] = choice; curr_d += 1
                    continue

                if R != 0:
                    m[i+1] = 0; curr_d -= 1
                    m[i] = choice; curr_d += 1
                    continue

                if curr_d == individual.n_digits:
                    idx = ls_nonzero(m)
                    if idx is not None:
                        m[idx] = 0; curr_d -= 1

                m[i] = choice; curr_d += 1

        mutated.append(m)

    individual.genome = [x for c in mutated for x in c]


# ============================ GPU Batch Evaluator ==============================
class FIRBatchEvaluatorGPU:
    """
    Batched GPU fitness evaluator using CuPy only.
    """
    def __init__(self, order, worN, target, wordlength, mode="complex"):
        self.order = order
        self.worN = worN
        self.wordlength = wordlength
        self.mode = mode

        self.w = cp.linspace(0.0, cp.pi, worN, dtype=cp.float64)  # [N]
        k = cp.arange(order, dtype=cp.float64)

        # Precompute exponentials (N, order)
        self.E = cp.exp(-1j * self.w[:, None] * k[None, :])

        # Precompute target response Ht(w)
        w_cpu = cp.asnumpy(self.w)
        Ht_cpu = cp.array([target(wi) for wi in w_cpu], dtype=cp.complex128)
        self.Ht = cp.asarray(Ht_cpu)

        # CSD decode weights
        self.weights = cp.power(2.0, -cp.arange(wordlength, dtype=cp.float64))

    def genomes_to_coeffs(self, pop):
        B = len(pop)
        G = cp.array([ind.genome for ind in pop], dtype=cp.float64)  # [B, order*W]
        G = G.reshape(B, self.order, self.wordlength)
        return cp.tensordot(G, self.weights, axes=(2, 0))  # [B, order]

    def batch_fitness(self, pop):
        coeffs = self.genomes_to_coeffs(pop)   # [B, order]
        H = self.E @ coeffs.T                  # [N, B]
        Ht = self.Ht[:, None]                 # [N,1]

        if self.mode == "complex":
            diff = Ht - H
        elif self.mode == "magnitude":
            diff = cp.abs(Ht) - cp.abs(H)
        elif self.mode == "phase":
            diff = cp.unwrap(cp.angle(Ht), axis=0) - cp.unwrap(cp.angle(H), axis=0)
        else:
            raise ValueError("Invalid mode")

        power = (diff.real**2 + diff.imag**2).sum(axis=0)
        norms = cp.sqrt(power)
        return -cp.asnumpy(norms)  # return CPU fitness array


# =============================== Selection ====================================
def selection(pop, fitness, mu):
    idx = fitness.argsort()[::-1][:mu]
    return [pop[i] for i in idx]


# ============================== GPU ES ========================================
def ES_gpu(
        n_generations,
        mu,
        lam,
        wordlength,
        order,
        n_digits,
        target,
        worN=2048,
        mode="complex",
        mutation_rate=None,
        plus_strategy=False
):
    pop = init_pop(mu, wordlength, order, n_digits)
    evaluator = FIRBatchEvaluatorGPU(order, worN, target, wordlength, mode)

    best_hist = []

    for _ in range(n_generations):
        offspring = []
        for _ in range(lam):
            parent = random.choice(pop)
            child = CSDIndividual(wordlength, order, n_digits, genome=parent.genome[:])
            mut(child, mutation_rate)
            offspring.append(child)

        if plus_strategy:
            pool = pop + offspring
            F = evaluator.batch_fitness(pool)
            pop = selection(pool, F, mu)
        else:
            F = evaluator.batch_fitness(offspring)
            pop = selection(offspring, F, mu)
        best_hist.append(pop[0])

    # final best
    F_final = evaluator.batch_fitness(pop)
    best = pop[F_final.argmax()]
    return best, best_hist