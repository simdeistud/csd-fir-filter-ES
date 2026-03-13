import random
import numpy as np
from scipy.signal import freqz
from multiprocessing import Pool


# ================================================================
# CSD Individual
# ================================================================
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

    # ------------------------------------------------------------
    def get_coefficients(self):
        wl = self.wordlength
        return [self.genome[i:i + wl] for i in range(0, len(self.genome), wl)]

    # ------------------------------------------------------------
    # NUMPY-VECTORIZED COEFFICIENT DECODING
    def get_real_coefficients(self):
        G = np.array(self.get_coefficients())  # [order, wordlength]
        weights = 2.0 ** -np.arange(self.wordlength)
        return G @ weights  # fast dot-product

    # ------------------------------------------------------------
    def get_response(self, worN=2048):
        c = self.get_real_coefficients()
        return freqz(c, worN=worN)

    # ------------------------------------------------------------
    def get_fitness(self, fitness, target, mode):
        return fitness(self, target, mode)


# ================================================================
# Initialization
# ================================================================
def init(wordlength, order, n_digits):
    ind = CSDIndividual(wordlength, order, n_digits)
    ind.genome = [
        x for _ in range(ind.order)
        for x in (lambda k, s: [s if j == k else 0 for j in range(ind.wordlength)])(
            random.randrange(ind.wordlength), random.choice([-1, 1])
        )
    ]
    return ind


def init_pop(n_pop, wordlength, order, n_digits):
    return [init(wordlength, order, n_digits) for _ in range(n_pop)]


# ================================================================
# MUTATION (unchanged semantics, cleaned)
# ================================================================
def mut(individual, mutation_rate=None):

    if mutation_rate is None:
        mutation_rate = 1 / individual.wordlength

    def ls_nonzero(coeff):
        for i in range(len(coeff) - 1, -1, -1):
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

                # --- modifying nonzero ---
                if old != 0:
                    m[i] = choice
                    if choice == 0:
                        curr_d -= 1
                    continue

                # --- inserting nonzero in zero position ---
                if i == 0:
                    if m[1] != 0:
                        m[1] = 0
                        curr_d -= 1
                    elif curr_d == individual.n_digits:
                        idx = ls_nonzero(m)
                        if idx is not None:
                            m[idx] = 0
                            curr_d -= 1
                    m[0] = choice
                    curr_d += 1
                    continue

                if i == len(m) - 1:
                    if m[i - 1] != 0:
                        m[i - 1] = 0
                        curr_d -= 1
                    elif curr_d == individual.n_digits:
                        idx = ls_nonzero(m)
                        if idx is not None:
                            m[idx] = 0
                            curr_d -= 1
                    m[i] = choice
                    curr_d += 1
                    continue

                L = m[i - 1]
                R = m[i + 1]

                if L != 0 and R != 0:
                    m[i - 1] = 0
                    m[i + 1] = 0
                    curr_d -= 2
                    m[i] = choice
                    curr_d += 1
                    continue

                if L != 0:
                    m[i - 1] = 0
                    curr_d -= 1
                    m[i] = choice
                    curr_d += 1
                    continue

                if R != 0:
                    m[i + 1] = 0
                    curr_d -= 1
                    m[i] = choice
                    curr_d += 1
                    continue

                if curr_d == individual.n_digits:
                    idx = ls_nonzero(m)
                    if idx is not None:
                        m[idx] = 0
                        curr_d -= 1

                m[i] = choice
                curr_d += 1

        mutated.append(m)

    # flatten
    individual.genome = [x for c in mutated for x in c]


# ================================================================
# FITNESS (vectorized)
# ================================================================
def fit(individual, target, mode="complex"):
    coeff = individual.get_real_coefficients()
    w, Hi = freqz(coeff)

    Ht = np.array([target(wi) for wi in w], dtype=complex)

    if mode == "complex":
        diff = Ht - Hi

    elif mode == "magnitude":
        diff = np.abs(Ht) - np.abs(Hi)

    elif mode == "phase":
        diff = np.unwrap(np.angle(Ht)) - np.unwrap(np.angle(Hi))

    else:
        raise ValueError("Invalid fitness mode.")

    return -np.linalg.norm(diff)


# ================================================================
# SELECTION
# ================================================================
def selection(pop, mu, fitness, target, mode):
    pop.sort(key=lambda x: fitness(x, target, mode), reverse=True)
    return pop[:mu]


# ================================================================
# PARALLEL EVALUATION WRAPPER
# ================================================================
def _eval(args):
    ind, target, mode = args
    return fit(ind, target, mode)


# ================================================================
# PARALLEL EVOLUTION STRATEGY
# ================================================================
def ES_parallel(
        n_generations,
        mu,
        lam,
        wordlength,
        order,
        n_digits,
        target,
        mode="complex",
        mutation_rate=None,
        plus_strategy=False,
        n_workers=8
):
    pop = init_pop(mu, wordlength, order, n_digits)

    best_hist = []

    with Pool(n_workers) as pool:

        for _ in range(n_generations):

            # --- offspring generation ---
            offspring = []
            for _ in range(lam):
                parent = random.choice(pop)
                child = CSDIndividual(
                    wordlength, order, n_digits,
                    genome=parent.genome[:]
                )
                mut(child, mutation_rate)
                offspring.append(child)

            # --- selection ---
            if plus_strategy:
                combined = pop + offspring
                F = pool.map(_eval, [(ind, target, mode) for ind in combined])
                idx = np.argsort(F)[::-1][:mu]
                pop = [combined[i] for i in idx]

            else:
                F = pool.map(_eval, [(ind, target, mode) for ind in offspring])
                idx = np.argsort(F)[::-1][:mu]
                pop = [offspring[i] for i in idx]
        best_hist.append(pop[0])

    best = max(pop, key=lambda ind: fit(ind, target, mode))
    return best, best_hist