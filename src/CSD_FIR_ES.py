import random
from math import inf

import numpy as np
from scipy.signal import freqz
from error_metrics import minimax, mae, mse, rmse

class CSDIndividual:
    def __init__(self, wordlength, order, n_digits, genome=None):
        """
        Initialize a CSDIndividual.
        - wordlength: number of bits per coefficient
        - order: number of coefficients
        - n_digits: maximum number of nonzero digits
        - genome: optional list of integers (CSD digits: -1, 0, +1)
        """
        self.wordlength = wordlength
        self.order = order
        self.n_digits = n_digits
        self.fitness = None
        if genome is None:
            self.genome = []
        else:
            if len(genome) != wordlength * order:
                raise ValueError("Genome length must match parameters")
            self.genome = genome

    def get_coefficients(self):
        """
        Return list of CSD-encoded coefficients
        """
        return [
            self.genome[i:(i + self.wordlength)]
            for i in range(0, len(self.genome), self.wordlength)
        ]

    def get_real_coefficients(self):
        """
        Return list of the CSD-encoded coefficients as real numbers
        """
        return [
            sum(c[i] * 2 ** (-i) for i in range(len(c)))
            for c in self.get_coefficients()
        ]

    def get_response(self, worN):
        w, H = freqz(self.get_real_coefficients(), worN=worN)
        return w, H

    def get_fitness(self, fitness, target, mode, error_metric, worN):
        if self.fitness is None:
            self.fitness = fitness(self, target, mode, error_metric, worN)
        return self.fitness


def init(wordlength, order, n_digits):
    individual = CSDIndividual(wordlength, order, n_digits)
    individual.genome = [
        x
        for _ in range(individual.order)
        for x in (lambda k, s: [s if j == k else 0 for j in range(individual.wordlength)])(
            random.randrange(individual.wordlength),
            random.choice([-1, 1])
        )
    ]
    return individual

def init_pop(n_pop, wordlength, order, n_digits):
    return [init(wordlength, order, n_digits) for _ in range(n_pop)]



def mut(individual, mutation_rate):
    """
    Perform CSD-aware mutation on the genome.
    Mutation respects:
        - canonical CSD (no adjacent ±1)
        - maximum number of non-zero digits per word (n_digits)
    By default, mutation_rate = 1 / wordlength
    """
    rand = random.random
    pick = random.choice
    n_digits = individual.n_digits

    ALT = {
        -1: (0, 1),
         0: (-1, 1),
         1: (-1, 0),
    }

    mutated = []

    for word in individual.get_coefficients():
        word = word[:]  # copy
        n = len(word)

        # Track nonzero positions explicitly
        nonzero = {idx for idx, v in enumerate(word) if v != 0}

        # If coefficient is close to 0, it has a mutation_rate% chance of becoming exactly 0
        if len(nonzero) == 1 and word[-1] != 0 and rand() < mutation_rate:
            mutated.append([0]*n)
            continue


        for i in range(n):
            if rand() >= mutation_rate:
                continue

            old = word[i]
            new = pick(ALT[old])

            # Case 1: mutate existing nonzero digit
            if old != 0:
                word[i] = new
                if new == 0:
                    nonzero.remove(i)
                continue

            # Case 2: insert ±1 into a zero digit
            # Enforce canonical CSD: clear adjacent nonzeros
            for j in (i - 1, i + 1):
                if n > j >= 0 != word[j]:
                    word[j] = 0
                    nonzero.remove(j)

            # Enforce max nonzero-digit budget
            if len(nonzero) == n_digits:
                lsnz = max(nonzero)   # least significant nonzero index
                word[lsnz] = 0
                nonzero.remove(lsnz)

            word[i] = new
            nonzero.add(i)

        mutated.append(word)

    individual.genome = [x for word in mutated for x in word]


def fit(individual, target, mode, error_metric, worN):
    w, Hi = freqz(b=individual.get_real_coefficients(), worN=worN)  # Hi : complex array
    Ht = np.array([target(wi) for wi in w], dtype=complex)
    err = []

    match mode:
        case "complex":
            err = Ht - Hi
        case "magnitude":
            err = np.abs(Ht) - np.abs(Hi)
        case "phase":
            # unwrap to avoid 2π discontinuities
            ph_t = np.unwrap(np.angle(Ht))
            ph_i = np.unwrap(np.angle(Hi))
            err = ph_t - ph_i
        case _:
            raise ValueError("mode must be 'complex', 'magnitude', or 'phase'")

    match error_metric:
        case "minimax":
            return -minimax(err)
        case "mae":
            return -mae(err)
        case "mse":
            return -mse(err)
        case "rmse":
            return -rmse(err)
        case _:
            raise ValueError("error_metric must be 'minimax', 'mae', 'mse', or 'rmse'")

def selection(pop, mu, fitness, target, mode, error_metric, worN):
    """
    Perform truncated selection of best μ individuals
    """
    pop.sort(key=lambda x: x.get_fitness(fitness, target, mode, error_metric, worN), reverse=True)
    return pop[:mu]

def ES(
        n_generations,
        mu,
        lam,
        wordlength,
        order,
        n_digits,
        target,
        mode="magnitude",
        error_metric="minimax",
        worN=128,
        mutation_rate=None,
        plus_strategy=False,
):
    """
    Evolution Strategy (ES) driver.

    Parameters:
        - n_generations : number of iterations
        - mu            : number of parents
        - lam           : number of offspring
        - wordlength    : bits per coefficient
        - order         : number of coefficients
        - n_digits      : max nonzero digits per coefficient
        - target        : target transfer function handle
        - mode          : fitness metric ("complex", "magnitude", "phase")
        - mutation_rate : override default rate; default = 1/wordlength
        - plus_strategy : False -> (μ, λ)   ; True -> (μ + λ)

    Returns:
        - best individual
        - final population
    """

    if mutation_rate is None:
        mutation_rate = 1 / wordlength

    # --- 1. Initialize μ parents ---
    pop = init_pop(mu, wordlength, order, n_digits)

    best_hist = []

    # --- 2. Evolution loop ---
    for _ in range(n_generations):
        # --- Generate λ offspring ---
        offspring = []
        for _ in range(lam):
            # (simple mutation-only ES; no recombination)
            # copy parent:
            parent = random.choice(pop)
            child = CSDIndividual(
                wordlength, order, n_digits, genome=parent.genome[:]
            )
            mut(child, mutation_rate)   # apply CSD-aware mutation
            offspring.append(child)

        # --- Selection ---
        if plus_strategy:
            # (μ + λ)-ES: parents + offspring compete
            combined = pop + offspring
            pop = selection(combined, mu, fit, target, mode, error_metric, worN)
        else:
            # (μ, λ)-ES: only offspring compete
            pop = selection(offspring, mu, fit, target, mode, error_metric, worN)
        best_hist.append(pop[0])

    # --- Return best solution ---
    best = selection(pop, 1, fit, target, mode, error_metric, worN)[0]
    return best, best_hist
