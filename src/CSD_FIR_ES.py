import random
import numpy as np
from scipy.signal import freqz

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

    def get_response(self, worN=2048):
        w, H = freqz(self.get_real_coefficients(), worN=worN)
        return w, H

    def get_fitness(self, fitness, target, mode):
        if self.fitness is None:
            self.fitness = fitness(self, target, mode)
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


def mut(individual, mutation_rate=None):
    """
    Perform CSD-aware mutation on the genome.
    Mutation respects:
        - canonical CSD (no adjacent ±1)
        - maximum number of non-zero digits per word (n_digits)
    By default, mutation_rate = 1/wordlength
    """
    if mutation_rate is None:
        mutation_rate = 1 / individual.wordlength

    def get_least_significant_nonzero_index(coeff):
        return next((i for i in range(len(coeff) - 1, -1, -1) if coeff[i] != 0), None)

    mutated = []
    # Split into CSD words
    for coeff in individual.get_coefficients():
        curr_d = sum(1 for x in coeff if x != 0)
        mutated_coeff = coeff[:]  # copy

        for i in range(len(mutated_coeff)):
            if random.random() < mutation_rate:

                # pick alternative symbol
                choices = [-1, 0, 1]
                choices.remove(mutated_coeff[i])
                choice = random.choice(choices)

                # Case 1: modifying a nonzero digit
                if mutated_coeff[i] != 0:
                    mutated_coeff[i] = choice
                    if choice == 0:
                        curr_d -= 1
                    # if choice stays ±1, curr_d unchanged
                    continue

                # Case 2: digit is zero → insert ±1
                # FIRST BIT
                if i == 0:
                    if mutated_coeff[i + 1] != 0:
                        mutated_coeff[i + 1] = 0
                        curr_d -= 1
                    elif curr_d == individual.n_digits:
                        mutated_coeff[get_least_significant_nonzero_index(mutated_coeff)] = 0
                        curr_d -= 1
                    mutated_coeff[i] = choice
                    curr_d += 1
                    continue

                    # LAST BIT
                if i == len(mutated_coeff) - 1:
                    if mutated_coeff[i - 1] != 0:
                        mutated_coeff[i - 1] = 0
                        curr_d -= 1
                    elif curr_d == individual.n_digits:
                        mutated_coeff[get_least_significant_nonzero_index(mutated_coeff)] = 0
                        curr_d -= 1
                    mutated_coeff[i] = choice
                    curr_d += 1
                    continue

                    # MIDDLE REGION
                left = mutated_coeff[i - 1]
                right = mutated_coeff[i + 1]

                # both neighbors nonzero
                if left != 0 and right != 0:
                    mutated_coeff[i - 1] = 0
                    mutated_coeff[i + 1] = 0
                    curr_d -= 2
                    mutated_coeff[i] = choice
                    curr_d += 1
                    continue

                # only left nonzero
                if left != 0:
                    mutated_coeff[i - 1] = 0
                    curr_d -= 1
                    mutated_coeff[i] = choice
                    curr_d += 1
                    continue

                # only right nonzero
                if right != 0:
                    mutated_coeff[i + 1] = 0
                    curr_d -= 1
                    mutated_coeff[i] = choice
                    curr_d += 1
                    continue

                # both neighbors zero
                if curr_d == individual.n_digits:
                    mutated_coeff[get_least_significant_nonzero_index(mutated_coeff)] = 0
                    curr_d -= 1
                mutated_coeff[i] = choice
                curr_d += 1

        mutated.append(mutated_coeff)
    # Re-flatten: concatenate mutated words
    individual.genome = [x for coeff in mutated for x in coeff]

def fit(individual, target, mode="complex"):
    w, Hi = freqz(individual.get_real_coefficients())  # Hi : complex array
    Ht = np.array([target(wi) for wi in w], dtype=complex)

    if mode == "complex":
        diff = Ht - Hi

    elif mode == "magnitude":
        diff = np.abs(Ht) - np.abs(Hi)

    elif mode == "phase":
        # unwrap to avoid 2π discontinuities
        ph_t = np.unwrap(np.angle(Ht))
        ph_i = np.unwrap(np.angle(Hi))
        diff = ph_t - ph_i
    else:
        raise ValueError("mode must be 'complex', 'magnitude', or 'phase'")

    return -np.linalg.norm(diff)  # sqrt(sum |diff|^2)

def selection(pop, mu, fitness, target, mode):
    """
    Perform truncated selection of best μ individuals
    """
    pop.sort(key=lambda x: x.get_fitness(fitness, target, mode), reverse=True)
    return pop[:mu]

def ES(
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
            pop = selection(combined, mu, fit, target, mode)
        else:
            # (μ, λ)-ES: only offspring compete
            pop = selection(offspring, mu, fit, target, mode)
        best_hist.append(pop[0])

    # --- Return best solution ---
    best = selection(pop, 1, fit, target, mode)[0]
    return best, best_hist
