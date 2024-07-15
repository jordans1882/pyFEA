import itertools

import numpy as np


def combination_factorizer(fact_size, dim):
    if fact_size <= 0:
        return None
    factors = list(itertools.combinations(range(dim), fact_size))
    return factors


def linear_factorizer(fact_size, overlap, dim):
    if fact_size is None or fact_size <= overlap or fact_size <= 0:
        return None
    smallest = 0
    largest = fact_size
    factors = []
    while largest <= dim:
        factors.append([x for x in range(smallest, largest)])
        smallest = largest - overlap
        largest += fact_size - overlap
    if smallest < dim:
        factors.append([x for x in range(smallest, dim)])
    return factors


def coevolution_factorizer(fact_size, dim):
    k_1 = dim % fact_size
    k_2 = fact_size - k_1
    start = 0
    end = k_1
    factors = []
    i = 0
    while end <= dim:
        if i % 2 == 0:
            factors.append([x for x in range(start, end)])
            start += k_2
            end += k_2
        else:
            factors.append([x for x in range(start, end)])
            start += k_1
            end += k_1
        i += 1
    return factors


def random_factorizer(dim, num_covers, fact_size=None):
    if fact_size <= 0 or num_covers <= 0:
        return None
    factors = []
    for i in range(num_covers):
        covered = []
        while covered != list(range(dim)):
            if fact_size is None:
                fact_size = np.random.randint(1, dim)
            new_fact = list(np.random.choice(dim, new_fact_size, replace=False))
            for var in new_fact:
                if var not in covered:
                    covered.append(var)
            covered.sort()
            factors.append(new_fact)
    return factors
