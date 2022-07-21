import time
from tracemalloc import start
import pygad
import numpy as np

from KeyClassifier import *
from Bird import *
import constants
from play import manyPlaysResults


class GAKeyClassifier(KeyClassifier):
    def __init__(self, state, label):
        self.state = state
        self.label = label

    def find_closest(self, frame):
        dist = [np.linalg.norm(self.state[i:i+3]- frame) for i in range(0, len(self.state), 3)]
        # print('dist to closest:', min(dist))

        if min(dist) < 150:
            return dist.index(min(dist))
        else: 
            return -1


    def keySelector(self, distance, obaltitude, speed):
        # print(self.state)
        if obaltitude == 0:
            return "K_NO"
        
        closest = self.find_closest([distance, obaltitude, speed])
        
        if closest % 2 == 0:
            return "K_DOWN"
        elif closest == -1:
            return "K_NO"
        return "K_UP"


    def updateState(self, state):
        self.state = state

def fitness_func(solution, solution_idx):
    constants.aiPlayer = GAKeyClassifier(solution, initial_state_label)
    res, value = manyPlaysResults(3)

    return value

def check_generation(instance):
    actual_time = time.process_time() - start_time 
    print('finished a generation in', actual_time, 'seconds')
    if actual_time > time_max:
        return "stop"
    return "continue"

def genetic_algorithm(state, max_time, label):
    global initial_state_label
    initial_state_label = label

    global time_max, start_time
    time_max = max_time

    fitness_function = fitness_func

    num_generations = 500
    num_parents_mating = 4

    sol_per_pop = 15
    gene_type = int
    gene_space = [None, [260, 300, 325, 345], None,
                  None, [260, 300, 325, 345], None,
                  None, [260, 300, 325, 345], None,
                  None, [260, 300, 325, 345], None,
                  None, [260, 300, 325, 345], None,
                  None, [260, 300, 325, 345], None]
    num_genes = len(label) * 3

    init_range_low = 0
    init_range_high = 600

    parent_selection_type = "tournament"
    keep_parents = 2

    crossover_type = "two_points"
    crossover_probability = 0.4

    mutation_type = "random"
    mutation_probability = 0.2

    stop_criteria= "saturate_100"


    ga_instance = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_function,
                            sol_per_pop=sol_per_pop,
                            gene_type=gene_type,
                            num_genes=num_genes,
                            gene_space=gene_space,
                            init_range_low=init_range_low,
                            init_range_high=init_range_high,
                            parent_selection_type=parent_selection_type,
                            keep_parents=keep_parents,
                            crossover_type=crossover_type,
                            crossover_probability=crossover_probability,
                            mutation_type=mutation_type,
                            mutation_probability=mutation_probability,
                            stop_criteria=stop_criteria,
                            on_generation=check_generation)

    start_time = time.process_time()
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    return solution, solution_fitness

