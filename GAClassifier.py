import time
from tracemalloc import start
import pygad
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter

from KeyClassifier import *
from Bird import *
import constants
from play import manyPlaysResults, playGame


class GAKeyClassifier(KeyClassifier):
    """
    Classe que implementa o classificador KNN

    parametros:
    - state: Vetor de coordenada dos pontos classificados
    """
    def __init__(self, state):
        self.state = state

        # Normaliza o estado
        self.scaler = StandardScaler()
        self.state = self.scaler.fit_transform(np.asarray(self.state).reshape(constants.initial_state_size,constants.coord_size))
        # self.state = self.state.reshape(constants.initial_state_size)

    def find_closest(self, frame, k):
        """
        Função que busca o ponto mais próximo dos dados atuais

        parametros:
        - frame: Vetor contendo os dados atuais
            (distancia, altura do obstáculo, velocidade)
        - k: Número de pontos mais próximos aos dados atuais que serão considerados

        retorno:
        - index: Índice do ponto mais próximo
        """
        dist = [np.linalg.norm(self.state[i]- frame) for i in range(0, len(self.state))]

        # print(min(dist))

        dist = np.asarray(dist)
        idx = np.argpartition(dist, k)[:k]

        keys = [constants.label_state[val] for val in idx]

        count_keys = Counter(keys)
        
        return count_keys.most_common(1)[0][0]
        # if min(dist) < 0.2:
        #     return dist.index(min(dist))
        # else: 
        #     return -1

    def norm_state(self, params):
        """
        Função que normaliza o vetor de coordenadas dos pontos classificados, e os dados atuais

        parametros:
        - params: Vetor de coordenadas dos dados atuais

        retorno:
        - params: Vetor normalizado
        """

        params = self.scaler.transform(np.asarray(params).reshape(1,constants.coord_size))
        params = params.reshape(constants.coord_size)

        return params
        


    def keySelector(self, params):
        """
        Função que retorna a tecla a ser pressionada para o jogador

        parametros:
        - distance: Distância do jogador até o obstáculo
        - obaltitude: Altura do obstáculo
        - speed: Velocidade do jogador

        retorno:
        - key: Tecla a ser pressionada
        """

        # caso em que não há obstáculo
        if params[1] == 0:
            return "K_NO"
        
        params = self.norm_state(params)

        closest = self.find_closest(params, 2)

        return closest

        # if closest == -1:
        #     return "K_NO"
        # elif closest % 2 == 0:
        #     return "K_DOWN"
        # return "K_UP"


    def updateState(self, state):
        """
        Função que atualiza o vetor de coordenadas dos pontos classificados

        parametros:
        - state: Vetor de coordenadas dos pontos classificados
        """
        self.state = state

def fitness_func(solution, solution_idx):
    """
    Função que calcula o fitness de uma solução
    
    parametros:
    - solution: Solução a ser avaliada
    - solution_idx: Índice da solução na população
    """ 

    constants.aiPlayer = GAKeyClassifier(solution)
    # res, value = manyPlaysResults(3)
    value = playGame()

    return value

def check_generation(instance):
    """
    Função que verifica se o tempo de execução do algoritmo está atingindo o limite

    parametros:
    - instance: Instância do algoritmo

    retorno:
    - "stop": Se o tempo de execução tiver atingido o limite
    - "continue": Se o tempo de execução não tiver atingido o limite
    """
    actual_time = time.process_time() - start_time 
    print('finished a generation in', actual_time, 'seconds')

    if actual_time > time_max:
        return "stop"
    return "continue"

def genetic_algorithm(state_size, max_time):
    """
    Função que implementa o algoritmo genético

    parametros:
    - state: Quantidade de pontos que serão buscados
    - max_time: Tempo máximo de execução do algoritmo

    retorno:
    - best_solution: Melhor solução encontrada
    - best_fitness: Fitness da melhor solução encontrada
    """
    global time_max, start_time
    time_max = max_time

    fitness_function = fitness_func

    num_generations = 800
    num_parents_mating = 4

    sol_per_pop = 20
    gene_type = int

    # Define o espaço para os genes referentes a altitude do obstáculo
    gene_space = [None, [260, 300, 325, 345], None] * state_size

    num_genes = state_size * 3

    init_range_low = 0
    init_range_high = 1000

    parent_selection_type = "tournament"
    keep_parents = 2

    crossover_type = "two_points"
    crossover_probability = 0.8
    K_tournament=3

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
                            K_tournament=K_tournament,
                            crossover_probability=crossover_probability,
                            mutation_type=mutation_type,
                            mutation_probability=mutation_probability,
                            stop_criteria=stop_criteria,
                            on_generation=check_generation)

    start_time = time.process_time()
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # print("Parameters of the best solution : {solution}".format(solution=solution))
    # print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    return solution, solution_fitness

