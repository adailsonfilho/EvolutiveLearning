import numpy as np

"""
ATTRIBUTES DESCRIPTIONS

This project provides a general schema for Genetics Algorithms, allowing a developer define the basic steps and components of genetics algorithms. There are following the arguments descriptions:

 - fitness: a function that will be run for calculate the fitness of each individual

 - population: numpy arrays with the initial population

 - crossover_prob: a double between 0 and 1 that determines a probability of ocorring a crossover

 - mutation_prob: a double between 0 and 1 that determines a probability of ocorring a mutation

 - crossover: a function that perform a crossover between two individuals (that must be specified by the developer) and it is called when raffled based on the "crossover_prob" argument, it returns a list of new children for the two given parents

 - mutation: a function that perform a mutation on a single individual and it is called when raffled based on the "mutation_prob" argument

 - objective: "minimize" is the defalt value. The options are: "maximize" or "minimize", for the obvious reasons

 - max_epochs maximum number of interactions

 - generational: True if the progenitors always die, and False if they can still live and the generation die

 - mutation_extra_individual: True or False, defines if a mutation of a individual is considered a new individual or replace de individual

 - stop_if_reachs: fitness good enough for stopping, if not set, will only stop when achieve the "max_epochs" epochs

 - offsprings: number of offsprings

 - progenitors_amount: amount of individuals considered in the wheel selection

 - elitist: True or False, defines if only the best one survives

"""


def ga(population, fitness, crossover_prob, mutation_prob, crossover, mutation, max_epochs, offsprings,
       progenitors_amount, stop_if_reachs=None, objective="minimize", generational=False,
       mutation_extra_individual=False, elitist=True, each_individual=None, program_ends=None):
    """
    CONTROL PARAMETTERS INITIALIZATION
    """

    # Best individual found
    best = {'individual': None, 'fitness': None}
    epoch = 0
    fitness_values = np.zeros(len(population))

    # define if the algorithm achieved the objective
    def objective_achieved():

        if best['fitness'] is not None and stop_if_reachs is not None:
            if objective == 'minimize':
                return best['fitness'] <= stop_if_reachs
            elif objective == 'maximize':
                return best['fitness'] >= stop_if_reachs

            raise ('given "objective" argument is invalid')

        return False

    def bestOne(fit1, fit2):

        if objective == 'maximize':

            if fit1 > fit2:
                return fit1
            else:
                return fit2

        elif objective == 'minimize':

            if fit1 < fit2:
                return fit1
            else:
                return fit2

        raise "Objetive type Error!"

    # Population need to be inserted already initilized
    def calculate_all_fitness():

        for i, individual in enumerate(population):

            fit_val = fitness(individual)
            fitness_values[i] = fit_val

            if best['individual'] is None:
                best['individual'] = individual
                best['fitness'] = fit_val

            if bestOne(fit_val, best['fitness']) == fit_val:
                best['fitness'] = fit_val
                best['individual'] = individual

            if each_individual is not None:
                each_individual(population[i], fitness_values[i], epoch)

    # simulate a do while around the fitness
    calculate_all_fitness()
    epoch += 1

    while epoch < max_epochs and not objective_achieved():

        """
        Calculate fitness
        """

        calculate_all_fitness()

        mean = np.mean(fitness_values)

        print('Epoch:', epoch, '> Best: ', best['fitness'], '|| > Mean: ', mean)

        # raffle crossover
        doCrossover = np.random.random()
        if doCrossover <= crossover_prob:

            """
            Generate offspring 
            """

            offspring = []
            offspring_fitness_val = []

            """
            Select progenitors (Using ranking)
            """

            # storage prob of each candidate for mating
            probs = []

            # get sample of population for participate of selection process
            candidates = np.random.choice(range(len(population)), size=progenitors_amount, replace=False)

            # rank by fitness using pre computed values
            candidates = sorted(candidates, key=lambda x: fitness_values[x])

            # calculate prob of each one
            fit_sum = sum(list(range(progenitors_amount + 1)))

            for i in range(progenitors_amount):
                probs.append((i + 1) / fit_sum)

            for i in range(offsprings):

                # randomly select 2 progenitors
                progenitors = np.random.choice(candidates, size=2, replace=False, p=probs)

                """
                Crossover
                """
                # generate new individual
                new_individuals = crossover(population[progenitors[0]], population[progenitors[1]])

                """
                Mutation
                """
                # raffle mutation
                for new_individual in new_individuals:

                    if np.random.random() <= mutation_prob:
                        mutated = mutation(new_individual)

                        if mutation_extra_individual:
                            offspring.append(mutated)
                            offspring_fitness_val.append(fitness(mutated))
                        else:
                            new_individual = mutated

                    offspring.append(new_individual)
                    offspring_fitness_val.append(fitness(new_individual))

            """
            Surviving
            """
            # joining all living individuals
            population_size = len(population)
            population = np.concatenate((population, offspring))
            fitness_values = np.concatenate((fitness_values, offspring_fitness_val))
            population_rank = sorted(list(range(len(population))), key=lambda x: fitness_values[x], reverse=True)

            new_population = []
            for i in population_rank:
                new_population.append(population[i])

            new_population = np.array(new_population)

            if elitist:

                # only the best survives
                population = new_population[:population_size]

                # warant that the next parent selection has not a fitness bias
                np.random.shuffle(population)

            else:
                # A fare surviving chance for every individual


                # storage prob of each candidate survive
                probs = []

                # calculate prob of each one
                fit_sum = sum(list(range(len(new_population) + 1)))

                for i in range(len(new_population)):
                    probs.append((i + 1) / fit_sum)

                # randomly select who lives
                indexes = np.random.choice(range(len(new_population)), size=population_size, replace=False, p=probs)
                population = new_population[indexes]

        epoch += 1

    if program_ends is not None:
        program_ends()

    return best
