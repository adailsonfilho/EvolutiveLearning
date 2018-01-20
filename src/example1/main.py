import numpy as np
from neural import Neuron
from geneticalgorithm import ga
import matplotlib.pyplot as plt

n = Neuron(2)


dataset = np.array(
    [
        [3, 1.7, 0],
        [1, 1.6, 0],
        [2, 1.9, 0],
        [3, 2.5, 0],
        [1, 2.1, 0],
        [2, 2.7, 0],
        [1, 1, 1],
        [2, 1.25, 1],
        [3, 1, 1],
        [1.3, 1, 1],
        [2.3, 1, 1],
        [3.1, 1, 1],
    ]
)

index_class0 = dataset[:, 2] == 0
index_class1 = dataset[:, 2] == 1

cutpoint = 4

train_set = np.concatenate([dataset[index_class0][:cutpoint], dataset[index_class1][:cutpoint]])
test_set = np.concatenate([dataset[index_class0][cutpoint:], dataset[index_class1][cutpoint:]])

plt.scatter(dataset[:6][:, 0], dataset[:6][:, 1], marker='^', c='red')
plt.scatter(dataset[6:][:, 0], dataset[6:][:, 1], marker='o', c='blue')
plt.show()


def evaluateneuron(data, neuron, debug=False):
    errors = 0
    for row in data:
        sample = row[:2]
        target = row[2]
        guess = neuron.stimulate(sample)

        if debug:
            print(f"returned {guess}, expected {target}")

        if target != guess:
            errors += 1

    return errors


# start train
population_size = 20
population = [Neuron(2) for i in range(population_size)]


def crossover(ind1, ind2):
    size = ind1.weights.shape[0]
    mid = int(size/2)
    weights = np.concatenate([ind1.weights[:mid], ind2.weights[mid:]])
    threshold = np.random.normal(((ind1.threshold + ind2.threshold)/2), 0.1)
    new_neuron = Neuron(weights=weights, threshold=threshold)

    return [new_neuron]


def mutation(individual):
    weights = individual.weights[:]
    weights = np.random.normal(weights, 1.5)
    threshold = individual.threshold * np.random.normal()
    return Neuron(weights=weights, threshold=threshold)


def fitness(individual):
    return evaluateneuron(train_set, individual)


result = ga(population=population, progenitors_amount=2, offsprings=2, objective='minimize', crossover=crossover, mutation=mutation, fitness=fitness, max_epochs=100, crossover_prob=0.98, mutation_prob=0.3, elitist=False)
errors = evaluateneuron(test_set, result['individual'], debug=True)

print(f"Error: {(errors/len(dataset))*100}%")
print(f"Weights: {result['individual'].weights}")
print(f"Threshold: {result['individual'].threshold}")


w0 = result['individual'].weights[0]
w1 = result['individual'].weights[1]
w2 = result['individual'].threshold

slope = -(w0 / w2) / (w0 / w1)
intercept = -w0 / w2

plt.scatter(dataset[:6][:, 0], dataset[:6][:, 1], marker='^', c='red')
plt.scatter(dataset[6:][:, 0], dataset[6:][:, 1], marker='o', c='blue')
plt.plot([-1, 4], [(intercept+slope*-1), (intercept+slope*4)])
plt.show()
