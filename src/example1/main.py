import numpy as np
from neural import Neuron
from geneticalgorithm import ga
import matplotlib.pyplot as plt

n = Neuron(2)

# data format: x1,x2, target
dataset = np.array(
    [
        [2, 1.9, 0],
        [3, 1.7, 0],
        [3, 2.5, 0],
        [1, 1.6, 0],
        [2, 2.7, 0],
        [1, 2.1, 0],
        [1, 1, 1],
        [3, 1, 1],
        [3.1, 1, 1],
        [2, 1.25, 1],
        [2.3, 1, 1],
        [1.3, 1, 1],
    ]
)

index_class0 = dataset[:, 2] == 0
index_class1 = dataset[:, 2] == 1

# cutpoint = 4
#
# train_set = np.concatenate([dataset[index_class0][:cutpoint], dataset[index_class1][:cutpoint]])
# test_set = np.concatenate([dataset[index_class0][cutpoint:], dataset[index_class1][cutpoint:]])

train_set = dataset
test_set = dataset

plt.scatter(dataset[:6][:, 0], dataset[:6][:, 1], marker='^', c='red')
plt.scatter(dataset[6:][:, 0], dataset[6:][:, 1], marker='o', c='blue')
plt.show()


def evaluateneuron(data, neuron, debug=False, plt=None):
    errors = 0
    for row in data:
        sample = row[:2]
        target = row[2]
        guess = neuron.stimulate(sample)

        if debug:
            print(f"returned {guess}, expected {target}")

        if target != guess:
            errors += 1
            if plt is not None:
                plt.scatter(sample[0],sample[1], marker='+', s=200, c='black')

    return errors


def crossover(ind1, ind2):
    size = ind1.weights.shape[0]
    mid = int(size/2)
    weights = np.concatenate([ind1.weights[:mid], ind2.weights[mid:]])
    threshold = np.random.normal(((ind1.threshold + ind2.threshold)/2)*np.random.random(), 0.1)
    n1 = Neuron(weights=weights, threshold=threshold)

    weights = np.concatenate([ind2.weights[:mid], ind1.weights[mid:]])
    threshold = np.random.normal(((ind1.threshold + ind2.threshold) / 2)*np.random.random(), 0.1)
    n2 = Neuron(weights=weights, threshold=threshold)

    weights = (ind2.weights + ind1.weights)/2
    threshold = np.random.normal(((ind1.threshold + ind2.threshold) / 2)*np.random.random(), 0.1)
    n3 = Neuron(weights=weights, threshold=threshold)

    return [n1, n2, n3]


def mutation(individual):
    weights = individual.weights[:]
    weights = np.random.normal(weights)
    temp = weights[0]
    weights[0] = weights[1]
    weights[1] = temp
    threshold = individual.threshold + np.random.normal(0,1)*temp
    return Neuron(weights=weights, threshold=threshold)


def fitness(individual):
    return evaluateneuron(train_set, individual)

# start train
epochs = 100
population_size = 20
population = [Neuron(2) for i in range(population_size)]

fitness_history = []
def each_individual(individual, fit_val, epoch):
    if len(fitness_history) < epoch+1:
        fitness_history.append([])

    fitness_history[-1].append(fit_val)

result = ga(population=population, progenitors_amount=4, offsprings=4, objective='minimize', crossover=crossover, mutation=mutation, fitness=fitness, max_epochs=epochs, crossover_prob=0.98, mutation_prob=0.05, elitist=False, each_individual=each_individual, mutation_extra_individual=False, generational=False)
errors = evaluateneuron(test_set, result['individual'], debug=True,plt=plt)

print(f"Error (%): {(errors/len(dataset))*100}%")
print(f"Error (count): {errors}")
print(f"Weights: {result['individual'].weights}")
print(f"Threshold: {result['individual'].threshold}")

#TODO: this line is so wrong, I have to stop and understand the problem. Maybe later.
w1 = result['individual'].weights[0]
w2 = result['individual'].weights[1]
b = result['individual'].threshold

# Decision landscape
# w1*x1+ w2*x2 + b = 0
# w2*x2 = -b - w1*x1
# x2 = (-b -w1*x1)/w2

line_x_coords = np.array([0.5, 3.5])
line_y_coords = (-b - w1*line_x_coords)/w2

plt.plot(line_x_coords, line_y_coords)

# Plot data
plt.scatter(dataset[:6][:, 0], dataset[:6][:, 1], marker='^', c='red')
plt.scatter(dataset[6:][:, 0], dataset[6:][:, 1], marker='o', c='blue')
plt.show()

plt.title("Fitness History")
plt.plot(np.min(fitness_history, axis=1), label="Min perceptron error")
plt.plot(np.average(fitness_history, axis=1), label="Avg perceptron population error")
plt.legend()
plt.show()


