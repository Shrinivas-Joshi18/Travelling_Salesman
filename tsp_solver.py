import numpy as np
import matplotlib.pyplot as plt

def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i + 1]]
    total_distance += distance_matrix[route[-1], route[0]]
    return total_distance

def generate_population(num_routes, num_cities):
    population = []
    for _ in range(num_routes):
        route = np.random.permutation(num_cities)
        population.append(route)
    return population

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child = np.hstack((parent1[:crossover_point], [city for city in parent2 if city not in parent1[:crossover_point]]))
    return child

def mutate(route):
    mutation_point1, mutation_point2 = np.random.choice(len(route), 2, replace=False)
    route[mutation_point1], route[mutation_point2] = route[mutation_point2], route[mutation_point1]
    return route

def evolve(population, distance_matrix, num_generations):
    for generation in range(num_generations):
        population = sorted(population, key=lambda x: calculate_total_distance(x, distance_matrix))
        new_population = population[:2]
        while len(new_population) < len(population):
            parents = np.random.choice(np.concatenate(population[:10]), 2, replace=False)
            parent1, parent2 = population[parents[0]], population[parents[1]]
            child = crossover(parent1, parent2)
            if np.random.rand() < 0.1:
                child = mutate(child)
            new_population.append(child)
        population = new_population
    return population[0]

def visualize_route(route, cities):
    x = [cities[i][0] for i in route]
    y = [cities[i][1] for i in route]
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y, marker='o')
    plt.show()


# Example usage
num_cities = 10
cities = np.random.rand(num_cities, 2)
distance_matrix = np.linalg.norm(cities[:, np.newaxis, :] - cities, axis=2)

num_routes = 50
num_generations = 500
population = generate_population(num_routes, num_cities)

best_route = evolve(population, distance_matrix, num_generations)
print("Best route:", best_route)
print("Total distance:", calculate_total_distance(best_route, distance_matrix))

visualize_route(best_route, cities)