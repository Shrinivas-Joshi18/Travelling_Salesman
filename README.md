# Travelling_Salesman
Travelling Salesman Problem Solver using Genetic Algorithm
This project implements a solution to the Travelling Salesman Problem (TSP) using a Genetic Algorithm. The goal is to find the shortest possible route that visits a set of cities exactly once and returns to the starting city.

# Overview
The project consists of Python code that utilizes the NumPy library for numerical operations and the Matplotlib library for visualization.

# Key Components:

**__**1. Calculate_total_distance(route, distance_matrix)**__**

  Calculates the total distance of a given route based on the provided distance matrix.

**__**2. generate_population(num_routes, num_cities)**__**

  Generates a random population of routes for the genetic algorithm.

**__**3. crossover(parent1, parent2)**__**

  Performs crossover between two parent routes to create a child route.

**__**4. mutate(route)**__**

  Applies mutation to a route, introducing small random changes.

**__**5. evolve(population, distance_matrix, num_generations)**__**

  Evolves the population using a genetic algorithm for a specified number of generations.

**__**6. visualize_route(route, cities)**__**

  Visualizes the TSP route using Matplotlib.

  **Visualize the best route**

  visualize_route(best_route, cities)


# How to Run

Make sure you have Python installed on your machine.

Install the required libraries using 
                    
    pip install numpy 
    
    pip install matplotlib

Run the script using python **filename.py**, replacing **filename.py** with the actual name of your Python script.

# Example parameters
    num_cities = 10
    num_routes = 50
    num_generations = 500

    # Randomly generate city coordinates and calculate the distance matrix
    cities = np.random.rand(num_cities, 2)
    distance_matrix = np.linalg.norm(cities[:, np.newaxis, :] - cities, axis=2)

    # Generate an initial population of routes
    population = generate_population(num_routes, num_cities)

    # Evolve the population to find the best route
    best_route = evolve(population, distance_matrix, num_generations)

    # Print results
    print("Best route:", best_route)
    print("Total distance:", calculate_total_distance(best_route, distance_matrix))

# License

This project is licensed under the **MIT License** - see the LICENSE file for details.

# Acknowledgments

Inspired by the **Genetic Algorithm** and its application to combinatorial optimization problems.

The NumPy and Matplotlib libraries were instrumental in the implementation.
