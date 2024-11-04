import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Initial parameters
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Set up Streamlit form for city input
with st.form("city_input_form"):
    num_cities = st.number_input("Number of Cities", min_value=2, max_value=20, value=10)
    city_coords = {}

    for i in range(num_cities):
        col1, col2, col3 = st.columns([1, 1, 1])
        city_name = col1.text_input(f"City {i + 1}", f"City_{i + 1}")
        x_coord = col2.number_input(f"x-coordinate (City {i + 1})", key=f"x_{i}", step=1)
        y_coord = col3.number_input(f"y-coordinate (City {i + 1})", key=f"y_{i}", step=1)
        
        # Store each city's coordinates
        city_coords[city_name] = (x_coord, y_coord)

    # Button to submit the form
    submit_button = st.form_submit_button("Run Coordinates")

# Run the Genetic Algorithm if the form is submitted
if submit_button and len(city_coords) == num_cities:

    # Pastel Palette for Cities
    colors = sns.color_palette("pastel", len(city_coords))

    # City Icons
    city_icons = {
        "Gliwice": "♕",
        "Cairo": "♖",
        "Rome": "♗",
        "Krakow": "♘",
        "Paris": "♙",
        "Alexandria": "♔",
        "Berlin": "♚",
        "Tokyo": "♛",
        "Rio": "♜",
        "Budapest": "♝"
    }

    def initial_population(cities_list, n_population=250):
        population_perms = []
        possible_perms = list(permutations(cities_list))

        # Check if possible permutations are less than the population size
        if len(possible_perms) < n_population:
            # Sample with replacement if needed
            for _ in range(n_population):
                population_perms.append(list(random.choice(possible_perms)))
        else:
            # Sample without replacement if there are enough unique routes
            random_ids = random.sample(range(len(possible_perms)), n_population)
            for i in random_ids:
                population_perms.append(list(possible_perms[i]))
        return population_perms

    def dist_two_cities(city_1, city_2):
        city_1_coords = city_coords[city_1]
        city_2_coords = city_coords[city_2]
        return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2))

    def total_dist_individual(individual):
        total_dist = 0
        for i in range(len(individual)):
            if i == len(individual) - 1:
                total_dist += dist_two_cities(individual[i], individual[0])
            else:
                total_dist += dist_two_cities(individual[i], individual[i + 1])
        return total_dist

    def fitness_prob(population):
        total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - np.array(total_dist_all_individuals)
        population_fitness_sum = population_fitness.sum()
        return population_fitness / population_fitness_sum

    def roulette_wheel(population, fitness_probs):
        population_fitness_probs_cumsum = fitness_probs.cumsum()
        selected_individual_index = np.searchsorted(population_fitness_probs_cumsum, np.random.rand())
        return population[selected_individual_index]

    def crossover(parent_1, parent_2):
        cut = round(random.uniform(1, len(city_coords) - 1))
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not
