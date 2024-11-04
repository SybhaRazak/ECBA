import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns
import streamlit as st


st.set_page_config(
 page_title="Traveling Salesman"
)
st.header("Traveling Salesman", divider="gray")

# Initial parameters
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Default city names list (modifiable by the user)
default_city_names = ["Putrajaya", "Johor", "Perlis", "Negeri Sembilan", "Bedong", "Sungai Petani", "Alor Setar", "Skudai", "Tampoi", "Muar"]

# Set up Streamlit form for city names and coordinates input
city_coords = {}

st.write("Enter the city names and coordinates for 10 cities:")
for i, city in enumerate(default_city_names):  # Remove extra indentation here
    col1, col2, col3 = st.columns([1, 1, 1])
    city_name = col1.text_input(f"Enter name for City {i + 1}", city, key=f"city_name_{i}")  # Editable city name without "City" in label
    x_coord = col2.number_input(f"x-coordinate", key=f"x_{i}", value=random.randint(0, 15), step=1)
    y_coord = col3.number_input(f"y-coordinate", key=f"y_{i}", value=random.randint(0, 15), step=1)
    city_coords[city_name] = (x_coord, y_coord)

submit_button = st.button("Run Coordinates")

# Run the Genetic Algorithm if the form is submitted
if submit_button and len(city_coords) == len(default_city_names):

    # Pastel Palette for Cities
    colors = sns.color_palette("pastel", len(city_coords))

    # City Icons
    city_icons = {
        "Putrajaya": "♕",
        "Johor": "♖",
        "Perlis": "♗",
        "Negeri Sembilan": "♘",
        "Bedong": "♙",
        "Sungai Petani": "♔",
        "Alor Setar": "♚",
        "Skudai": "♛",
        "Tampoi": "♜",
        "Muar": "♝"
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
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    def mutation(offspring):
        index_1, index_2 = random.sample(range(len(city_coords)), 2)
        offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
        return offspring

    def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
        population = initial_population(cities_names, n_population)
        for _ in range(n_generations):
            fitness_probs = fitness_prob(population)
            parents = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child_1, child_2 = crossover(parents[i], parents[i + 1])
                    if random.random() < mutation_per:
                        child_1 = mutation(child_1)
                    if random.random() < mutation_per:
                        child_2 = mutation(child_2)
                    offspring.extend([child_1, child_2])
            population = parents + offspring
            fitness_probs = fitness_prob(population)
            sorted_indices = np.argsort(fitness_probs)[::-1]
            population = [population[i] for i in sorted_indices[:n_population]]
        return population

    # Run Genetic Algorithm
    cities_names = list(city_coords.keys())
    best_population = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)
    total_distances = [total_dist_individual(ind) for ind in best_population]
    best_index = np.argmin(total_distances)
    shortest_path = best_population[best_index]
    min_distance = total_distances[best_index]

    # Plotting the Best Route with Icons and Paths
    # Plot the Best Route with Icons and Paths
fig, ax = plt.subplots()

# Draw each city, its icon, and label
# Ensure color exists for each city
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i % len(colors)]  # Use modulo to avoid index errors
    icon = city_icons.get(city, "")  # Get the icon if it exists, or default to an empty string
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                textcoords='offset points')


    # Connect cities with opaque lines
    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

# Set figure size and display the plot in Streamlit
fig.set_size_inches(16, 12)
st.pyplot(fig)

# Replace best_mixed_offspring with best_population
total_dist_all_individuals = []
for i in range(0, n_population):
    total_dist_all_individuals.append(total_dist_individual(best_population[i]))

index_minimum = np.argmin(total_dist_all_individuals)
minimum_distance = min(total_dist_all_individuals)
minimum_distance

# shortest path
shortest_path = best_population[index_minimum]
shortest_path

x_shortest = []
y_shortest = []
for city in shortest_path:
    x_value, y_value = city_coords[city]
    x_shortest.append(x_value)
    y_shortest.append(y_value)

x_shortest.append(x_shortest[0])
y_shortest.append(y_shortest[0])

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i in range(len(x_shortest)):
    for j in range(i + 1, len(x_shortest)):
        ax.plot([x_shortest[i], x_shortest[j]], [y_shortest[i], y_shortest[j]], 'k-', alpha=0.09, linewidth=1)

plt.title(label="TSP Best Route Using GA",
          fontsize=25,
          color="k")

str_params = '\n'+str(n_generations)+' Generations\n'+str(n_population)+' Population Size\n'+str(crossover_per)+' Crossover\n'+str(mutation_per)+' Mutation'
plt.suptitle("Total Distance Travelled: "+
             str(round(minimum_distance, 3)) +
             str_params, fontsize=18, y = 1.047)

for i, txt in enumerate(shortest_path):
    ax.annotate(str(i+1)+ "- " + txt, (x_shortest[i], y_shortest[i]), fontsize= 20)

fig.set_size_inches(16, 12)
# plt.grid(color='k', linestyle='dotted')
st.pyplot(fig)

