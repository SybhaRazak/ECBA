import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from itertools import permutations
from random import shuffle, sample
import random

# Streamlit UI for Input Form
st.title("Traveling Salesman Problem using Genetic Algorithm")

# Initialize empty lists for city names and coordinates
cities_names = []
x = []
y = []

# Input form for cities
st.subheader("Enter City Names and Coordinates")
with st.form("city_form"):
    city_name = st.text_input("City Name")
    city_x = st.number_input("X Coordinate", value=0.0)
    city_y = st.number_input("Y Coordinate", value=0.0)
    add_city = st.form_submit_button("Add City")

    # Add city to list if button is clicked
    if add_city and city_name:
        cities_names.append(city_name)
        x.append(city_x)
        y.append(city_y)
        st.write(f"Added City: {city_name} at ({city_x}, {city_y})")

# Display current list of cities
st.subheader("Current List of Cities")
if cities_names:
    st.write("Cities:", cities_names)
    st.write("Coordinates:", list(zip(x, y)))

# Define parameters for the genetic algorithm
n_population = st.number_input("Population Size", value=250)
crossover_per = st.slider("Crossover Percentage", 0.0, 1.0, 0.8)
mutation_per = st.slider("Mutation Percentage", 0.0, 1.0, 0.2)
n_generations = st.number_input("Number of Generations", value=200)

# Pastel color palette
colors = sns.color_palette("pastel", len(cities_names))

# Define city coordinates and icons after adding all cities
if cities_names:
    city_coords = dict(zip(cities_names, zip(x, y)))
    city_icons = {city: "♔" for city in cities_names}  # Use ♔ for all cities by default or customize as needed

    # Visualize cities
    fig, ax = plt.subplots()
    ax.grid(False)

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons[city]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    # Implementing the Genetic Algorithm
    # (Code remains the same as your genetic algorithm logic with fitness function, selection, crossover, mutation, etc.)
    # Note: Make sure to replace `city_coords` and `cities_names` variables with dynamically updated ones

    # Run the genetic algorithm
    best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

    # Calculate shortest path and visualize
    total_dist_all_individuals = []
    for i in range(0, n_population):
        total_dist_all_individuals.append(total_dist_individual(best_mixed_offspring[i]))

    index_minimum = np.argmin(total_dist_all_individuals)
    minimum_distance = min(total_dist_all_individuals)
    st.write(f"Minimum Distance: {minimum_distance}")

    # Shortest path
    shortest_path = best_mixed_offspring[index_minimum]
    st.write(f"Shortest Path: {shortest_path}")

    # Visualize the shortest path
    x_shortest, y_shortest = [], []
    for city in shortest_path:
        city_x, city_y = city_coords[city]
        x_shortest.append(city_x)
        y_shortest.append(city_y)
    x_shortest.append(x_shortest[0])
    y_shortest.append(y_shortest[0])

    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    plt.legend()

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

    plt.title("TSP Best Route Using GA", fontsize=25)
    str_params = f"\n{n_generations} Generations\n{n_population} Population Size\n{crossover_per} Crossover\n{mutation_per} Mutation"
    plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}{str_params}", fontsize=18, y=1.047)

    for i, txt in enumerate(shortest_path):
        ax.annotate(f"{i + 1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)
else:
    st.write("Add at least one city to visualize and calculate the TSP solution.")
