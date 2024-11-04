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

# Create a form for entering city names and coordinates
with st.form("city_input_form"):
    num_cities = st.number_input("Number of Cities", min_value=2, max_value=20, value=10)
    city_coords = {}

    for i in range(num_cities):
        col1, col2, col3 = st.columns([1, 1, 1])
        city_name = col1.text_input(f"City {i + 1}", f"City_{i + 1}")
        x_coord = col2.number_input(f"x-coordinate (City {i + 1})", key=f"x_{i}", step=1.0)
        y_coord = col3.number_input(f"y-coordinate (City {i + 1})", key=f"y_{i}", step=1.0)
        
        # Store each city's coordinates
        city_coords[city_name] = (x_coord, y_coord)

    # Button to submit the form
    submit_button = st.form_submit_button("Submit Coordinates and Run GA")

# Run the Genetic Algorithm if the form is submitted
if submit_button and len(city_coords) == num_cities:
    
    # Rest of your code remains unchanged for GA implementation
    # ...

    # Plotting the Best Route (This part remains the same)
    fig, ax = plt.subplots()
    x_coords, y_coords = zip(*[city_coords[city] for city in shortest_path])
    x_coords += (x_coords[0],)  # To return to the start
    y_coords += (y_coords[0],)
    ax.plot(x_coords, y_coords, '--go', label='Best Route', linewidth=2.5)

    # Draw cities and annotate
    for i, (city, (x, y)) in enumerate(city_coords.items()):
        color = colors[i]
        ax.scatter(x, y, c=[color], s=1200, zorder=2)
        ax.annotate(f"{i + 1}- {city}", (x, y), fontsize=14, ha='center', va='center', zorder=3)

    ax.set_title(f"TSP Best Route Using GA\nTotal Distance: {min_distance:.2f}\nGenerations: {n_generations} | Population: {n_population}")
    fig.set_size_inches(12, 8)
    st.pyplot(fig)
