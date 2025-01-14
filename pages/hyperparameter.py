import streamlit as st
import numpy as np
import pandas as pd
import random

# Load the dataset
dataset_path = '/content/hyperparameter_dataset.csv'
data = pd.read_csv(dataset_path)

# Extract relevant columns
hyperparameters = data[['learning_rate', 'batch_size', 'epochs', 'hidden_layer_sizes', 'activation']]
accuracy = data['accuracy']

# Genetic Algorithm Parameters
POPULATION_SIZE = 20
GENERATIONS = 50
MUTATION_RATE = 0.1
TARGET_FITNESS = 0.90  # Define target fitness for early stopping

# Convert categorical data to numerical values
def encode_activation(activation):
    mapping = {'relu': 0, 'tanh': 1}
    return mapping.get(activation, -1)

data['activation_encoded'] = data['activation'].apply(encode_activation)

# Define fitness function (maximize accuracy)
def fitness_function(individual):
    lr, batch, epochs, hidden, activation = individual
    subset = data[(data['learning_rate'] == lr) & 
                  (data['batch_size'] == batch) & 
                  (data['epochs'] == epochs) & 
                  (data['hidden_layer_sizes'] == hidden) & 
                  (data['activation_encoded'] == activation)]
    if not subset.empty:
        return subset['accuracy'].values[0]
    return 0  # Return 0 if no matching combination is found

# Initialize population
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = [
            random.choice(data['learning_rate'].unique()),
            random.choice(data['batch_size'].unique()),
            random.choice(data['epochs'].unique()),
            random.choice(data['hidden_layer_sizes'].unique()),
            random.choice(data['activation_encoded'].unique())
        ]
        population.append(individual)
    return population

# Perform crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Perform mutation
def mutate(individual):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(individual) - 1)
        if idx == 0:
            individual[idx] = random.choice(data['learning_rate'].unique())
        elif idx == 1:
            individual[idx] = random.choice(data['batch_size'].unique())
        elif idx == 2:
            individual[idx] = random.choice(data['epochs'].unique())
        elif idx == 3:
            individual[idx] = random.choice(data['hidden_layer_sizes'].unique())
        elif idx == 4:
            individual[idx] = random.choice(data['activation_encoded'].unique())

# Run Genetic Algorithm with Target Fitness
def genetic_algorithm_with_target():
    global POPULATION_SIZE, GENERATIONS, TARGET_FITNESS, MUTATION_RATE  # Declare global variables before using them
    
    population = initialize_population()
    for generation in range(GENERATIONS):
        # Evaluate fitness
        fitness_scores = [fitness_function(ind) for ind in population]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]

        # Check if target fitness is achieved
        best_fitness = max(fitness_scores)
        if best_fitness >= TARGET_FITNESS:
            return sorted_population[0], best_fitness, generation + 1  # Return early if target fitness is reached

        # Select top individuals
        next_generation = sorted_population[:POPULATION_SIZE // 2]

        # Crossover
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(next_generation[:POPULATION_SIZE // 4], 2)
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([child1, child2])

        # Mutation
        for individual in next_generation:
            mutate(individual)

        population = next_generation

    # If target fitness is not found within GENERATIONS
    best_individual = max(population, key=fitness_function)
    return best_individual, fitness_function(best_individual), GENERATIONS

# Streamlit app
st.title("Genetic Algorithm for Hyperparameter Optimization")
st.write("Use this app to optimize the hyperparameters of a model using a Genetic Algorithm.")

# Input parameters
target_accuracy = st.slider("Target Accuracy", 0.0, 1.0, 0.90, step=0.01)
population_size = st.slider("Population Size", 10, 100, POPULATION_SIZE)
generations = st.slider("Generations", 10, 100, GENERATIONS)
mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, MUTATION_RATE, step=0.01)

# Run Genetic Algorithm when the button is clicked
if st.button("Run Optimization"):
    st.write("Running genetic algorithm...")
    
    # Update the parameters based on the user input
    POPULATION_SIZE = population_size
    GENERATIONS = generations
    TARGET_FITNESS = target_accuracy
    MUTATION_RATE = mutation_rate
    
    # Execute the GA
    best_solution, best_accuracy, generation = genetic_algorithm_with_target()

    # Display results
    st.write(f"Optimization completed in {generation} generations.")
    st.write(f"Best Hyperparameters: {best_solution}")
    st.write(f"Best Accuracy: {best_accuracy:.4f}")
