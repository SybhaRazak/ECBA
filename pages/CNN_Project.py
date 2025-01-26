import streamlit as st
import random
import time
import matplotlib.pyplot as plt
import numpy as np

# Genetic Algorithm Parameters
POP_SIZE = 100
MUT_RATE = 0.2
TARGET_FITNESS = 0.9580
MAX_GENERATIONS = 1000

# Initialize Population
def initialize_pop():
    predefined_learning_rates = [0.001, 0.005, 0.01]
    population = []
    for i in range(POP_SIZE):
        individual = {
            "learning_rate": random.choice(predefined_learning_rates),
            "batch_size": random.randint(16, 128),
            "hidden_layers": [random.randint(10, 100) for _ in range(random.randint(1, 5))],
            "activation": random.choice(['relu', 'sigmoid', 'tanh']),
            "epochs": random.randint(10, 100)
        }
        population.append(individual)
    return population

# Fitness Function (Simulating with Random Values)
def fitness_cal(individual):
    accuracy = random.uniform(0.90, 1.0)  # Simulating accuracy
    return accuracy

# Selection
def selection(population):
    fitness_scores = [(ind, fitness_cal(ind)) for ind in population]
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    return [ind[0] for ind in fitness_scores[:POP_SIZE // 2]]

# Crossover
def crossover(parent1, parent2):
    child = {
        "learning_rate": random.choice([parent1["learning_rate"], parent2["learning_rate"]]),
        "batch_size": random.choice([parent1["batch_size"], parent2["batch_size"]]),
        "hidden_layers": parent1["hidden_layers"][:len(parent1["hidden_layers"]) // 2] +
                         parent2["hidden_layers"][len(parent2["hidden_layers"]) // 2:],
        "activation": random.choice([parent1["activation"], parent2["activation"]]),
        "epochs": random.choice([parent1["epochs"], parent2["epochs"]])
    }
    return child

# Mutation
def mutate(individual):
    if random.random() < MUT_RATE:
        individual["batch_size"] = random.randint(16, 128)
    if random.random() < MUT_RATE:
        individual["hidden_layers"] = [random.randint(10, 100) for _ in range(random.randint(1, 5))]
    if random.random() < MUT_RATE:
        individual["activation"] = random.choice(['relu', 'sigmoid', 'tanh'])
    if random.random() < MUT_RATE:
        individual["epochs"] = random.randint(10, 100)
    return individual

# Genetic Algorithm Main Loop
def genetic_algorithm():
    population = initialize_pop()
    generation = 1
    best_fitness_values = []
    found = False

    while not found and generation <= MAX_GENERATIONS:
        selected = selection(population)
        new_generation = []
        for _ in range(POP_SIZE):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_generation.append(child)

        population = new_generation
        best_individual = max(population, key=lambda ind: fitness_cal(ind))
        best_fitness = fitness_cal(best_individual)
        best_fitness_values.append(best_fitness)

        # Update Streamlit progress
        st.write(f"Generation {generation}, Best Fitness: {best_fitness:.6f}")
        generation += 1

        if best_fitness >= TARGET_FITNESS:
            found = True

    return best_fitness_values, best_individual, generation

# Visualization Function
def visualize_fitness(fitness_values):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_values) + 1), fitness_values, marker='o', linestyle='-', color='b', label='Best Fitness')
    plt.title("Genetic Algorithm Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Value")
    plt.grid()
    plt.legend()
    st.pyplot(plt)

# Streamlit Interface
def main():
    st.title("Genetic Algorithm for CNN Hyperparameter Optimization")
    st.write("This application uses a genetic algorithm to optimize CNN hyperparameters.")

    # Display Parameters
    st.sidebar.header("Genetic Algorithm Parameters")
    pop_size = st.sidebar.slider("Population Size", min_value=10, max_value=200, value=POP_SIZE, step=10)
    mut_rate = st.sidebar.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=MUT_RATE, step=0.05)
    target_fitness = st.sidebar.number_input("Target Fitness", value=TARGET_FITNESS, step=0.001, format="%.3f")

    # Run Genetic Algorithm
    if st.button("Run Genetic Algorithm"):
        st.write("Running Genetic Algorithm...")
        best_fitness_values, best_individual, generations = genetic_algorithm()

        # Results
        st.success(f"Optimization Complete in {generations} generations!")
        st.write(f"**Best Individual:** {best_individual}")
        st.write(f"**Best Fitness Achieved:** {best_fitness_values[-1]:.6f}")

        # Highlight Best Hyperparameters
        st.subheader("Optimal Hyperparameters")
        st.write(f"- **Learning Rate:** {best_individual['learning_rate']}")
        st.write(f"- **Batch Size:** {best_individual['batch_size']}")
        st.write(f"- **Hidden Layers:** {best_individual['hidden_layers']}")
        st.write(f"- **Activation Function:** {best_individual['activation']}")
        st.write(f"- **Epochs:** {best_individual['epochs']}")

        # Visualize Fitness
        st.subheader("Fitness Over Generations")
        visualize_fitness(best_fitness_values)

if __name__ == "__main__":
    main()
