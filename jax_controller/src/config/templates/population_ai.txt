PLANT = "population" # Options: "bathtub", "cournot", "population"

CONTROLLER = "ai" # Options: "classic", "ai"

INITIAL_KP = 0.1 # Options: float
INITIAL_KI = 0.1 # Options: float
INITIAL_KD = 0.1 # Options: float

ACTIVATION_FUNCTION = "relu" # Options: "sigmoid", "tanh", "relu"
NUM_HIDDEN_LAYERS = 3 # Options: 0, 1, 2, 3, 4, 5
NEURONS_PER_LAYER = [64, 64, 32] # Options: Length must correspond with NUM_HIDDEN_LAYERS
INITIAL_WEIGHT_BIAS_RANGE = (-0.1, 0.1)

NUM_EPOCHS = 50
NUM_TIMESTEPS = 50
LEARNING_RATE = 0.0001
DISTURBANCE_RANGE = (-1, 1)

CROSS_SECTIONAL_AREA_BATHTUB = 10
CROSS_SECTIONAL_AREA_DRAIN = CROSS_SECTIONAL_AREA_BATHTUB / 100
INITIAL_HEIGHT_BATHTUB_WATER = 5

MAXIMUM_PRICE_COURNOT = 3
MARGINAL_COST_COURNOT = 0.1
TARGET_COURNOT = 0.55
INITIAL_Q1_COURNOT = 0.4 # Options: 0 < Q1 < 1
INITIAL_Q2_COURNOT = 0.7 # Options: 0 < Q2 < 1

# Based on Lotka-Volterra equations
INITIAL_POPULATION = 90
TARGET_POPULATION = 100 # Equilibrium: set this to predator_mortality_rate / predator_growth_rate
INITIAL_PREDATOR_POPULATION = 30 # Usually less than prey population
PREY_GROWTH_RATE = 0.2
PREDATION_RATE = 0.01
PREDATOR_GROWTH_RATE = 0.002
PREDATOR_MORTALITY_RATE = 0.2
