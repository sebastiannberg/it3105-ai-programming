PLANT = "bathtub" # Options: "bathtub", "cournot", "insert name model 3"

CONTROLLER = "ai" # Options: "classic", "ai"

ACTIVATION_FUNCTION = "sigmoid" # Options: "sigmoid", "tanh", "relu"
NUM_HIDDEN_LAYERS = 2 # Options: 0, 1, 2, 3, 4, 5
# TODO kanskje all for mane neurons
NEURONS_PER_LAYER = [124, 64] # Options: Must correspond with NUM_HIDDEN_LAYERS
INITIAL_WEIGHT_BIAS_RANGE = (-1, 1) # TODO Check if this is correct interpretation of the task description

NUM_EPOCHS = 10
NUM_TIMESTEPS = 25
LEARNING_RATE = 0.1
DISTURBANCE_RANGE = (-0.01, 0.01)

CROSS_SECTIONAL_AREA_BATHTUB = 10
CROSS_SECTIONAL_AREA_DRAIN = CROSS_SECTIONAL_AREA_BATHTUB / 100
INITIAL_HEIGHT_BATHTUB_WATER = 5

MAXIMUM_PRICE_COURNOT = 3
MARGINAL_COST_COURNOT = 0.1
TARGET_COURNOT = 0.55
INITIAL_Q1_COURNOT = 0.4 # Options: 0 < Q1 < 1
INITIAL_Q2_COURNOT = 0.7 # Options: 0 < Q2 < 1

# TODO add at least two parameters for third plant
