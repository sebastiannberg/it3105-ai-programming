PLANT = "bathtub" # Options: "bathtub", "cournot", "insert name model 3"

CONTROLLER = "classic" # Options: "classic", "ai"

ACTIVATION_FUNCTION = "sigmoid" # Options: "sigmoid", "tanh", "relu"
NUM_HIDDEN_LAYERS = 2
NEURONS_PER_LAYER = 128
INITIAL_WEIGHT_BIAS_RANGE = (0, 1) # TODO Check if this is correct interpretation of the task description

NUM_EPOCHS = 5
NUM_TIMESTEPS = 50
LEARNING_RATE = 0.01
DISTURBANCE_RANGE = (-0.01, 0.01) # TODO Check if this is correct interpretation and what should the values be?

CROSS_SECTIONAL_AREA_BATHTUB = 10
CROSS_SECTIONAL_AREA_DRAIN = 2
INITIAL_HEIGHT_BATHTUB_WATER = 3

MAXIMUM_PRICE_COURNOT = 100
MARGINAL_COST_COURNOT = 20

# TODO add at least two parameters for third plant
