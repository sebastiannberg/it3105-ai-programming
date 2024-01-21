PLANT = "cournot" # Options: "bathtub", "cournot", "insert name model 3"

CONTROLLER = "classic" # Options: "classic", "ai"

ACTIVATION_FUNCTION = "sigmoid" # Options: "sigmoid", "tanh", "relu"
NUM_HIDDEN_LAYERS = 2
NEURONS_PER_LAYER = 128
INITIAL_WEIGHT_BIAS_RANGE = (0, 1) # TODO Check if this is correct interpretation of the task description

NUM_EPOCHS = 50
NUM_TIMESTEPS = 50
LEARNING_RATE = 0.01
DISTURBANCE_RANGE = (-0.01, 0.01)

CROSS_SECTIONAL_AREA_BATHTUB = 10
CROSS_SECTIONAL_AREA_DRAIN = CROSS_SECTIONAL_AREA_BATHTUB / 100
INITIAL_HEIGHT_BATHTUB_WATER = 5

MAXIMUM_PRICE_COURNOT = 3
MARGINAL_COST_COURNOT = 0.1
TARGET_COURNOT = 0.55
# TODO enforce q1 and q2 restriction in all code between 0 and 1
INITIAL_Q1_COURNOT = 0.4 # Options: 0 < q1 < 1
INITIAL_Q2_COURNOT = 0.7 # Options: 0 < q2 < 1

# TODO add at least two parameters for third plant
