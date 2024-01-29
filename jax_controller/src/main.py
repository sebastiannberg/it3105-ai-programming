import sys
sys.path.append('jax_controller/') # To be able to import our own modules

from math import sqrt

import config.config as config

from consys import Consys
from plants.bathtub_model import BathtubModel
from plants.cournot_model import CournotModel
from plants.population_model import PopulationModel
from controllers.classic_pid_controller import ClassicPIDController
from controllers.ai_pid_controller import AIPIDController


if config.PLANT == "bathtub":
    init_plant_state = {
        "A": config.CROSS_SECTIONAL_AREA_BATHTUB,
        "C": config.CROSS_SECTIONAL_AREA_DRAIN,
        "H": config.INITIAL_HEIGHT_BATHTUB_WATER,
        "V": sqrt(2 * 9.81 * config.INITIAL_HEIGHT_BATHTUB_WATER),
        "Q": sqrt(2 * 9.81 * config.INITIAL_HEIGHT_BATHTUB_WATER) * config.CROSS_SECTIONAL_AREA_DRAIN,
        "target": config.INITIAL_HEIGHT_BATHTUB_WATER
    }
    plant = BathtubModel(init_plant_state=init_plant_state)
elif config.PLANT == "cournot":
    init_plant_state = {
        "max_price": config.MAXIMUM_PRICE_COURNOT,
        "marginal_cost": config.MARGINAL_COST_COURNOT,
        "target": config.TARGET_COURNOT,
        "q1": config.INITIAL_Q1_COURNOT,
        "q2": config.INITIAL_Q2_COURNOT,
        "q": config.INITIAL_Q1_COURNOT + config.INITIAL_Q2_COURNOT,
        "price": config.MAXIMUM_PRICE_COURNOT - (config.INITIAL_Q1_COURNOT + config.INITIAL_Q2_COURNOT),
        "producer_one_profit": config.INITIAL_Q1_COURNOT * (config.MAXIMUM_PRICE_COURNOT - (config.INITIAL_Q1_COURNOT + config.INITIAL_Q2_COURNOT) - config.MARGINAL_COST_COURNOT)
    }
    plant = CournotModel(init_plant_state=init_plant_state)
elif config.PLANT == "population":
    init_plant_state = {
        "P": config.INITIAL_POPULATION,
        "target": config.TARGET_POPULATION,
        "PP": config.INITIAL_PREDATOR_POPULATION,
        "prey_growth_rate": config.PREY_GROWTH_RATE,
        "predation_rate": config.PREDATION_RATE,
        "predator_mortality_rate": config.PREDATOR_MORTALITY_RATE,
        "predator_growth_rate": config.PREDATOR_GROWTH_RATE
    }
    plant = PopulationModel(init_plant_state=init_plant_state)

if config.CONTROLLER == "classic":
    controller = ClassicPIDController(kp=config.INITIAL_KP,
                                      ki=config.INITIAL_KI,
                                      kd=config.INITIAL_KD)
elif config.CONTROLLER == "ai":
    controller = AIPIDController(activation_function=config.ACTIVATION_FUNCTION,
                                 num_hidden_layers=config.NUM_HIDDEN_LAYERS,
                                 neurons_per_layer=config.NEURONS_PER_LAYER,
                                 initial_weight_bias_range=config.INITIAL_WEIGHT_BIAS_RANGE)

consys = Consys(controller=controller,
                plant=plant,
                learning_rate=config.LEARNING_RATE,
                disturbance_range=config.DISTURBANCE_RANGE)
consys.run_system(epochs=config.NUM_EPOCHS,
                  timesteps=config.NUM_TIMESTEPS)
