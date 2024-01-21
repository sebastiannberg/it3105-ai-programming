import sys
sys.path.append('jax_controller/') # To be able to import our own modules

from math import sqrt

import config.config as config

from consys import Consys
from plants.bathtub_model import BathtubModel
from controllers.classic_pid_controller import ClassicPIDController


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

if config.CONTROLLER == "classic":
    controller = ClassicPIDController()


consys = Consys(controller=controller,
                plant=plant,
                disturbance_range=config.DISTURBANCE_RANGE)
consys.run_system(epochs=config.NUM_EPOCHS,
                  timesteps=config.NUM_TIMESTEPS)
