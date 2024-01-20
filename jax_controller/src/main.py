import sys
sys.path.append('jax_controller/') # To be able to import our own modules

import config.config as config

from consys import Consys
from plants.bathtub_model import BathtubModel
from controllers.classic_pid_controller import ClassicPIDController


if config.PLANT == "bathtub":
    plant = BathtubModel(A=config.CROSS_SECTIONAL_AREA_BATHTUB, C=config.CROSS_SECTIONAL_AREA_DRAIN, target=config.INITIAL_HEIGHT_BATHTUB_WATER)

if config.CONTROLLER == "classic":
    controller = ClassicPIDController(learning_rate=config.LEARNING_RATE)


consys = Consys(controller=controller,
                plant=plant, 
                epochs=config.NUM_EPOCHS, 
                timesteps=config.NUM_TIMESTEPS,
                disturbance_range=config.DISTURBANCE_RANGE)
consys.run()
