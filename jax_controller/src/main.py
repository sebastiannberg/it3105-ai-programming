import sys
sys.path.append('jax_controller/')  # Adjust the path accordingly

from config.config import LEARNING_RATE
from controllers.base_controller import BaseController

base_controller = BaseController(learning_rate=LEARNING_RATE)

base_controller.learning_rate