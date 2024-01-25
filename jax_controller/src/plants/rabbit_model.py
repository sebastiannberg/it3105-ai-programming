from plants.base_plant import BasePlant
import jax.numpy as jnp
import jax

class RabbitPopulationModel(BasePlant):

    def __init__(self, init_plant_state: dict) -> None:
        if self.check_valid_init_state(init_plant_state):
            self.init_plant_state = init_plant_state

    def check_valid_init_state(self, init_plant_state):
        required_keys = ("P", "K", "r", "PC", "target")
        if not all(key in init_plant_state for key in required_keys):
            missing_keys = [key for key in required_keys if key not in init_plant_state]
            raise ValueError(f"Missing keys in init_plant_state: {missing_keys}")
        return True

    def get_init_plant_state(self):
        return self.init_plant_state

    def update_plant(self, state, disturbance):
        # Logistic Growth Model with Control and Disturbance
        growth = state["r"] * state["P"] * (1 - state["P"] / state["K"])
        predator_effect = state["PC"] + disturbance
        population_change = growth - predator_effect
        state["P"] += population_change
        state["plant_output"] = state["P"]
        state["current_error"] = state["target"] - state["plant_output"]
        return state
