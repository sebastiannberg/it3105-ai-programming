from plants.base_plant import BasePlant

import jax.numpy as jnp
import jax


class PopulationModel(BasePlant):

    def __init__(self, init_plant_state: dict) -> None:
        if self.check_valid_init_state(init_plant_state):
            self.init_plant_state = init_plant_state
    
    def check_valid_init_state(self, init_plant_state):
        # TODO change
        required_keys = ("P", "r", "K", "target")
        if not all(key in init_plant_state for key in required_keys):
            missing_keys = [key for key in required_keys if key not in init_plant_state]
            raise ValueError(f"Missing keys in init_plant_state: {missing_keys}")
        return True

    def get_init_plant_state(self):
        return self.init_plant_state

    def update_plant(self, state, disturbance):
        state["r"] += state["control_signal"]
        jax.debug.print("{x}", x=state["r"])
        population_change = state["r"] * state["P"] * (1 - state["P"]/state["K"])
        population_change += disturbance
        state["P"] += population_change
        state["plant_output"] = state["P"]
        # Update current_error
        state["current_error"] = state["target"] - state["plant_output"]
        # jax.debug.print("{x}", x=state["P"])
        return state
