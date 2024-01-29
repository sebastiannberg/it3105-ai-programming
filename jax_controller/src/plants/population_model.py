from plants.base_plant import BasePlant

import jax.numpy as jnp
import jax


# Lotka-Volterra Equations
class PopulationModel(BasePlant):

    def __init__(self, init_plant_state: dict) -> None:
        if self.check_valid_init_state(init_plant_state):
            self.init_plant_state = init_plant_state
    
    def check_valid_init_state(self, init_plant_state):
        required_keys = ("P", "target", "PP", "prey_growth_rate", "predation_rate", "predator_mortality_rate", "predator_growth_rate")
        if not all(key in init_plant_state for key in required_keys):
            missing_keys = [key for key in required_keys if key not in init_plant_state]
            raise ValueError(f"Missing keys in init_plant_state: {missing_keys}")
        return True

    def get_init_plant_state(self):
        return self.init_plant_state

    def update_plant(self, state, disturbance):
        # Update prey population
        prey_population_change = state["prey_growth_rate"] * state["P"] - state["predation_rate"] * state["P"] * state["PP"]
        prey_population_change += state["control_signal"]
        state["P"] += prey_population_change
        # Update predator population
        predator_population_change = state["predator_growth_rate"] * state["P"] * state["PP"] - state["predator_mortality_rate"] * state["PP"]
        predator_population_change += disturbance
        state["PP"] += predator_population_change
        # Set plant output
        state["plant_output"] = state["P"]
        # Update current_error
        state["current_error"] = state["target"] - state["plant_output"]
        # jax.debug.print("Prey Population: {x}", x=state["P"])
        # jax.debug.print("Predator Population: {x}", x=state["PP"])
        # jax.debug.print("Error: {x}\n", x=state["current_error"])
        return state
