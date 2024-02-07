import numpy as np
import jax
import jax.numpy as jnp

from plants.base_plant import BasePlant
from controllers.base_controller import BaseController
from controllers.classic_pid_controller import ClassicPIDController
from controllers.ai_pid_controller import AIPIDController
from visualization.plotting import Plotting


class Consys:

    def __init__(self, controller: BaseController, plant: BasePlant, learning_rate, disturbance_range) -> None:
        self.controller = controller
        self.plant = plant
        self.learning_rate = learning_rate
        self.disturbance_range = disturbance_range
        self.plotting = Plotting()

    def run_system(self, epochs, timesteps):
        gradient_function = jax.value_and_grad(self.run_system_one_epoch)
        # Init params
        params = self.controller.init_params()
        for epoch in range(epochs):
            # Init system state (error history and plant state)
            state = {
                "error_history": [],
                # Assuming plant start in ideal state
                "current_error": 0.0,
                # Set control signal to 0 initially, it is updated first hand
                "control_signal": 0.0,
                # Set plant output to 0 initially, it is updated first hand
                "plant_output": 0.0
            }
            init_plant_state = self.plant.get_init_plant_state()
            # Add init plant state to the system state dictionary
            state.update(init_plant_state)

            # Executing run_system_one_epoch via jax gradient function
            mse, gradient = gradient_function(params, state, timesteps)
            print("==MSE AND GRADIENT==\n", mse, "\n", gradient, "\n\n\n")
            # Update parameters based on the gradient
            params = self.update_params(params, gradient)

            if isinstance(self.controller, ClassicPIDController):
                self.plotting.add(epoch=epoch, mse=mse, kp=params[0], ki=params[1], kd=params[2])
            elif isinstance(self.controller, AIPIDController):
                self.plotting.add(epoch=epoch, mse=mse)

        self.plotting.plot_mse_and_params()

    def run_system_one_epoch(self, params, state, timesteps):
        # Generate disturbance vector
        disturbance_vector = np.random.uniform(*self.disturbance_range, size=timesteps)
        for t in range(timesteps):
            state = self.run_system_one_timestep(params, state, disturbance_vector[t])
        # Return the MSE over the error history
        mse = jnp.mean(jnp.array([error**2 for error in state["error_history"]]))
        return mse

    def run_system_one_timestep(self, params, state, disturbance):
        state = self.run_plant_one_timestep(state, disturbance)
        state = self.run_controller_one_timestep(params, state)
        # Save error in error history
        state["error_history"].append(state["current_error"])
        return state

    def run_plant_one_timestep(self, state, disturbance):
        # Update plant
        state = self.plant.update_plant(state, disturbance)
        return state

    def run_controller_one_timestep(self, params, state):
        # Update controller
        state = self.controller.update_controller(params, state)
        return state

    def update_params(self, params, gradient):
        # Gradient descent
        if isinstance(self.controller, ClassicPIDController):
            params -= self.learning_rate * gradient
            return params
        elif isinstance(self.controller, AIPIDController):
            new_params = []
            for layer, layer_gradient in zip(params, gradient):
                layer -= self.learning_rate * layer_gradient
                new_params.append(layer)
            return new_params
