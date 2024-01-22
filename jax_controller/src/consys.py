import numpy as np
import jax
import jax.numpy as jnp

# TODO wrong typing, should be base
from controllers.classic_pid_controller import ClassicPIDController
from visualization.plotting import Plotting


class Consys:

    # TODO wrong typing for controller, should be the base class
    def __init__(self, controller: ClassicPIDController, plant, learning_rate, disturbance_range) -> None:
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
            print(mse, gradient)
            # Update parameters based on the gradient
            params = self.update_params(params, gradient)

            if isinstance(self.controller, ClassicPIDController):
                self.plotting.add(epoch=epoch, mse=mse, kp=params[0], ki=params[1], kd=params[2])
            # TODO if isintance ai controller add only epoch and mse to plotting
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
        params -= self.learning_rate * gradient
        return params
