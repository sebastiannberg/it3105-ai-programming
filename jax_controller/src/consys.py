import numpy as np
import jax
import jax.numpy as jnp

# TODO wrong typing, should be base
from controllers.classic_pid_controller import ClassicPIDController


class Consys:

    def __init__(self, controller: ClassicPIDController, plant, epochs, timesteps, disturbance_range) -> None:
        self.controller = controller
        self.plant = plant
        self.epochs = epochs
        self.timesteps = timesteps
        self.disturbance_range = disturbance_range
    
    def mse(self, controller_params, disturbance, compute_control_signal: callable, target, plant_history):
        return jnp.mean([(target - plant_output(controller_params, disturbance[t], compute_control_signal))**2 for t, plant_output in enumerate(plant_history)])

    def run(self):
        print("--- Running Consys ---")
        # Init controller parameters
        self.controller.init_params()
        for _ in range(self.epochs):
            # Init any other controller variables (error hisotry etc.)
            self.controller.init_error_history()
            self.controller.init_plant_history()
            # Reset plant to initial state
            self.plant.reset_plant()
            # Generate disturbance vector
            disturbance_vector = np.random.uniform(*self.disturbance_range, size=self.timesteps)
            for t in range(self.timesteps):
                # update plant
                # update controller
                # save the error E  (and plant output) for this timestep in an error history (maybe inside controller object)
                plant_output = self.controller.save_plant_output(self.plant.compute_plant_output(self.controller.params, disturbance_vector[t], self.controller.compute_control_signal))
                print(plant_output)
                self.controller.save_error(self.plant.target - plant_output)
            # compute MSE over the error history
            mse = self.mse(self.controller.params, disturbance_vector, self.controller.compute_control_signal, self.plant.target, self.controller.plant_history)
            print(mse)
            # TODO visualization here?
            # compute the gradients (derivative of MSE over params (three k values or weights))
            loss_gradient = jax.grad(self.mse)
            # Update params based on gradient
            self.controller.get_updated_params()
            self.controller.set_params()
