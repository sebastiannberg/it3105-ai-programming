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
    
    def run_system(self, epochs):
        gradient_function = jax.value_and_grad(self.run_system_one_epoch)
        # Init params
        params = self.controller.init_params()

        for _ in range(epochs):
            # Init state (error history and plant state)
            state = {
                "current_error": 0,
                "error_history": [],
                "control_signal": 0
            }
            # Executing run_system_one_epoch via jax gradient function
            mse, gradient = gradient_function(params, state)

            # update_params(params, gradient)
    
    def run_system_one_epoch(self, params, state):
        # Generate disturbance vector
        disturbance_vector = np.random.uniform(*self.disturbance_range, size=self.timesteps)
        for t in range(self.timesteps):
            self.run_system_one_timestep(params, state, disturbance_vector[t])

    def run_system_one_timestep(self, params, state, disturbance):
        self.run_plant_one_timestep(params, state, disturbance)
        self.run_controller_one_timestep(params, state)

    def run_plant_one_timestep(self, params, state, disturbance):
        pass

    def run_controller_one_timestep(self, params, state):
        pass
    
    # # TODO edit this, now we have state history
    # def mse(self, controller_params, error_history, disturbance_vector):
    #     # Construct errors array
    #     errors = []
    #     for t in range(1, len(error_history) - 1):
    #         # TODO check that [:t] is correct
    #         errors.append((self.plant.target - self.plant.compute_plant_output(controller_params, disturbance_vector[t-1], error_history[:t], self.controller.compute_control_signal))**2)
    #     return jnp.mean(jnp.array(errors))

    # def run(self):
    #     print("--- Running Consys ---")
    #     # Init controller parameters
    #     self.controller.init_params()
    #     for _ in range(self.epochs):
    #         # Init any other controller variables (error hisotry etc.)
    #         self.controller.init_error_history()
    #         # TODO maybe state history in consys eller ingen state history
    #         self.controller.init_state_history()
    #         # Reset plant to initial state
    #         self.plant.reset_plant()
    #         # Generate disturbance vector
    #         disturbance_vector = np.random.uniform(*self.disturbance_range, size=self.timesteps)
    #         for t in range(self.timesteps):
    #             # Update plant
    #             # Update controller
    #             # TODO maybe consys should contain the state_history and controller only contain error_history
    #             incoming_error = 0.0 if not self.controller.error_history else self.plant.target - 
    #             plant_output = self.plant.compute_plant_output(self.controller.params, disturbance_vector[t], self.controller.error_history, self.controller.compute_control_signal)
    #             print(plant_output)
    #             # Save the error E (state for differentiation) for this timestep in an error history
    #             # self.controller.save_state(timestep=t, 
    #             #                            disturbance=disturbance_vector[t],
    #             #                            error_history=self.controller.error_history)
    #             # TODO maybe this should be before save state? no i think after
    #             self.controller.save_error(self.plant.target - plant_output)
    #         # compute MSE over the error history
    #         mse = self.mse(self.controller.params, self.controller.error_history, disturbance_vector)
    #         print("MSE: ", mse, end="\n\n")
    #         # TODO visualization here?
    #         # compute the gradients (derivative of MSE over params (three k values or weights))
    #         loss_gradient = jax.grad(self.mse)
    #         print(loss_gradient(self.controller.params, self.controller.error_history, disturbance_vector))
    #         # Update params based on gradient
    #         # self.controller.get_updated_params()
    #         # self.controller.set_params()
