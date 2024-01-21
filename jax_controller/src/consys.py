import numpy as np
import jax
import jax.numpy as jnp

# TODO wrong typing, should be base
from controllers.classic_pid_controller import ClassicPIDController


class Consys:

    def __init__(self, controller: ClassicPIDController, plant, disturbance_range) -> None:
        self.controller = controller
        self.plant = plant
        self.disturbance_range = disturbance_range
    
    def run_system(self, epochs, timesteps):
        gradient_function = jax.value_and_grad(self.run_system_one_epoch)
        # Init params
        params = self.controller.init_params()

        for _ in range(epochs):
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
            
            print(state)

            # Executing run_system_one_epoch via jax gradient function
            mse, gradient = gradient_function(params, state, timesteps)

            print(mse, gradient)

            # update_params(params, gradient) Update parameters based on the gradient
    
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
        print("before run plant")
        print(state)
        state = self.plant.update_plant(state, disturbance)
        print("after", state)
        return state

    def run_controller_one_timestep(self, params, state):
        # Update controller
        state = self.controller.update_controller(params, state)
        return state
    
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
