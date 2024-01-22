import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Plotting:

    def __init__(self) -> None:
        self.epochs = []
        self.mse_history = []
        self.kp = []
        self.ki = []
        self.kd = []
    
    def add(self, epoch, mse, kp=None, ki=None, kd=None):
        self.epochs.append(epoch)
        self.mse_history.append(mse)
        if kp:
            self.kp.append(kp)
        if ki:
            self.ki.append(ki)
        if kd:
            self.kd.append(kd)

    def plot_mse_and_params(self):
        if self.kp or self.ki or self.kd:
            plt.figure(figsize=(8,7))
            # Create a 2x1 subplot grid
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.mse_history, label="MSE")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title("MSE")
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.kp, label='Kp')
            plt.plot(self.epochs, self.ki, label='Ki')
            plt.plot(self.epochs, self.kd, label='Kd')
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title("Parameters")
            plt.legend()
            # Or else the titles of the two plots are overlapping
            plt.tight_layout()
        else:
            plt.figure(figsize=(6,5))
            plt.plot(self.epochs, self.mse_history, label="MSE")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title("MSE")
            plt.legend()
        plt.show()
