import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, activation_function="sigmoid", cost_function="mse"):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.weights = []
        self.biases = []
        self._initialize_network()

    def _initialize_network(self):
        # Define the architecture of the network: input -> hidden layers -> output
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.random.randn(1, layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    def forward(self, X):
        self.a = [X]  # Input activations
        self.z = []  # Layer activations
        for i in range(len(self.weights)):
            self.z.append(np.dot(self.a[i], self.weights[i]) + self.biases[i])
            if self.activation_function == "sigmoid":
                self.a.append(self.sigmoid(self.z[-1]))
            elif self.activation_function == "relu":
                self.a.append(self.relu(self.z[-1]))
        return self.a[-1]

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        if self.cost_function == "mse":
            dz = self.mse_derivative(y, output) * self.sigmoid_derivative(output)
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.a[i].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.sigmoid_derivative(self.a[i])

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return self.forward(X)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        predicted_classes = (predictions > 0.5).astype(int)
        return np.mean(predicted_classes == y)


# GUI Implementation with Tkinter
class NeuralNetworkUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network for Pima Indians Diabetes Dataset")

        self.nn = None
        self.X_train = None
        self.y_train = None
        self.dataset = None

        # Load the dataset
        self.load_dataset()

        # Create UI elements
        self.create_widgets()

    def load_dataset(self):
        # Load the Pima Indians Diabetes dataset
        filepath = 'diabetes.csv'  # Path to dataset
        self.dataset = pd.read_csv(filepath)
        self.X = self.dataset.drop('Outcome', axis=1).values
        self.y = self.dataset['Outcome'].values.reshape(-1, 1)
        self.X_train = self.X
        self.y_train = self.y

    def create_widgets(self):
        # Buttons
        self.train_button = tk.Button(self.root, text="Train", command=self.train_model)
        self.train_button.pack(pady=10)

        self.test_button = tk.Button(self.root, text="Test", command=self.test_model)
        self.test_button.pack(pady=10)

        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict_and_save)
        self.predict_button.pack(pady=10)

        self.visualize_button = tk.Button(self.root, text="Visualize NN", command=self.visualize_nn)
        self.visualize_button.pack(pady=10)

        # Activation and Cost function options
        self.activation_label = tk.Label(self.root, text="Choose Activation Function:")
        self.activation_label.pack(pady=10)
        self.activation_var = tk.StringVar(value='sigmoid')
        self.activation_options = tk.OptionMenu(self.root, self.activation_var, *['sigmoid', 'relu'])
        self.activation_options.pack(pady=10)

        self.cost_label = tk.Label(self.root, text="Choose Cost Function:")
        self.cost_label.pack(pady=10)
        self.cost_var = tk.StringVar(value='mse')
        self.cost_options = tk.OptionMenu(self.root, self.cost_var, *['mse', 'cross_entropy'])
        self.cost_options.pack(pady=10)

    def train_model(self):
        activation_function = self.activation_var.get()
        cost_function = self.cost_var.get()

        # Initialize the neural network with correct number of neurons in the first layer
        self.nn = NeuralNetwork(input_size=self.X_train.shape[1], hidden_layers=[8, 12],
                                output_size=1,
                                activation_function=activation_function, cost_function=cost_function)
        self.nn.train(self.X_train, self.y_train, epochs=1000, learning_rate=0.01)
        messagebox.showinfo("Training", "Dataset learned.")

    def test_model(self):
        if self.nn:
            accuracy = self.nn.accuracy(self.X_train, self.y_train)
            messagebox.showinfo("Test Results", f"Accuracy: {accuracy * 100:.2f}%")
        else:
            messagebox.showerror("Error", "Please train the model first.")

    def predict_and_save(self):
        if self.nn:
            # Choose the file for prediction
            file_path = filedialog.askopenfilename(title="Select File to Predict", filetypes=[("CSV Files", "*.csv")])
            if file_path:
                new_data = pd.read_csv(file_path)
                new_data = new_data.reindex(columns=self.dataset.columns.drop('Outcome'), fill_value=0)
                predictions = self.nn.predict(new_data.values)

                # Show predictions in UI
                prediction_result = "\n".join([str(int(pred > 0.5)) for pred in predictions.flatten()])
                messagebox.showinfo("Predictions", f"Predictions: \n{prediction_result}")

                # Save predictions to a new file
                save_path = filedialog.asksaveasfilename(title="Save Predictions", defaultextension=".csv",
                                                         filetypes=[("CSV Files", "*.csv")])
                if save_path:
                    binary_predictions = (predictions > 0.5).astype(int)
                    prediction_df = pd.DataFrame(binary_predictions, columns=["Prediction"])
                    prediction_df.to_csv(save_path, index=False)
                    messagebox.showinfo("Save", "Predictions saved successfully.")
        else:
            messagebox.showerror("Error", "Please train the model first.")

    def visualize_nn(self):
        # Visualize neural network architecture
        self.plot_nn()

    def plot_nn(self):
        input_size = self.X_train.shape[1]  # Input layer size
        hidden_layers = [12, 12]  # Example hidden layers
        output_size = 1  # Output layer size for binary classification

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Neural Network Architecture")

        # Define positions for layers
        layer_positions = {}
        layers = [input_size] + hidden_layers + [output_size]

        for i, layer_size in enumerate(layers):
            layer_positions[i] = np.linspace(0, layer_size - 1, layer_size)
            ax.scatter([i] * layer_size, layer_positions[i], s=100, label=f"Layer {i + 1}")

        # Connect nodes (weights)
        for i in range(len(layers) - 1):
            for j in range(layers[i]):
                for k in range(layers[i + 1]):
                    ax.plot([i, i + 1], [layer_positions[i][j], layer_positions[i + 1][k]], 'k-', lw=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
        plt.show()


# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkUI(root)
    root.mainloop()





