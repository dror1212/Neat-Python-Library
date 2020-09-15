from Matrix import Matrix
from mpmath import mp


def sigmoid(x):
    return 1 / (1 + mp.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, inputsCount, hiddenCount, outputCount):
        if (isinstance(inputsCount, NeuralNetwork)):
            a = inputsCount
            self.input_nodes = a.input_nodes
            self.hidden_nodes = a.hidden_nodes
            self.output_nodes = a.output_nodes

            self.weights_inputs_hidden = a.weights_inputs_hidden.copy()
            self.weights_hidden_outputs = a.weights_hidden_outputs.copy()

            self.bias_hidden = a.bias_hidden.copy()
            self.bias_output = a.bias_output.copy()
        else:
            self.input_nodes = inputsCount
            self.hidden_nodes = hiddenCount
            self.output_nodes = outputCount

            self.weights_inputs_hidden = Matrix(self.hidden_nodes, self.input_nodes)
            self.weights_hidden_outputs = Matrix(self.output_nodes, self.hidden_nodes)
            self.weights_inputs_hidden.randomize()
            self.weights_hidden_outputs.randomize()

            self.bias_hidden = Matrix(self.hidden_nodes, 1)
            self.bias_output = Matrix(self.output_nodes, 1)
            self.bias_hidden.randomize()
            self.bias_output.randomize()
        self.learning_rate = 0.1

    def predict(self, input_array):
        inputs = Matrix.fromArray(input_array)
        hidden = Matrix.algebra_multiply(self.weights_inputs_hidden, inputs)

        hidden.add(self.bias_hidden)
        hidden.map(sigmoid)

        output = Matrix.algebra_multiply(self.weights_hidden_outputs, hidden)
        output.add(self.bias_output)
        output.map(sigmoid)

        return output.toArray()

    def train(self, inputs_array, targets):
        inputs = Matrix.fromArray(inputs_array)
        hidden = Matrix.algebra_multiply(self.weights_inputs_hidden, inputs)
        hidden.add(self.bias_hidden)

        hidden.map(sigmoid)

        outputs = Matrix.algebra_multiply(self.weights_hidden_outputs, hidden)
        outputs.add(self.bias_output)
        outputs.map(sigmoid)

        target = Matrix.fromArray(targets)

        output_errors = Matrix.sub(target, outputs)

        gradients = Matrix.staticMap(outputs, dsigmoid)
        gradients.multiply(output_errors)
        gradients.multiply(self.learning_rate)

        hidden_t = Matrix.transpose(hidden)
        weight_hidden_out_deltas = Matrix.algebra_multiply(gradients, hidden_t)

        self.weights_hidden_outputs.add(weight_hidden_out_deltas)
        self.bias_output.add(gradients)

        who_t = Matrix.transpose(self.weights_hidden_outputs)
        hidden_errors = Matrix.algebra_multiply(who_t, output_errors)

        hidden_gradient = Matrix.staticMap(hidden, dsigmoid)
        hidden_gradient.multiply(hidden_errors)
        hidden_gradient.multiply(self.learning_rate)

        inputs_t = Matrix.transpose(inputs)
        weight_ih_deltas = Matrix.algebra_multiply(hidden_gradient, inputs_t)

        self.weights_inputs_hidden.add(weight_ih_deltas)
        self.bias_hidden.add(hidden_gradient)

    def copy(self):
        return NeuralNetwork(self, 0, 0)