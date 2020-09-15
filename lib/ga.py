from nn import NeuralNetwork
import random

def basic_mutate(x, mutation_rate, mutation_power):
    if random.random() <= mutation_rate:
        offset = random.uniform(-1, 1)

        x = x * (1 - mutation_power) + offset * mutation_power
    return x


class ga:
    def __init__(self, inputs, hidden, outputs, mutation_rate, mutation_power):
        self.nn = NeuralNetwork(inputs, hidden, outputs)
        self.fitness = 0
        self.mutation_rate = mutation_rate
        self.mutation_power = mutation_power

    def predict(self, inputs_array):
        return self.nn.predict(inputs_array)

    def mutate(self, func = basic_mutate):
        self.nn.weights_inputs_hidden.BigMap(func, self.mutation_rate, self.mutation_power)
        self.nn.weights_hidden_outputs.BigMap(func, self.mutation_rate, self.mutation_power)
        self.nn.bias_hidden.BigMap(func, self.mutation_rate, self.mutation_power)
        self.nn.bias_output.BigMap(func, self.mutation_rate, self.mutation_power)

    def copy(self):
        cop = self.nn.copy()
        g = ga(self.nn.input_nodes, self.nn.hidden_nodes, self.nn.output_nodes, self.mutation_rate, self.mutation_power)
        g.nn = cop

        return g

    def crossOver(self, partner):
        child = self.copy()
        child.nn.weights_inputs_hidden.VariableMap(myCrossOver, partner.nn.weights_inputs_hidden)
        child.nn.weights_hidden_outputs.VariableMap(myCrossOver, partner.nn.weights_hidden_outputs)
        child.nn.bias_hidden.VariableMap(myCrossOver, partner.nn.bias_hidden)
        child.nn.bias_output.VariableMap(myCrossOver, partner.nn.bias_output)

        return child

def myCrossOver(first, second):
    if random.random() > 0.5:
        return first
    else:
        return second
