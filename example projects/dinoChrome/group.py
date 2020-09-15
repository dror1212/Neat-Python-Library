from ga import ga
import random
import math
from graph import DrawNN
from multiprocessing import Process
from FileHandler import FileHandler

class group:
    def __init__(self, population, function, inputs, hidden, outputs, display = True, mutation_rate = 0.01, survival_rate = 0.3,
                 mutation_power = 0.3, species = 1, save_to_file = False, best_mode = False):
        self.population = population
        self.func = function
        self.fitness_sums = []
        self.species = species
        self.best_mode = best_mode
        self.save_to_file = save_to_file
        if self.save_to_file and FileHandler.read('generation.pickle'):
            self.nets = FileHandler.read('generation.pickle')
        if(self.best_mode and FileHandler.read('best.pickle')):
            self.best = FileHandler.read('best.pickle')
            self.nets = [[self.best]]
        elif not self.save_to_file or not FileHandler.read('generation.pickle'):
            if best_mode:
                print("Could not find a file, acting ordirnally")
            self.nets = [[ga(inputs, hidden, outputs, mutation_rate, mutation_power) for i in range(int(self.population / species))] for i in range(species)]
        self.best = None
        self.p = None
        self.survival_rate = survival_rate
        self.display = display

    def chooseOne(self, sp):
        rand = random.randrange(0, math.floor(self.fitness_sums[sp]))
        index = 0
        while rand > self.nets[sp][index].fitness:
            rand -= self.nets[sp][index].fitness
            index += 1

        return self.nets[sp][index].copy()

    def init(self):
        for j in range(self.species):
            amount = int(self.population * self.survival_rate / self.species)
            new = self.nets[j][:amount]

            for i in range(amount):
                new[i] = new[i].copy()

            for i in range(int(len(self.nets[j]) - amount)):
                first = self.chooseOne(j)
                partner = self.chooseOne(j)
                child = first.crossOver(partner)
                child.mutate()
                new.append(child)
            self.nets[j] = new

    def getBest(self):
        self.fitness_sums = [0 for i in range(self.species)]
        self.best = self.nets[0][0]

        for i in range(self.species):
            for b in self.nets[i]:
                self.fitness_sums[i] += b.fitness
                if self.best.fitness < b.fitness:
                    self.best = b
        if(self.save_to_file):
            FileHandler.save(self.best.copy(),'best.pickle')
            FileHandler.save(self.nets,'generation.pickle')

    def run(self):
        while True:
            arr = []
            for i in self.nets:
                for j in i:
                    arr.append(j)
            if(not self.best_mode):
                self.func(arr)
                self.getBest()
                if self.display:
                    if self.p != None:
                        self.p.kill()
                    self.p = Process(target=self.plot_graph, args = ())
                    self.p.start()
                for i in range(len(self.nets)):
                    self.nets[i].sort(key=lambda net: net.fitness, reverse=True)
                self.init()
            else:
                self.func(arr)
                if self.display:
                    if self.p != None:
                        self.p.kill()
                    self.p = Process(target=self.plot_graph, args = ())
                    self.p.start()

    def plot_graph(self):
        draw = DrawNN(self.best.nn)
        draw.draw()
