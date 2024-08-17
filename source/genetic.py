from deap import algorithms, creator, base, tools
from scipy.stats import bernoulli


class Genetic:
    def __init__(self, evaluate, geneSize=25, popSize=200, cxpb=0.5, mutpb=0.2, ngen=50):
        self.evaluate = evaluate
        self.geneSize = geneSize
        self.popSize = popSize
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen

    def run(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("binary", bernoulli.rvs, 0.5)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.binary,
            n=self.geneSize,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        pop = toolbox.population(n=self.popSize)

        toolbox.register("evaluate", self.evaluate)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)

        # stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        # stats.register("avg", np.mean)
        # stats.register("std", np.std)
        # stats.register("min", np.min)
        # stats.register("max", np.max)

        resultPop, logbook = algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.ngen,
            verbose=False,
            # stats=stats,
        )

        return tools.selBest(resultPop, 1)[0]

        # logbook.header = "gen", "nevals", "avg", "std", "min", "max"


if __name__ == "__main__":

    def feasible(individual):
        return sum(individual) < 10

    def eval(individual):
        if not feasible(individual):
            return (0,)
        return (sum(individual),)

    genetic = Genetic(eval)
    best = genetic.run()
    print(sum(best))
