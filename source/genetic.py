import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Genetic:
    def __init__(
        self,
        F,
        n_iter=50,
        ACC=20,
        SIZE=200,
        CROSS_RATE=0.8,
        MUTATION_RATE=0.005,
        BOUND=None,
        limit=None,
        tol=5,
    ):
        self.F = F
        self.n_iter = n_iter
        self.ACC = ACC
        self.SIZE = SIZE
        self.CROSS_RATE = CROSS_RATE
        self.MUTATION_RATE = MUTATION_RATE
        self.BOUND = BOUND
        self.limit = limit
        self.tol = tol

    def decode(self, DNA):
        if self.BOUND is None:
            return DNA

        DNA = DNA.reshape(-1, self.ACC)
        x = np.zeros(shape=DNA.shape[0])
        for i in range(self.ACC):
            x += DNA[:, i] * (2 ** (self.ACC - i - 1))
        x = x / (2**self.ACC - 1)
        x = x * (self.BOUND[1] - self.BOUND[0]) + self.BOUND[0]
        return x

    def update(self, DNA):
        DNA_new = np.copy(DNA)
        for i in range(DNA.shape[0]):
            if np.random.rand() < self.CROSS_RATE:
                mother = DNA[np.random.randint(DNA.shape[0])]
                crossPoint = np.random.randint(DNA.shape[1])
                DNA_new[i, crossPoint:] = mother[crossPoint:]

            if np.random.rand() < self.MUTATION_RATE:
                mutationPoint = np.random.randint(DNA.shape[1])
                DNA_new[i, mutationPoint] = np.abs(DNA_new[i, mutationPoint] - 1)

        return DNA_new

    def select(self, DNA):
        score = self.F(self.decode(DNA))
        p = (score - score.min()) / ((score.max() - score.min()) + 1e-3) + 1e-2
        idx = np.random.rand(DNA.shape[0]) < p

        if not idx.any():
            return DNA[np.argmax(score)].reshape(-1, self.ACC)

        return DNA[idx]

    def run(self):
        DNA = np.random.randint(0, 2, size=(self.SIZE, self.ACC))
        history = []

        for i in range(self.n_iter):
            DNA = self.update(DNA)
            DNA = self.select(DNA)
            history.append(self.decode(DNA))

            if DNA.shape[0] == 1:
                self.tol -= 1
            if self.tol < 0:
                break

        x = self.decode(DNA[np.argmax(self.F(self.decode(DNA)))])
        history.append(x)
        return history


if __name__ == "__main__":

    def F(x):
        return 0.2 * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

    # 定义x在F函数中的映射值越小，适应度越大
    def Target(x):
        return -F(x)

    BOUND = (2, 6)
    x = np.linspace(BOUND[0], BOUND[1], 50)
    y = F(x)

    model = Genetic(Target, BOUND=BOUND, ACC=30, tol=2)
    history = model.run()

    fig, ax = plt.subplots()
    ax.set_xlim(BOUND)
    ax.plot(x, y)
    sca = ax.scatter([], [])

    def animate(i):
        data = np.stack([history[i], F(history[i])]).T
        sca.set_offsets(data)
        ax.set_title(f"step: {i+1}")

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=len(history) - 1, interval=150
    )
    plt.show()
