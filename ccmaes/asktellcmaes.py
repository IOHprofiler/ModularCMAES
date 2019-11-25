from collection import deque
from .configurablecmaes import ConfigurableCMAES


class AskTellCMAES(ConfigurableCMAES):

    def __init(self, *args, **kwargs):
        super().__init__(lambda x: None, *args, **kwargs)
        self.queue = deque()

    def run(self):
        raise NotImplemented()

    def step(self):
        raise NotImplemented()

    def pre_ask(self):
        if not self.asked:
            self.mutate()
            self.queue = deque(self.population.x.tolist())

    def ask(self):
        if not any(self.queue):
            self.pre_ask()
        return self.queue.pop()

    def tell(self, x, f):
        if not any(self.queue) and x == self.population.x:
            self.parameters.population.f = f
            self.recombine()
            self.select()
            self.adapt()`
