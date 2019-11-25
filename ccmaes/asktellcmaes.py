from .configurablecmaes import ConfigurableCMAES


class AskTellCMAES(ConfigurableCMAES):

    def __init(self, *args, **kwargs):
        super().__init__(lambda x: None, *args, **kwargs)

    def run(self):
        raise NotImplemented()

    def step(self):
        raise NotImplemented()

    def ask(self):
        self.mutate()
        for x in self.parameters.population.x.copy():
            yield x

    def tell(self, x, f):
        # checking if asked
        self.parameters.population.f = f
        self.recombine()
        self.select()
        self.adapt()
