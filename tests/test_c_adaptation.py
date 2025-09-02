import unittest
import numpy as np 
from modcma.c_maes import parameters, Parameters, Population, ModularCMAES, options


class TestMutation(unittest.TestCase):
  
    def get_cma(self, modules, adapt_sigma=True):
        settings = parameters.Settings(2, modules)
        par = Parameters(settings)
        
        cma = ModularCMAES(par)
        if adapt_sigma:
            cma.mutate(sum)
            cma.select()
            cma.recombine()
        return cma
    
    def test_matrix_adaptation(self):
        modules = parameters.Modules()
        modules.matrix_adaptation = options.MatrixAdaptationType.MATRIX
        cma = self.get_cma(modules)
        
        M = cma.p.adaptation.M.copy()
        z = np.sum(cma.p.weights.positive * cma.p.pop.Z[:, :cma.p.mu], axis=1, keepdims=True)
        ps = ((1.0 - cma.p.weights.cs) * cma.p.adaptation.ps + (np.sqrt(cma.p.weights.cs * (2.0 - cma.p.weights.cs) * cma.p.weights.mueff) * z.ravel())).reshape(-1, 1)
        old_M = ((1 - 0.5 * cma.p.weights.c1 - 0.5 * cma.p.weights.cmu) * M)
        scaled_ps = ((0.5 * cma.p.weights.c1) * M.dot(ps).dot(ps.T))
        new_M = ((0.5 * cma.p.weights.cmu * cma.p.weights.positive) * cma.p.pop.Y[:, :cma.p.mu]).dot(cma.p.pop.Z[:, :cma.p.mu].T)
        M = old_M + scaled_ps + new_M

        cma.adapt()
        self.assertTrue(np.all(np.isclose(cma.p.adaptation.M, M)))
          

if __name__ == "__main__":
    unittest.main()