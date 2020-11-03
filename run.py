import os

import numpy as np
from modcma.modularcmaes import evaluate
from modcma.bbob import bbobbenchmarks

import numpy as np
import cma
from inverse_covariance import QuicGraphicalLasso 
from scipy.stats import special_ortho_group


def correlation_matrix(C):
    c = C.copy()
    for i in range(c.shape[0]):
        fac = c[i, i]**0.5
        c[:, i] /= fac
        c[i, :] /= fac
    c = (c + c.T) / 2.0
    return c


def reset_ccov_learning_rate(es, nz):
    es.opts['CMA_rankone'] = 1.
    es.opts['CMA_rankmu'] = 1.
    es.sp = cma.evolution_strategy._CMAParameters(N = es.N,
                                              opts = es.opts,
                                              ccovfac = 1.,
                                              verbose = es.opts['verbose'] > 0)
    

    
    ccovfac = 1.0
    alphacov = 2.0
    mueff = es.sp.weights.mueff

    rankone_factor = 2. / (2 * (nz/es.N + 0.15) * (es.N + 1.3)**1. +
                          es.sp.weights.mueff) / es.sp.c1

    # this happens in PyCMA
    c1 = (1.0 * rankone_factor * ccovfac * min(1, es.sp.popsize / 6) *
            2 / ((es.N + 1.3)** 2.0 + mueff))

    c1_ = 2 / (
            (nz + 1.3) * (es.N + 1.3) + mueff
        )
    cmu_ = min(1 - c1_, (
        2 * ( (mueff + 1 / mueff - 1.75) / (
                (nz + 2) * (es.N + 2) + mueff
                )
            )
        )
    )

    breakpoint()


    # rankmu_factor = 2.*(0.25+es.sp.weights.mueff + 1. / es.sp.weights.mueff - \
    #     2. + 0*es.popsize / ( 2 * (es.popsize + 5))) / \
    #     (2 * (nz/es.N + 0.5) * (es.N + 2)**1. +
    #     es.sp.weights.mueff) / es.sp.cmu
    # # this happens in PyCMA
    # sp.cmu = min(1 - sp.c1,
    #         rankmu_factor * ccovfac * alphacov *
    #         (rankmu_offset + mu + 1 / mu - 2) /
    #         ((N + 2)** 2.0 + alphacov * mu / 2))

    # breakpoint()
    ccovfac = 1. # For testing purpose
    es.opts['CMA_rankone'] = rankone_factor * ccovfac
    es.opts['CMA_rankmu'] = rankmu_factor * ccovfac
    es.sp = cma.evolution_strategy._CMAParameters(N = es.N,
                                                opts = es.opts,
                                                verbose = es.opts['verbose'] > 0)

def run_pycma_reg(testfun=None):
    dim = 5
    P1 = np.eye(dim)
    B1 = special_ortho_group.rvs(int(dim/2))
    B2 = special_ortho_group.rvs(int(dim/2))
    B = np.eye(dim)
    P2 = np.random.permutation(np.eye(dim))
    testfun = testfun or (lambda x: cma.ff.elli(np.linalg.multi_dot([P2, B, P1, np.array(x)])))
    for dim in [dim]:
        for threshold in np.linspace(0., 1., 2):
            factor = 1.
            alpha = 1
            
            thresholds = [threshold]
            prefix = '2block_elli_dim_'+str(dim)+'_thr'+str(int(10000*threshold))+'e-4'
            es = cma.CMAEvolutionStrategy(dim*[3], 1.0, inopts = {'ftarget':1e-10,
                                                                  'CMA_active':False,
                                                                  'verbose':1,
                                                                  'verb_filenameprefix':prefix,
                                                                  #'AdaptSigma':False,
                                                                  
                                                                 })
            es.adapt_sigma.initialize(es)
            
            sm_D, sm_B = es.sm.D.copy(), es.sm.B.copy()

            while not es.stop():
                C_tilde = es.sm.correlation_matrix
                P = np.linalg.inv(C_tilde)
                P_tilde = correlation_matrix(P)
               
                #Regularize the sample matrix
                W = alpha * np.float_(np.abs(P_tilde) < threshold)  #*(1-np.abs(P_tilde))**2.

                est = QuicGraphicalLasso(lam=W,
                    Sigma0=C_tilde,
                    Theta0=P,
                    init_method=lambda x: (x.copy(),
                                            np.max(np.abs(np.triu(x)))),
                ).fit(C_tilde)

                diag_root_C = np.diag(es.sm.C)

                sample_matrix = np.linalg.multi_dot(
                    (np.diag(diag_root_C**.5), est.covariance_,
                    np.diag(diag_root_C**.5)))
                
                
                sm_D, sm_B = np.linalg.eigh(sample_matrix)

                def my_transform_inverse(x):
                    return np.dot(sm_B, np.dot(sm_B.T, x) / sm_D**.5)


                es.sm.transform_inverse = my_transform_inverse
                nz = np.sum(np.abs(np.triu(est.precision_, 1))>0)
                reset_ccov_learning_rate(es, nz+dim) # this is also differrent!

                #Sample and Update
                arz = np.random.randn(es.sp.popsize, dim)
                X = es.ask()
                X = np.dot(sm_B, (sm_D**.5 * arz).T).T
                Y = [es.mean+es.sigma*x for x in X] #needs additionally sigma_vec
                fit = [testfun(y) for y in Y]

                es.tell(Y, fit)
                es.disp(100)

def toast(msg):
    msg = (msg
        .replace('\n', '`n')
        .replace('\t', '')
        .replace(' ', '` ')
        .replace(',', '`,')
        .replace('(', '`(')
        .replace(')', '`)')
    )
    os.system(f"powershell.exe -command New-BurntToastNotification -Text '{msg}'")

if __name__ == "__main__":
    import sys, shutil
    # run_pycma_reg()

    for f in (1, 2):
        *_, msg = evaluate(f, 5, 5, regularization=False, label="CMA-ES")
        for tau in np.linspace(.2, .99, 10):
            print()
            *_, msg = evaluate(f, 5, 5, regularization=True, tau=tau, label=f"CMA-ES (reg, tau={tau})")

    # iterations = 5
    # for f in range(1, 25):
    #     *_, msg = evaluate(f, 4, iterations, regularization=False, label="CMA-ES")
    #     print()
    # exp_name = 'test-regularization'
    # if os.path.isdir(os.path.join("./data", exp_name)):
    #     shutil.rmtree(os.path.join("./data", exp_name))

    # init sigma should be 1

    # f = int(sys.argv[1])
    # for d in (5, 10, 20, 40, 80):
    #     *_, msg = evaluate(f, d, iterations, regularization=False, label="CMA-ES", logging=True, data_folder = "./data", exp_name=exp_name)
    #     print()
    #     for tau in np.linspace(.0, .99, 10):
    #         *_, msg = evaluate(f, d, iterations,  regularization=True, tau=tau, label=f"CMA-ES (reg, tau={tau})", logging=True, data_folder = "./data", exp_name=exp_name)
    #         print()
    #     print("*"*80)
    #     print()

    

     