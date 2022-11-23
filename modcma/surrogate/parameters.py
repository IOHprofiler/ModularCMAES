
from parameters import *

# #######################################
# SurrogateStrategy : It evaluates current population with <SurrogateModel>
#       if the evaluation is not precise enough, more data will be added
#       to the <SurrogateData>
# SurrogateModel      : Given training data it does the regression
#       It uses samples and weights provided by <SurrogateData>
# SurrogateData       : Storage of data. It can be sorted and weighted.

class SurrogateStrategy_Proportion_Settings(AnnotatedStruct):
    true_eval: float = 0.5

class SurrogateStrategy_Settings(AnnotatedStruct):
    # Minimal number / proportion of population to be evaluated 
    # using true objective function
    min_evals_percent: int = 2  # Max 100
    min_evals_absolute: int = 1

    proportion = SurrogateStrategy_Proportion_Settings()




class SurrogateModel_Settings(AnnotatedStruct):
    pass


class SurrogateData_Settings(AnnotatedStruct):
    max_size: Optional[int] = None

    #  max_size_relative_df: Optional[float] = 2.0  # df

    # Weighting
    weight_function: str = 'linear'
    weight_max: float = 20.
    weight_min: float = 1.

    # Sorting method:
    sorting: str = 'LQ'



'''
# **************** MODEL **************** 

class KendalTau_SurrogateEvaluation_Settings(SurrogateEvaluation_Settings):

    # require another true evaluations if the kendall tau \in (-1, 1) is 
    # lower than this value
    tau_truth_threshold: float = 0.85



class SurrogateStrategy_Settings(AnnotatedStruct):
    # do not use model if the number of training samples are not bigger than the value
    minimum_model_size = 3 #  absolute minimum number of true evaluation to build a model

    # return true fitness only if all samples are evaluated
    return_true_fitness_if_all_evaluated: bool = True


class LQ_SurrogateStrategy_Settings(SurrogateStrategy_Settings):
    # ************ VARIABLES ***************

    # Max degrees of freedom 
    # (usually depends on the model and dimensionality of the problem)
    dfMax = smp.symbols('dfMax', positive=True, integer=True)

    # size of population - it can be increased when restart ...
    lam = smp.symbols('lam', positive=True, integer=True)

    # number of evaluations avail.
    surrogate_size = smp.symbols('surrogate_size', positive=True, integer=True)

    # number of evaluations as training samples in the current model



    # *********** SETTINGS ************


    # every unsuccesfull (kendall tau) iteration, increse
    # the number of evauated samples by this fraction
    # evals = evals + evals * increase_of_number_of_evaluated
    increase_of_number_of_evaluated: float = 0.5



    # TODO: ???
    truncation_ratio = 0.75


    # TODO: ???
    number_of_evaluated = smp.Integer(1) + \
        smp.floor(smp.maximum(lam * min_evals_percent / 100,
                              3. / truncation_ratio - surrogate_size
        ))

    n_for_tau = smp.max(
        15,
        smp.min(1.2 * ,
                0.75 * lam)
    )

        int(max((15, min((1.2 * nevaluated, 0.75 * popsi)))))


'''



