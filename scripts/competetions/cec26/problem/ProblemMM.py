# ********* Class of New Composite Multimodal Optimization Problems for CEC 2026 competition **************
# ********* see example1.py in the upper folder for instrcutions on using this class **************
# code developed by Ali Ahrari
# last update on 10-Feb-2026 by A. A.
# random seed numbers have been generated using MATLAB:
# seed 0: for 10000 uniform numbers with precision of 10
# seed 1: for normal numbers with precision of 10
# seed 2: for 248 permutations of size 10000
# seed 3: for 8 global minimum values

import numpy as np
import pandas as pd

if __name__ != "__main__":  # relative import
    from .BasicFun import BasicFun
    from .UtilityMethod import UtilityMethod
else:  # absolute import
    from scripts.competetions.cec26.problem.BasicFun import BasicFun
    from scripts.competetions.cec26.problem.UtilityMethod import UtilityMethod

from scipy.spatial.distance import cdist
import os, copy, sys


class ProblemMM:
    """The main class. It creates an object that determines the problem"""

    __slots__ = [
        "max_instance_no",
        "pid",
        "fun_id",
        "n_minima",
        "hard_GO",
        "hard_NU",
        "instance_no",
        "dim",
        "low_bound",
        "up_bound",
        "lambda0",
        "d_min",
        "max_eval_coeff",
        "init_mult",
        "sigma_width",
        "max_eval",
        "used_eval",
        "master_seed",
        "rotation",
        "minima",
        "depend",
        "normal_numbers",
        "uniform_numbers",
        "index_normal",
        "index_uniform",
    ]

    def __init__(
        self, pid, instance_no, dim
    ):  # requires functions ID and problem dimensionality
        self.max_instance_no = 15  # a total of 15 problem instances
        if (
            (instance_no > self.max_instance_no)
            or (instance_no < 1)
            or (type(instance_no) != int)
        ):
            print(
                "Error: instance_no should be an integer between 1 and 15 (inclusive)"
            )
            sys.exit(1)
        if pid < 1 or pid > 16 or type(pid) != int:
            print("Error: pid should be an integer between 1 and 16 (inclusive)")
            sys.exit(1)
        if dim < 2 or dim > 80 or type(dim) != int:
            print("Error: dim should be an integer between 2 and 50 (inclusive)")
            sys.exit(1)
        folder = os.path.dirname(__file__)  # folder of current class file
        path2file = os.path.join(folder, "data/pid-data.xlsx")  # path to data file
        data = pd.read_excel(path2file)
        self.pid = pid
        # problem ID
        self.fun_id = int(
            0.5 + data.fun_id[pid - 1]
        )  # function ID for the basic function (different from problem ID)
        self.n_minima = int(
            0.5 + data.n_minima[pid - 1]
        )  # (scalar) number of global minima
        self.hard_GO = [
            data.hard_GO_min[pid - 1],
            data.hard_GO_max[pid - 1],
        ]  # a number in [0,1] that specifies the hardness from global optimization perspective
        self.hard_NU = data.hard_NU[
            pid - 1
        ]  # a non-negative Real number specifying the non-uniformity in the distribution of global minima
        self.instance_no = instance_no  # problem instance No
        self.dim = dim  # dimensionality
        self.low_bound = -5 * np.ones(self.dim)  # the lower bound of the search space
        self.up_bound = 5 * np.ones(self.dim)  # the upper bound of the search space
        self.lambda0 = data.lambda0[
            pid - 1
        ]  # for scaling the search range of the basic function
        self.d_min = 0.3 * self.dim**0.5  # distance threshold between global minima
        self.max_eval_coeff = data.max_eval_coeff[
            pid - 1
        ]  # the coefficient for the evaluation budget
        self.init_mult = 5  # candidate multiplier for samling uniform solutions for coordinates of global minima
        self.sigma_width = 0.5  # controls the impact extent of a basic function

        # protected variables
        self.used_eval = 0  # used evaluation so far
        self.master_seed = None
        # problem special number used for reading random numbers from CSV files
        self.max_eval = None
        # the evaluation budget
        self.depend = Dependency(
            self.pid, self.dim, self.instance_no, self.max_instance_no
        )  # the dependency structure among variables
        self.rotation = Rotation()  # data for rotation of the modes
        self.minima = (
            Minima()
        )  # information about the global minima (locations,...): For postprocessing of results only
        self.normal_numbers = (
            None  # a series of random numbers from the standard normal distribution
        )
        self.uniform_numbers = (
            None  # a series of random numbers from the standard uniform distribution
        )
        self.index_normal = 0
        # current index for reading number from self.normal_numbers
        self.index_uniform = 0
        # current index for reading number from self.uniform_numbers

    def __str__(self):  # display the problem
        output = (
            "\npid:"
            + "\n    pid = "
            + str(self.pid)
            + "\n\tinstance_no = "
            + str(self.instance_no)
            + "\n\tdim = "
            + str(self.dim)
            + "\n\tlow_bound = "
            + str(self.low_bound)
            + "\n\tup_bound = "
            + str(self.up_bound)
            + "\n\tmax_eval = "
            + str(self.max_eval)
            + "\n\tused_eval = "
            + str(self.used_eval)
            + "\n\tn_global_min = "
            + str(self.n_minima)
            + "\n\thard_GO = "
            + str(self.hard_GO)
            + "\n\thard_NU = "
            + str(self.hard_NU)
        )
        return output

    def form(
        self,
    ):  # ******************* formulate problem *******************************
        """******************* Load problem data from CSV files *********************"""
        folder = os.path.dirname(__file__)  # folder of current class file

        path2file = os.path.join(folder, "data/num-uniform.csv")  # path to file
        num_uniform = np.loadtxt(path2file, delimiter=",", dtype=float)
        # array of uniformly distributed random numbers in (0,1)

        path2file = os.path.join(folder, "data/num-normal.csv")  # path to  file
        num_normal = np.loadtxt(path2file, delimiter=",", dtype=float)
        # array of numbers with standard normal distribution

        path2file = os.path.join(folder, "data/sequences.csv")  # path to file
        sequences = np.loadtxt(path2file, delimiter=",", dtype=int)
        # matrix, permutations of the aforementioned random numbers to be used successively

        path2file = os.path.join(folder, "data/fstar-data.csv")  # path to file
        fstar_data = np.loadtxt(path2file, delimiter=",", dtype=float)
        # global minimum values - 1-D array

        """ ****************************** set the evaluation budget and the global minimum value **************************** """
        self.max_eval = int(0.5 + self.max_eval_coeff * self.dim)
        self.minima.f = fstar_data[self.fun_id - 1]

        """ ***** specify array of random numbers (uniform and normal) to be used for benchmark generation ****** """
        self.master_seed = self.max_instance_no * (self.pid - 1) + self.instance_no
        # index number for this pid and instance_no
        used_seq = sequences[self.master_seed - 1, :] - 1
        # use this sequence of random numbers
        self.normal_numbers = num_normal[
            used_seq
        ]  # rearranged random numbers from normal distribution
        self.uniform_numbers = num_uniform[
            used_seq
        ]  # rearranged random numbers from uniform distribution
        del used_seq

        self.det_minima_coords()  # specify the locations of global minima
        self.det_minima_hardness()  # determine the hardness of each mode from global optimization or convergence perspective
        self.det_niche_rad()  # the niche radius for each global minimum based on the half of the distance to the closest global minima
        self.det_depend_base_struct()  # form the base dependency structure
        self.det_depend_struct()  # form the dependency structure for each mode by perturbation of the base dependency struture
        self.det_rotation_mat()  # create the rotation matrices given the dependency structures

    def det_minima_coords(self):
        """********************** determine locations of global minima ***********************"""
        temp = self.uniform_numbers[
            self.index_uniform : self.index_uniform
            + self.init_mult * self.dim * self.n_minima
        ]
        # select random uniform numbers several times of n_minima*dim
        self.index_uniform += temp.size  # for reading subsequent numbers
        rand_points = temp.reshape(self.n_minima * self.init_mult, self.dim)
        # solutions with random distribution from which global minima are selected
        # set the reference solution for redistribution
        self.minima.crowd_basin_ind = int(
            np.ceil(self.uniform_numbers[self.index_uniform] * self.n_minima) - 1
        )  # index of Xref is selected randomly
        self.index_uniform += 1
        # update the index of used random number with uniform distribution
        # Now create a relatively uniformly distributed set of points from randomly distributed points
        uniform_points, tmp = UtilityMethod.keep_farthest(rand_points, self.n_minima)
        # select farthest ones (remove closest ones iteratively)
        uniform_points = (
            uniform_points[0 : self.n_minima, :] * self.minima.range_coeff
            + (1 - self.minima.range_coeff) / 2
        )
        # make sure minima are not too close to the bounds
        uniform_points = (
            uniform_points * (self.up_bound - self.low_bound) + self.low_bound
        )
        # map uniform distribution from [-1,1]^dim to search space
        # Now set global minima locations by redistributing the uniform points (to make them non-uniform)
        self.minima.X, tmp = UtilityMethod.redist_glob_min(
            uniform_points,
            uniform_points[self.minima.crowd_basin_ind, :],
            self.hard_NU,
            self.d_min,
        )
        # non-uniform distribution

    def det_minima_hardness(self):
        """******************************** determine the hardness of each global minimum  ***************************"""
        ind0 = np.argsort(
            self.uniform_numbers[
                self.index_uniform : self.index_uniform + self.n_minima
            ]
        )
        # use random numbers to sort out the hardness
        self.index_uniform += self.n_minima
        if self.n_minima > 1:
            coef = (ind0) / (self.n_minima - 1)
            # This is n_minima uniformly distributed numbers in [0,1] with random order
            self.minima.hard_GO = (
                coef * (self.hard_GO[1] - self.hard_GO[0]) + self.hard_GO[0]
            )  # hard_GO for modes uniformly changes from the self.hard_GO[0] to self.hard_GO[1]
        else:  # in an unwanted case when there is only one global mode
            self.minima.hard_GO = np.array(
                [0.5 * self.hard_GO[1] + 0.5 * self.hard_GO[0]]
            )

    def det_niche_rad(self):
        """************************************** determine the niche_rad for each global minimum *******************************"""
        if self.n_minima == 1:
            self.minima.niche_rad = 5 * np.sqrt(self.dim)
        else:  # there are more than one global minima
            tmp = cdist(np.atleast_2d(self.minima.X), np.atleast_2d(self.minima.X))
            tmp = tmp + np.max(tmp) * np.eye(self.n_minima)
            # ignore diagonal elements (distance to self)
            self.minima.niche_rad = np.min(tmp, axis=0) / 2.0
            # half of distance to the closest global minimum

    def det_depend_base_struct(self):
        """****************************** determine base dependency structure for variables ********************************"""
        if self.depend.n_blocks == self.dim:  # fully separable
            self.depend.base_struct = [np.array([k]) for k in range(self.dim)]
        elif self.depend.n_blocks == 1:  # fully rotated
            self.depend.base_struct = [np.arange(self.dim)]
        elif (
            self.depend.n_blocks < self.dim and self.depend.n_blocks > 1
        ):  # block separability
            temp = self.uniform_numbers[
                self.index_uniform : self.index_uniform + self.dim
            ]
            self.index_uniform += self.dim
            rand_perm = np.argsort(
                temp
            )  # a random permutation of dimensions (0 to dim-1)
            self.depend.block_sizes = self.depend.block_size_min * np.ones(
                self.depend.n_blocks
            ).astype(
                int
            )  # minimum size of each block is 1
            candid_blocks = np.arange(
                self.depend.n_blocks
            )  # all blocks can increase their size
            while (
                self.depend.block_sizes.sum() < self.dim
            ):  # until the sum of the sizes of all blocks equald to problem dimensionality
                temp = self.uniform_numbers[self.index_uniform]
                self.index_uniform += 1
                ind = int(
                    temp * candid_blocks.size
                )  # choose of the candidate blocks randomly
                ind2 = candid_blocks[ind]  # index of block to increase in size
                self.depend.block_sizes[
                    ind2
                ] += 1  # increase the size of this block by one
                if (
                    self.depend.block_sizes[ind2] >= self.depend.block_size_max
                ):  # if this block size is equal or greater than the upper limit
                    candid_blocks = np.setdiff1d(
                        candid_blocks, ind2
                    )  # do not consider this block for enlarging in the next iteration
            # now given the block sizes and random ordering of variables (rand_perm), assigns variables to blocks
            ind = 0
            for k in range(self.depend.n_blocks):
                these_dims = rand_perm[ind : ind + self.depend.block_sizes[k]]
                self.depend.base_struct[k] = these_dims.astype(int)
                ind += self.depend.block_sizes[k]

    def det_depend_struct(self):
        """************* set all dependency structures for all modes by perturbation of base dependency structure **************"""
        for glob_no in range(self.n_minima):
            self.depend.struct.append(
                copy.deepcopy(self.depend.base_struct)
            )  # the dependency structure of the mode initially gets the base dependency structure
            if (
                self.depend.n_blocks < self.dim and self.depend.n_blocks > 1
            ):  # apply random perturbation (unless special case of fully separable or fully rotated, for each perturbation is meaningless)
                for swap_count in range(
                    self.depend.n_swap
                ):  # apply a predefined number of swaps
                    temp = self.uniform_numbers[
                        self.index_uniform : self.index_uniform + 2
                    ]
                    self.index_uniform += temp.size
                    indexes = (temp * self.dim).astype(
                        int
                    )  # choose two indexes from 0 to dim-1 to be swapped
                    # find the block_no and element number for each
                    block_size_cumsum = np.cumsum(
                        self.depend.block_sizes
                    )  # cumulative sizes of block sizes
                    block_ind = np.sum(
                        indexes.reshape(-1, 1) >= block_size_cumsum, axis=1
                    )  # indexes of blocks of dimensions to swap
                    block_size_cumsum_plus = np.hstack(
                        (np.array([0]), block_size_cumsum)
                    )  # put zero before cumulative sum
                    index_in_block = (
                        indexes - block_size_cumsum_plus[block_ind]
                    )  # indexes in the corresponding blocks
                    # Now having indexes of blocks and indexes at blocks of both variables, swap them
                    temp = (
                        self.depend.struct[glob_no][block_ind[0]][index_in_block[0]] + 0
                    )  # keep the first value that should be swapped
                    self.depend.struct[glob_no][block_ind[0]][index_in_block[0]] = (
                        self.depend.struct[glob_no][block_ind[1]][index_in_block[1]] + 0
                    )
                    self.depend.struct[glob_no][block_ind[1]][index_in_block[1]] = temp

            if 1:  # c=heck point only-
                term1 = np.zeros(0)
                term2 = np.zeros(0)
                for k in range(self.depend.n_blocks):
                    term1 = np.hstack((term1, self.depend.base_struct[k]))
                    term2 = np.hstack((term2, self.depend.struct[glob_no][k]))
                term1 = np.array(term1)
                term2 = np.array(term2)
                # print(np.sum(term1==term2))
                if (
                    np.sum(term1 == term2) < self.dim - self.depend.n_swap * 2
                ):  # if variation is more than the upper limit
                    print(
                        "Error: variation in dependency blocks is more than the upper limit"
                    )
                    sys.exit(1)
                if np.max(np.abs(np.sort(term1) - np.arange(self.dim))) > 0.0001:
                    print(
                        "Error: some dimensions are not in the base dependency structure"
                    )
                    sys.exit(1)
                if np.max(np.abs(np.sort(term2) - np.arange(self.dim))) > 0.0001:
                    print(
                        "Error: some dimensions are not in the dependency strcuture for this mode"
                    )
                    sys.exit(1)

    def det_rotation_mat(self):
        """******************************* determine the rotation matrices for modes***********************************"""
        self.rotation.mat = [
            np.eye(self.dim) for k in range(self.n_minima)
        ]  # default value
        for glob_no in range(self.n_minima):  # for each mode
            keep = np.arange(self.dim)  # for checkpoint
            for block_no in range(self.depend.n_blocks):  # for each block of this mode
                block_size = self.depend.struct[glob_no][block_no].size
                if block_size > 1:  # create the subspace rotation matrix
                    temp0 = (
                        2 * block_size
                    )  # you need this number of random numbers (normal distribution) to form the rotation matrix for this block
                    temp_uv = self.normal_numbers[
                        self.index_normal : self.index_normal + temp0
                    ]
                    self.index_normal += temp0  # get first temp0 numbers of the sequence of Normal numbers
                    u0 = temp_uv[
                        :block_size
                    ]  # First random vector to create the rotation matrix
                    v0 = temp_uv[
                        block_size:
                    ]  # # Second random vector to create the rotation matrix
                    angle = (
                        self.uniform_numbers[self.index_uniform]
                        * self.rotation.angle_max
                    )
                    self.index_uniform += 1  # the rotation angle
                    subspace_rot_mat = UtilityMethod.gen_rot_mat_pseudo(
                        u0, v0, angle
                    )  # create the subspace rotation matrix
                    # now replace the corresponding elements of the full rotation matrix for this mode by the elements of the subspace rotation matrix

                    temp = list(self.depend.struct[glob_no][block_no])
                    self.rotation.mat[glob_no][np.ix_(temp, temp)] = subspace_rot_mat[
                        :, :
                    ]
                    if not np.all(
                        np.isin(np.array(temp), keep)
                    ):  # checkpoint: the modified rows/columns must not have previously been modified because there is no overlap among blocks
                        print(
                            "Error: Why there is an overlap between subspaces for rotation?"
                        )
                        sys.exit(1)
                    keep = np.setdiff1d(keep, np.array(temp))

    # objective function accepts a matrix where each row is a solution
    def func_eval(self, x0):
        x = np.atleast_2d(x0)
        N = x.shape[0]
        f = np.zeros(N)
        for k in np.arange(1, N + 1):
            f[k - 1] = self._func_eval_single(x[k - 1, :])
        return f

    def _func_eval_single(self, x):  # calculate the fitness of the solution x
        F = np.zeros(self.n_minima)
        # fitness values
        for k in np.arange(
            1, self.n_minima + 1
        ):  # calculate the fitness value for each basic function independently
            shift = self.minima.X[k - 1, :]
            x_rot = (x - shift) @ self.rotation.mat[k - 1]
            F[k - 1] = BasicFun.evaluate(
                x_rot / self.lambda0, self.minima.hard_GO[k - 1], self.fun_id
            )
            pass
        # Calculate weights
        dis = cdist(np.atleast_2d(x), np.atleast_2d(self.minima.X)).ravel()
        # distance of the point to all global basins
        norm_dis2 = (dis / (self.sigma_width * self.minima.niche_rad)) ** 2
        norm_dis2min = np.min(norm_dis2)
        if (
            norm_dis2min <= 1
        ):  # some addition in case all distances are too large and all weights might become zero
            C0 = 0
        else:
            C0 = 1 - norm_dis2min
        weights = np.exp(-norm_dis2 - C0)
        # raw weight of each basic function on the fitness of solution x
        max_weights = np.max(weights)
        # highest weight
        chk = np.abs(weights - max_weights) < 1e-14
        weights = (
            weights * (1 - max_weights**10) * (1 - chk) + weights * chk
        )  # the highest raw weight does not change, the rest share the rest
        weights = weights / np.sum(weights)
        # adjusts weights
        f = np.sum(F * weights)
        # the  fitness is the weighted average all basic functions
        self.used_eval = self.used_eval + 1
        # update this property
        return f  + self.minima.f


class Rotation:
    """The class for string data for rotation of the landscapes, including full weak and subspace but strong rotations"""

    __slots__ = ["angle_max", "all_angles", "mat"]

    def __init__(self):  # requires functions ID and problem dimensionality
        self.angle_max = np.pi  # upper limit for the angles of rotations
        self.all_angles = "NA"  #  rotation angles for modes
        self.mat = None
        # Rotation matrix for each basic function

    def __str__(self):  # display the problem
        output = (
            "\n\tangle_max = "
            + str(self.angle_max)
            + "\n\tall_angles = "
            + str(self.all_angles)
        )
        return output


class Dependency:
    """The class for string data for dependency (interactions among variable) of the problem"""

    __slots__ = [
        "n_blocks",
        "block_size_delta_coeff",
        "n_swap",
        "block_size_min",
        "block_size_mean",
        "block_size_max",
        "block_sizes",
        "base_struct",
        "struct",
    ]

    def __init__(
        self, pid, dim, instance_no, max_instance_no
    ):  # requires functions ID and problem dimensionality

        folder = os.path.dirname(__file__)  # folder of current class file
        path2file = os.path.join(folder, "data/pid-data.xlsx")  # path to data file
        data = pd.read_excel(path2file)

        self.n_blocks = 1 + int(
            (instance_no - 1) / (max_instance_no - 1) * (dim - 1) + 0.5
        )  # number of blocks
        self.block_size_delta_coeff = data.block_size_delta_coeff[pid - 1]
        # variation in block sizes
        self.n_swap = 1 + int(
            dim * data.n_swap_coeff[pid - 1]
        )  # number of random swaps applied to each connectivity
        self.block_size_mean = 1 / self.n_blocks * dim  # mean value for the block size
        temp = int(
            np.min(
                (
                    np.floor(self.block_size_mean),
                    np.floor(
                        0.5 + self.block_size_mean * (1 - self.block_size_delta_coeff)
                    ),
                )
            )
        )
        self.block_size_min = np.max((1, temp))  # minimum value for block size
        self.block_size_max = int(
            np.max(
                (
                    np.ceil(self.block_size_mean),
                    np.floor(
                        0.5 + self.block_size_mean * (1 + self.block_size_delta_coeff)
                    ),
                )
            )
        )
        self.block_sizes = None  # actual sizes for the blocks
        self.base_struct = [
            [] for k in range(self.n_blocks)
        ]  # mean (base) dependency structure
        self.struct = (
            []
        )  # all connectivity structures for all mode. for each mode, the dependency structure is determined by some random swaps of the base structure

    def __str__(self):  # display the problem
        output = (
            "\n\tn_blocks = "
            + str(self.n_blocks)
            + "\n\tblock_size_delta_coeff = "
            + str(self.block_size_delta_coeff)
            + "\n\tn_swap= "
            + str(self.n_swap)
            + "\n\tblock_size_min= "
            + str(self.block_size_min)
            + "\n\tblock_size_mean= "
            + str(self.block_size_mean)
            + "\n\tblock_size_max= "
            + str(self.block_size_max)
            + "\n\tbase_struct "
            + str(self.base_struct)
            + "\n\tblock_size_delta_coeff = "
            + str(self.block_size_delta_coeff)
            #'\n\tstruct \n = ' + str(self.struct)
        )
        return output


class Minima:
    """The class for string data for rotation of the landscapes, including full weak and subspace but strong rotations"""

    __slots__ = ["X", "f", "hard_GO", "range_coeff", "crowd_basin_ind", "niche_rad"]

    def __init__(self):  # requires functions ID and problem dimensionality
        self.X = None  # (matrix) global minima of the static problem (or at time step #0 if the problem is dynamic)
        self.f = None  # the global minimum value (scalar)
        self.hard_GO = (
            None  # hardness of finding each global minimum from GO perspective
        )
        self.range_coeff = 0.9
        # global minima are inside this fraction of each dimensionality, excluding close to bounds regions
        self.crowd_basin_ind = None
        # index of the global minimum that other solutions are redistributed wrt
        self.niche_rad = None
        # Niching radius

    def __str__(self):  # display the problem
        output = (
            "\n\tX = "
            + str(self.X)
            + "\n\tf = "
            + str(self.f)
            + "\n\thard_GO = "
            + str(self.hard_GO)
            + "\n\trange_coeff = "
            + str(self.range_coeff)
            + "\n\tcrowd_basin_ind = "
            + str(self.minima.crowd_basin_ind)
            + "\n\tniche_rad = "
            + str(self.niche_rad)
        )
        return output


if __name__ == "__main__":  # a simple test of this class
    pid = 6
    instance_no = 3
    dim = 20
    problem = ProblemMM(pid, instance_no, dim)
    # problem.rotation.angle_max=np.pi*.5
    # problem.n_minima=1
    problem.form()
    X = np.ones(problem.dim)
    f = problem.func_eval(X)
    print("rotation matrix for the first mode=\n", problem.rotation.mat[0])
    print("base dependency structure=\n", problem.depend.base_struct)
    print("f(X)=", f)
