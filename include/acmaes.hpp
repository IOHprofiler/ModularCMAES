#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <float.h>
#include <stdint.h>
#include <ctime>
#include "evaluator.h"


namespace acmaes {

    class AcmaesOptimizer {

    public:

        AcmaesOptimizer(long runid_, Fitness* fitfun_, int popsize_, int mu_,
            const vec& guess_, const vec& inputSigma_, int maxEvaluations_,
            double accuracy_, double stopfitness_, double stopTolHistFun_,
            int update_gap_, long seed);


        ~AcmaesOptimizer();
        // param zmean weighted row matrix of the gaussian random numbers generating the current offspring
        // param xold xmean matrix of the previous generation
        // return hsig flag indicating a small correction

        bool updateEvolutionPaths(const vec& zmean, const vec& xold);

        // param hsig flag indicating a small correction
        // param bestArx fitness-sorted matrix of the argument vectors producing the current offspring
        // param arz unsorted matrix containing the gaussian random values of the current offspring
        // param arindex indices indicating the fitness-order of the current offspring
        // param xold xmean matrix of the previous generation

        double updateCovariance(bool hsig, const mat& bestArx, const mat& arz,
            const ivec& arindex, const mat& xold);

        // Update B and diagD from C
        // param negccov Negative covariance factor.

        void updateBD(double negccov);
        mat ask_all();
        int tell_all(mat ys, mat xs);
        int tell_all_asked(mat ys, mat xs);
        mat getPopulation();
        vec ask();
        int tell(double y, const vec& x);
        void updateCMA();
        int doOptimize();
        int do_optimize_delayed_update(int workers);
        vec getBestX();
        double getBestValue();
        double getIterations();
        int getStop();
        Fitness* getFitfun();
        int getDim();
        int getPopsize();
        Fitness* getFitfunPar();
        mat popX;

        int n_updates;

    private:
        long runid;
        Fitness* fitfun;
        vec guess;
        double accuracy;
        int popsize; // population size
        vec inputSigma;
        int dim;
        int maxEvaluations;
        double stopfitness;
        double stopTolUpX;
        double stopTolX;
        double stopTolFun;
        double stopTolHistFun;
        int mu; //
        vec weights;
        double mueff; //
        double sigma;
        double cc;
        double cs;
        double damps;
        double ccov1;
        double ccovmu;
        double chiN;
        double ccov1Sep;
        double ccovmuSep;
        double lazy_update_gap = 0;
        vec xmean;
        vec pc;
        vec ps;
        double normps;
        mat B;
        mat BD;
        mat diagD;
        mat C;
        vec diagC;
        mat arz;
        mat arx;
        vec fitness;
        int iterations = 0;
        int last_update = 0;
        vec fitnessHistory;
        int historySize;
        double bestValue;
        vec bestX;
        int stop;
        int told = 0;
        pcg64* rs;
        bool computeArz;
    };
}
