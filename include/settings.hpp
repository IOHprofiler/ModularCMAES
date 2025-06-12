#pragma once

#include "common.hpp"
#include "modules.hpp"

namespace parameters
{
	struct Settings
	{
		size_t dim;
		Modules modules;

		std::optional<Float> target;
		std::optional<size_t> max_generations;
		size_t budget;

		Float sigma0;
		size_t lambda0;
		size_t mu0;

		std::optional<Vector> x0;
		Vector lb;
		Vector ub;
		std::optional<Float> cs;
		std::optional<Float> cc;
		std::optional<Float> cmu;
		std::optional<Float> c1;
		std::optional<Float> damps;
		std::optional<Float> acov;
		bool verbose;
		Float volume;
		bool one_plus_one;

		Settings(size_t dim,
			std::optional<Modules> mod = std::nullopt,
			std::optional<Float> target = std::nullopt,
			std::optional<size_t> max_generations = std::nullopt,
			std::optional<size_t> budget = std::nullopt,
			std::optional<Float> sigma = std::nullopt,
			std::optional<size_t> lambda = std::nullopt,
			std::optional<size_t> mu = std::nullopt,
			std::optional<Vector> x0 = std::nullopt,
			std::optional<Vector> lb = std::nullopt,
			std::optional<Vector> ub = std::nullopt,
			std::optional<Float> cs = std::nullopt,
			std::optional<Float> cc = std::nullopt,
			std::optional<Float> cmu = std::nullopt,
			std::optional<Float> c1 = std::nullopt,
			std::optional<Float> damps = std::nullopt,
			std::optional<Float> acov = std::nullopt,
			bool verbose = true,
			bool always_compute_eigv = false
		) : dim(dim),
			modules(mod.value_or(Modules())),
			target(target),
			max_generations(max_generations),
			budget(budget.value_or(dim * 1e4)),
			sigma0(sigma.value_or(2.0)),
			lambda0(lambda.value_or(4 + std::floor(3 * std::log(dim)))),
			mu0(mu.value_or(lambda0 / 2)),
			x0(x0),
			lb(lb.value_or(Vector::Ones(dim) * -5)),
			ub(ub.value_or(Vector::Ones(dim)* 5)),
			cs(cs),
			cc(cc),
			cmu(cmu),
			c1(c1),
			damps(damps),
			acov(acov),
			verbose(verbose),
			volume(0.0),
			one_plus_one(false)
		{
			if (modules.mirrored == Mirror::PAIRWISE and lambda0 % 2 != 0)
				lambda0++;

			if (mu0 > lambda0)
			{
				mu0 = lambda0 / 2;
			}

			if (modules.ssa == StepSizeAdaptation::SA || modules.matrix_adaptation == MatrixAdaptationType::CMSA)
			{
				mu0 = mu.value_or(lambda0 / 4);
			}


			if (modules.ssa != StepSizeAdaptation::CSA
				and modules.matrix_adaptation == MatrixAdaptationType::COVARIANCE
				and not always_compute_eigv
				)
			{
				modules.matrix_adaptation = MatrixAdaptationType::COVARIANCE_NO_EIGV;
			}

			if (
				modules.matrix_adaptation == MatrixAdaptationType::NONE
			)
			{
				modules.active = false;
			}

			if (lambda0 == 1)
			{
				mu0 = 1;
				one_plus_one = true;
				modules.elitist = true;
				modules.active = false;
				modules.sequential_selection = false;
				modules.weights = RecombinationWeights::EQUAL;
				modules.ssa = StepSizeAdaptation::SR;
			
				if (modules.restart_strategy == RestartStrategyType::BIPOP || modules.restart_strategy == RestartStrategyType::IPOP)
					modules.restart_strategy = RestartStrategyType::RESTART;
			}
			volume = (this->ub.cwiseMin(10 * sigma0) - this->lb.cwiseMax(-10 * sigma0)).prod();
		}
	};

}