#pragma once

#include "common.hpp"
#include "modules.hpp"
#include "restart_criteria.hpp"

namespace parameters
{
struct Parameters;
}

namespace restart
{
struct Strategy
{
    //! The most generations of the current run: `ExceededMaxIter` uses the smaller of this and its
    //! own limit. No limit, except in BIPOP's small runs
    size_t max_iter = std::numeric_limits<size_t>::max();

    virtual Float update(parameters::Parameters &p);
};

struct IPOP : Strategy
{
    Float ipop_factor = 2.0;
    //! The largest population: max_lambda_factor * lambda0 (2^9, the framework paper)
    size_t max_lambda_factor = 512;
    Float update(parameters::Parameters &) override;
};

struct BIPOP : Strategy
{
    size_t lambda_init;
    Float mu_factor;

    size_t lambda_large = 0;
    size_t lambda_small = 0;
    //! The evaluations used by the runs of each regime. They only grow (the budgets left, which
    //! every run decreased, wrapped around, #80). The first run counts as small (pycma)
    size_t used_small = 0;
    size_t used_large = 0;
    //! The evaluations up to the last restart
    size_t used_budget = 0;
    //! Whether the current run belongs to the large regime (not the first run)
    bool large_run = false;

    BIPOP(const Float mu, const Float lambda) :
        Strategy(),
        lambda_init(static_cast<size_t>(lambda)),
        mu_factor(mu / lambda)
    {
    }

    Float update(parameters::Parameters &) override;

    //! Whether the next run is large: the large regime has used no more evaluations than the
    //! small one (Hansen 2009; pycma's fmin)
    bool large() const { return used_large <= used_small; }
};

namespace strategy
{
inline std::shared_ptr<Strategy>
get(const parameters::Modules modules, const Float mu, const Float lambda)
{
    using namespace parameters;
    switch (modules.restart_strategy)
    {
    case RestartStrategyType::IPOP:
        return std::make_shared<IPOP>();
    case RestartStrategyType::BIPOP:
        return std::make_shared<BIPOP>(mu, lambda);
    default:
    case RestartStrategyType::STOP:
    case RestartStrategyType::NONE:
    case RestartStrategyType::RESTART:
        return std::make_shared<Strategy>();
    }
}
} // namespace strategy
} // namespace restart