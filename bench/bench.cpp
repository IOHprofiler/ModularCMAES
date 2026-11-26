/*
 * Benchmark harness for the C++ ModularCMAES (dev/overhaul-2.0.md, T0.2).
 *
 * Runs a fixed number of generations on the sphere function (a C++ objective, so the numbers are the
 * algorithm's own cost) and reports, per configuration and dimension:
 *   - microseconds per generation (minimum over the repetitions),
 *   - the share of the four steps of a generation: mutate (sampling + evaluation), select,
 *     recombine, adapt (evolution paths, step size, matrix update, restart criteria),
 *   - heap allocations per evaluation in the steady state (where the platform allows counting).
 *
 * Usage: modcma_bench [--dims 2,10,40,100] [--generations 2000] [--repetitions 5]
 *                     [--configs default,matrix,separable] [--csv]
 */
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "c_maes.hpp"

extern "C" unsigned long long modcma_malloc_count(void);
extern "C" int modcma_malloc_count_supported(void);

namespace
{
using Clock = std::chrono::steady_clock;

struct Result
{
    size_t lambda = 0;
    double us_per_generation = 0;
    double share[4] = {0, 0, 0, 0};
    double allocations_per_evaluation = -1;
};

Float sphere(Eigen::Ref<const Vector> x) { return x.squaredNorm(); }

//! A configuration name with the suffix "-fast" uses Settings.fast_sampling
bool is_fast(const std::string &name)
{
    return name.size() > 5 && name.compare(name.size() - 5, 5, "-fast") == 0;
}

parameters::Modules make_modules(std::string name)
{
    if (is_fast(name))
        name.resize(name.size() - 5);
    parameters::Modules m;
    if (name == "default")
        return m;
    if (name == "matrix")
        m.matrix_adaptation = parameters::MatrixAdaptationType::MATRIX;
    else if (name == "separable")
        m.matrix_adaptation = parameters::MatrixAdaptationType::SEPARABLE;
    else if (name == "cholesky")
        m.matrix_adaptation = parameters::MatrixAdaptationType::CHOLESKY;
    else if (name == "none")
        m.matrix_adaptation = parameters::MatrixAdaptationType::NONE;
    else if (name == "active")
        m.active = true;
    else if (name == "elitist")
        m.elitist = true;
    else if (name == "mirrored-orthogonal")
    {
        m.mirrored = parameters::Mirror::MIRRORED;
        m.orthogonal = true;
    }
    else
    {
        std::fprintf(stderr, "unknown configuration: %s\n", name.c_str());
        std::exit(2);
    }
    return m;
}

Result run_once(const std::string &config, const int dim, const size_t generations)
{
    rng::set_seed(1);
    parameters::Settings settings(dim, make_modules(config), std::nullopt, generations,
                                  std::numeric_limits<size_t>::max() / 2, 2.0);
    settings.verbose = false;
    settings.fast_sampling = is_fast(config);
    auto p = std::make_shared<parameters::Parameters>(settings);
    ModularCMAES cma(p);
    FunctionType objective = sphere;

    // Warm up: the first generations allocate the buffers that later generations may reuse.
    const size_t warmup = std::min<size_t>(10, generations / 10);
    for (size_t g = 0; g < warmup; g++)
        cma.step(objective);

    const auto evaluations0 = p->stats.evaluations;
    const auto allocations0 = modcma_malloc_count();
    double t[4] = {0, 0, 0, 0};
    size_t measured = 0;
    // Generations only: stop_conditions(), not break_conditions(), which holds after the final
    // center of mass (one evaluation, not a generation)
    while (!cma.stop_conditions())
    {
        const auto a = Clock::now();
        cma.mutate(objective);
        const auto b = Clock::now();
        cma.select();
        const auto c = Clock::now();
        cma.recombine();
        const auto d = Clock::now();
        cma.adapt();
        const auto e = Clock::now();
        using us = std::chrono::duration<double, std::micro>;
        t[0] += us(b - a).count();
        t[1] += us(c - b).count();
        t[2] += us(d - c).count();
        t[3] += us(e - d).count();
        measured++;
    }

    Result r;
    r.lambda = p->lambda;
    const double total = t[0] + t[1] + t[2] + t[3];
    r.us_per_generation = total / static_cast<double>(std::max<size_t>(1, measured));
    for (int i = 0; i < 4; i++)
        r.share[i] = total > 0 ? 100.0 * t[i] / total : 0.0;
    const auto evaluations = p->stats.evaluations - evaluations0;
    if (modcma_malloc_count_supported() && evaluations > 0)
        r.allocations_per_evaluation = static_cast<double>(modcma_malloc_count() - allocations0) /
                                       static_cast<double>(evaluations);
    return r;
}

std::vector<std::string> split(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
        if (!item.empty())
            out.push_back(item);
    return out;
}
} // namespace

int main(int argc, char **argv)
{
    std::vector<int> dims = {2, 10, 40, 100};
    std::vector<std::string> configs = {"default", "matrix", "separable"};
    size_t generations = 2000;
    int repetitions = 5;
    bool csv = false;

    for (int i = 1; i < argc; i++)
    {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--dims")
        {
            dims.clear();
            for (const auto &d : split(value()))
                dims.push_back(std::stoi(d));
        }
        else if (arg == "--configs")
            configs = split(value());
        else if (arg == "--generations")
            generations = std::stoul(value());
        else if (arg == "--repetitions")
            repetitions = std::stoi(value());
        else if (arg == "--csv")
            csv = true;
        else if (arg == "--eigen-solver")
            constants::eigen_solver = std::stoi(value()); // 0 Eigen, 1 dsyevd, 2 dsyevr (LAPACK)
        else
        {
            std::fprintf(stderr,
                         "usage: %s [--dims 2,10,40,100] [--generations 2000] [--repetitions 5] "
                         "[--configs default,matrix,separable] [--eigen-solver 0|1|2] [--csv]\n",
                         argv[0]);
            return arg == "--help" ? 0 : 2;
        }
    }

    if (csv)
        std::printf("config,dim,lambda,us_per_generation,mutate_pct,select_pct,recombine_pct,adapt_pct,"
                    "allocations_per_evaluation\n");
    else
        std::printf("%-20s %4s %4s %12s %8s %8s %8s %8s %12s\n", "config", "d", "lam", "us/gen(min)",
                    "mutate", "select", "recomb", "adapt", "malloc/eval");

    for (const auto &config : configs)
        for (const auto dim : dims)
        {
            Result best;
            best.us_per_generation = std::numeric_limits<double>::infinity();
            for (int r = 0; r < repetitions; r++)
            {
                const auto res = run_once(config, dim, generations);
                if (res.us_per_generation < best.us_per_generation)
                    best = res;
            }
            char allocations[32];
            if (best.allocations_per_evaluation < 0)
                std::snprintf(allocations, sizeof allocations, "n/a");
            else
                std::snprintf(allocations, sizeof allocations, "%.2f", best.allocations_per_evaluation);

            if (csv)
                std::printf("%s,%d,%zu,%.3f,%.1f,%.1f,%.1f,%.1f,%s\n", config.c_str(), dim, best.lambda,
                            best.us_per_generation, best.share[0], best.share[1], best.share[2],
                            best.share[3], allocations);
            else
                std::printf("%-20s %4d %4zu %12.2f %7.1f%% %7.1f%% %7.1f%% %7.1f%% %12s\n", config.c_str(),
                            dim, best.lambda, best.us_per_generation, best.share[0], best.share[1],
                            best.share[2], best.share[3], allocations);
            std::fflush(stdout);
        }
    return 0;
}
