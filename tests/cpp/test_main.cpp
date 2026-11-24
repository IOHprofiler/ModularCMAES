// C++ unit tests for behaviour that cannot be reached from Python (run with ctest).
// Deliberately without a test framework: each test is a function, CHECK reports and counts failures.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "c_maes.hpp"
#include "lapack.hpp"
#include "restart_criteria.hpp"
#include "common.hpp"
#include "functions.hpp"
#include "sampling.hpp"

static int failures = 0;

#define CHECK(condition)                                                                                \
    do                                                                                                  \
    {                                                                                                   \
        if (!(condition))                                                                               \
        {                                                                                               \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);          \
            failures++;                                                                                 \
        }                                                                                               \
    } while (0)

// A generator that replays a fixed sequence, with the interface of std::mt19937
struct ReplayGenerator
{
    using result_type = uint32_t;
    std::vector<result_type> values;
    size_t i = 0;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<uint32_t>::max(); }
    result_type operator()() { return values.at(i++); }
};

// B5: rng::uniform returned exactly 0 and 1 for the generator's extreme outputs, which ppf and the
// other sample transformations map to +-infinity.
static void test_uniform_is_open_interval()
{
    ReplayGenerator gen{{ReplayGenerator::min(), ReplayGenerator::max(), 12345u, ReplayGenerator::max(), 7u}};
    rng::uniform<double> uniform;
    const double a = uniform(gen);
    CHECK(gen.i == 3); // the two extremes were skipped
    CHECK(a == 12345.0 / 4294967295.0);
    const double b = uniform(gen);
    CHECK(gen.i == 5);
    CHECK(b == 7.0 / 4294967295.0);
    CHECK(std::isfinite(ppf(a)) && std::isfinite(ppf(b)));
}

// Every other output maps to the same number as before the fix
static void test_uniform_values_unchanged()
{
    std::mt19937 gen(42), reference(42);
    rng::uniform<double> uniform;
    for (int k = 0; k < 100000; k++)
    {
        const auto g = reference();
        const double old_value = static_cast<double>(g - reference.min()) / reference.max() - reference.min();
        CHECK(uniform(gen) == old_value);
    }
}

// WP3: ask/tell with an objective evaluated by the caller gives the trajectory of the classic loop
// (mutate, select, recombine, adapt with the objective), bit for bit
static Float rastrigin(Eigen::Ref<const Vector> x)
{
    return 10.0 * static_cast<Float>(x.size()) +
           (x.array().square() - 10.0 * (2.0 * M_PI * x.array()).cos()).sum();
}

static parameters::Settings ask_tell_settings(const int config)
{
    parameters::Modules m;
    size_t mu = 0, lambda = 0;
    switch (config)
    {
    case 0: // sequential selection, stopped after complete pairs; TPA after a stopped population
        m.sequential_selection = true;
        m.mirrored = parameters::Mirror::PAIRWISE;
        m.ssa = parameters::StepSizeAdaptation::TPA;
        mu = 2;
        lambda = 10;
        break;
    case 1: // restarts with hill-valley tests of the repelling archive
        m.restart_strategy = parameters::RestartStrategyType::IPOP;
        m.repelling_type = parameters::RepellingType::COVERAGE;
        break;
    default: // elitism, threshold convergence, bounds
        m.elitist = true;
        m.threshold_convergence = true;
        m.bound_correction = parameters::CorrectionMethod::SATURATE;
        break;
    }
    parameters::Settings s(5,
                           m,
                           std::nullopt,
                           std::nullopt,
                           size_t{6000},
                           2.0,
                           mu ? std::optional<size_t>(mu) : std::nullopt,
                           mu ? std::optional<size_t>(lambda) : std::nullopt);
    s.verbose = false;
    return s;
}

static void test_ask_tell_matches_classic_loop()
{
    for (int config = 0; config < 3; config++)
    {
        rng::set_seed(7);
        ModularCMAES classic(ask_tell_settings(config));
        FunctionType f = rastrigin;
        std::vector<Vector> trajectory;
        for (bool running = true; running;)
        {
            classic.mutate(f);
            classic.select();
            classic.recombine();
            classic.adapt();
            running = !classic.break_conditions();
            trajectory.push_back(classic.p->adaptation->m);
            trajectory.back().conservativeResize(trajectory.back().size() + 2);
            trajectory.back().tail(2) << classic.p->mutation->sigma,
                static_cast<Float>(classic.p->stats.evaluations);
        }

        rng::set_seed(7);
        ModularCMAES ask_tell(ask_tell_settings(config));
        size_t generation = 0;
        bool same = true;
        for (bool running = true; running; generation++)
        {
            bool complete = false;
            while (!complete)
            {
                const Matrix X = ask_tell.ask();
                Vector values(X.cols());
                for (Eigen::Index j = 0; j < X.cols(); j++)
                    values(j) = rastrigin(X.col(j));
                complete = ask_tell.tell(values);
            }
            running = !ask_tell.break_conditions();
            if (generation >= trajectory.size())
            {
                same = false;
                break;
            }
            const Vector &expected = trajectory[generation];
            const auto d = ask_tell.p->adaptation->m.size();
            same = same && expected.head(d) == ask_tell.p->adaptation->m &&
                   expected(d) == ask_tell.p->mutation->sigma &&
                   expected(d + 1) == static_cast<Float>(ask_tell.p->stats.evaluations);
        }
        CHECK(same);
        CHECK(generation == trajectory.size());
    }
    // config 1 restarts: the test above covers the restart phases only if it does
    rng::set_seed(7);
    ModularCMAES cma(ask_tell_settings(1));
    FunctionType f = rastrigin;
    cma(f);
    CHECK(cma.p->stats.centers.size() >= 1);
}

static void test_tell_errors()
{
    ModularCMAES cma(parameters::Settings(3));
    bool raised = false;
    try
    {
        cma.tell(Vector::Zero(1));
    }
    catch (const std::logic_error &)
    {
        raised = true;
    }
    CHECK(raised);

    const auto k = cma.ask().cols();
    raised = false;
    try
    {
        cma.tell(Vector::Zero(k + 1));
    }
    catch (const std::invalid_argument &)
    {
        raised = true;
    }
    CHECK(raised);
    CHECK(cma.has_pending());
    CHECK(cma.tell(Vector::Ones(k)));
    CHECK(!cma.has_pending());
}

// B10: the running medians of SplitMedian equal those of a sort, for every split
static Float sorted_median(std::vector<Float> v)
{
    if (v.empty())
        return std::numeric_limits<Float>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void test_split_median()
{
    std::mt19937 gen(3);
    std::uniform_int_distribution<int> small(0, 20); // many duplicates
    restart::SplitMedian sm;
    std::vector<Float> values;
    bool same = true;
    for (int k = 0; k < 3000; k++)
    {
        Float x = static_cast<Float>(small(gen));
        if (k % 97 == 0)
            x = std::numeric_limits<Float>::infinity();
        if (k % 89 == 0)
            x = -std::numeric_limits<Float>::infinity();
        values.push_back(x);
        sm.push_back(x);
        // forward as Stagnation moves it; now and then backward
        size_t split = static_cast<size_t>(0.3 * static_cast<Float>(values.size()));
        if (k % 500 == 499)
            split /= 2;
        sm.set_split(split);
        const std::vector<Float> head(values.begin(), values.begin() + split);
        const std::vector<Float> tail(values.begin() + split, values.end());
        const auto eq = [](Float a, Float b) { return a == b || (std::isnan(a) && std::isnan(b)); };
        same = same && eq(sm.head_median(), sorted_median(head)) &&
               eq(sm.tail_median(), sorted_median(tail));
    }
    CHECK(same);
    sm.assign({3.0, 1.0, 2.0, 5.0});
    sm.set_split(1);
    CHECK(sm.head_median() == 3.0);
    CHECK(sm.tail_median() == 2.0);
}

// LAPACK eigendecomposition (build option MODCMA_LAPACK): the same decomposition as Eigen's
static void test_lapack_eigen()
{
    if (!lapack::available())
    {
        std::printf("(no LAPACK in this build: test_lapack_eigen skipped)\n");
        return;
    }
    std::mt19937 gen(5);
    std::normal_distribution<double> normal;
    for (const int n : {1, 7, 60, 150})
    {
        Matrix A(n, n);
        for (auto &a : A.reshaped())
            a = normal(gen);
        const Matrix C = A * A.transpose() + Matrix::Identity(n, n);
        const Eigen::SelfAdjointEigenSolver<Matrix> reference(C);
        {
            // LU inverse, and dgemm on blocks (outer stride) with both transposes
            Matrix inv;
            CHECK(lapack::inverse(C, inv));
            CHECK((C * inv - Matrix::Identity(n, n)).norm() < 1e-10 * n);
            Matrix singular = Matrix::Zero(n, n);
            CHECK(!lapack::inverse(singular, inv));
            Matrix big = Matrix::Random(n + 3, n + 5);
            const auto block = big.middleCols(2, n).topRows(n);
            Matrix out = Matrix::Zero(n, n);
            lapack::gemm(1.0, C, false, block, true, 0.0, out);
            CHECK((out - C * block.transpose()).norm() < 1e-12 * (C.norm() * block.norm() + 1));
            Matrix acc = Matrix::Ones(n, n);
            lapack::gemm(0.5, block, true, C, false, 2.0, acc);
            const Matrix expected = 2.0 * Matrix::Ones(n, n) + 0.5 * block.transpose() * C;
            CHECK((acc - expected).norm() < 1e-12 * (expected.norm() + 1));
        }
        {
            Matrix L;
            CHECK(lapack::cholesky(C, L));
            const Matrix reference_L = Eigen::LLT<Matrix>(C).matrixL();
            CHECK((L - reference_L).norm() < 1e-12 * reference_L.norm());
            CHECK(L.triangularView<Eigen::StrictlyUpper>().toDenseMatrix().isZero(0));
            Matrix not_pd = -C;
            CHECK(!lapack::cholesky(not_pd, L));
        }
        for (const auto method : {lapack::Method::SYEVD, lapack::Method::SYEVR})
        {
            lapack::Workspace ws;
            Vector d;
            Matrix B;
            for (int call = 0; call < 2; call++) // the second call reuses the workspace
            {
                CHECK(lapack::symmetric_eigen(C, d, B, method, ws));
                const double scale = C.norm();
                CHECK((d - reference.eigenvalues()).cwiseAbs().maxCoeff() < 1e-12 * scale);
                CHECK((B * d.asDiagonal() * B.transpose() - C).norm() < 1e-12 * scale);
                CHECK((B.transpose() * B - Matrix::Identity(n, n)).norm() < 1e-12 * n);
            }
        }
    }
}

// xNES: the low-rank update of A equals A expm(eta G) with a dense matrix exponential
static void test_natural_gradient_low_rank_update()
{
    std::mt19937 gen(9);
    std::normal_distribution<double> normal;
    // the low-rank path: lambda < d and d >= 16 (below, the update is the dense matrix exponential)
    for (const auto [d, n] : {std::pair{16, 5}, std::pair{40, 12}, std::pair{64, 30}})
    {
        matrix_adaptation::NaturalGradientAdaptation a(d, Vector::Zero(d), std::sqrt(d), 1.0);
        CHECK(!a.dense_update(n));
        for (auto &x : a.A.reshaped())
            x = 0.3 * normal(gen);
        a.A.diagonal().array() += 1.0;
        a.gradient_samples.resize(d, n);
        for (auto &x : a.gradient_samples.reshaped())
            x = normal(gen);
        a.gradient_weights.resize(n);
        for (int i = 0; i < n; i++)
            a.gradient_weights(i) = i < n / 2 ? 1.0 / (i + 1) : -0.2 / (i + 1); // active: negative
        a.gradient_shift = 0.7;

        const Matrix expected = a.A * (0.3 * a.G()).exp();
        parameters::Weights w(parameters::Settings(d), std::sqrt(d));
        w.cc = 0.3;
        Population pop(d, n);
        parameters::Settings settings(d);
        parameters::Stats stats;
        CHECK(a.adapt_matrix_inner(w, settings.modules, pop, n / 2, settings, stats));
        CHECK((a.A - expected).norm() < 1e-10 * expected.norm());
    }
}

// The test functions of functions.hpp: their minima, and the rotation is a proper rotation
static void test_functions()
{
    for (const int d : {1, 2, 10})
    {
        CHECK(functions::sphere(Vector::Zero(d)) == 0.0);
        CHECK(functions::ellipse(Vector::Zero(d)) == 0.0);
        CHECK(std::abs(functions::rastrigin(Vector::Zero(d))) < 1e-12);
        CHECK(functions::rosenbrock(Vector::Ones(d)) == 0.0);
        CHECK(std::isfinite(functions::ellipse(Vector::Ones(d))));
        CHECK(functions::sphere(Vector::Ones(d)) == static_cast<Float>(d));

        const Matrix R = functions::random_rotation_matrix(d, 42);
        CHECK((R.transpose() * R - Matrix::Identity(d, d)).cwiseAbs().maxCoeff() < 1e-12);
        CHECK(std::abs(R.determinant() - 1.0) < 1e-12);
        CHECK(R == functions::random_rotation_matrix(d, 42));
    }
    // the condition number of the ellipse is 1e6: f(e_d) / f(e_1)
    Vector e1 = Vector::Zero(10), e10 = Vector::Zero(10);
    e1(0) = 1.0;
    e10(9) = 1.0;
    CHECK(std::abs(functions::ellipse(e10) / functions::ellipse(e1) - 1e6) < 1e-6);
    CHECK(functions::get(functions::ROSENBROCK)(Vector::Ones(3)) == 0.0);
    CHECK(functions::to_string(functions::RASTRIGIN) == "rastrigin");
}

int main()
{
    test_uniform_is_open_interval();
    test_uniform_values_unchanged();
    test_ask_tell_matches_classic_loop();
    test_tell_errors();
    test_split_median();
    test_lapack_eigen();
    test_natural_gradient_low_rank_update();
    test_functions();
    if (failures)
        std::fprintf(stderr, "%d check(s) failed\n", failures);
    else
        std::printf("all checks passed\n");
    return failures ? 1 : 0;
}
