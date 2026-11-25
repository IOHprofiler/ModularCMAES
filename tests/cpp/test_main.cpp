// C++ unit tests for behaviour that cannot be reached from Python (run with ctest).
// Deliberately without a test framework: each test is a function, CHECK reports and counts failures.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
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

// #84: the evaluation queue. Requests are answered in push order; a consumer may push a follow-up,
// which forms the next batch; wrong sizes and an empty queue are errors.
static void test_evaluation_queue()
{
    parameters::Settings settings(2, {});
    settings.verbose = false;
    parameters::Parameters p(settings);
    evaluation::EvaluationQueue q(2);
    std::vector<std::string> order;
    q.push(
        Matrix::Constant(2, 1, 1.0),
        [&order, &q](parameters::Parameters &, const evaluation::Evaluations &e) {
            order.push_back("a" + std::to_string(static_cast<int>(e.f(0))));
            q.push(Matrix::Constant(2, 1, 9.0),
                   [&order](parameters::Parameters &, const evaluation::Evaluations &e2) {
                       order.push_back("c" + std::to_string(static_cast<int>(e2.f(0))));
                   },
                   "c");
        },
        "a");
    q.push(
        Matrix::Constant(2, 2, 2.0),
        [&order](parameters::Parameters &, const evaluation::Evaluations &e) {
            CHECK(e.f.size() == 2 && e.X.cols() == 2 && !e.given);
            order.push_back("b" + std::to_string(static_cast<int>(e.f.sum())));
        },
        "b");
    CHECK(q.size() == 3);
    CHECK((q.sources() == std::vector<std::string>{"a", "b"}));
    CHECK(q.batch().col(0)(0) == 1.0 && q.batch().col(2)(0) == 2.0);

    bool raised = false;
    try
    {
        q.answer(p, Vector::Zero(2));
    }
    catch (const std::invalid_argument &)
    {
        raised = true;
    }
    CHECK(raised);

    Vector f(3);
    f << 1.0, 2.0, 3.0;
    q.answer(p, f);
    CHECK((order == std::vector<std::string>{"a1", "b5"}));
    CHECK(q.size() == 1 && q.batch().col(0)(0) == 9.0); // the follow-up: the next batch
    q.answer(p, Vector::Constant(1, 7.0));
    CHECK((order == std::vector<std::string>{"a1", "b5", "c7"}));
    CHECK(q.empty());

    raised = false;
    try
    {
        q.answer(p, Vector::Zero(0));
    }
    catch (const std::logic_error &)
    {
        raised = true;
    }
    CHECK(raised);
}

// #84: a request whose number of points varies must fit the budget
static void test_push_variable_refuses()
{
    parameters::Settings settings(2, {});
    settings.verbose = false;
    settings.budget = 10;
    parameters::Parameters p(settings);
    const auto ignore = [](parameters::Parameters &, const evaluation::Evaluations &) {};
    p.stats.evaluations = 4;
    CHECK(p.budget_left() == 6);
    p.push_variable(Matrix::Zero(2, 5), ignore, "five");
    CHECK(p.budget_left() == 1);
    bool raised = false;
    try
    {
        p.push_variable(Matrix::Zero(2, 2), ignore, "two");
    }
    catch (const std::logic_error &)
    {
        raised = true;
    }
    CHECK(raised);
    CHECK(p.queue.size() == 5);
}

// A module that re-evaluates a random number (0..3) of the best points after the population and
// averages the values: a variable number of evaluations per generation (#84)
static void add_random_reevaluation(parameters::Parameters &p)
{
    p.pipeline.on(evaluation::GenerationStage::POPULATION_EVALUATED, "test-reevaluation",
                  [](parameters::Parameters &q) {
                      std::uniform_int_distribution<int> draw(0, 3);
                      auto k = static_cast<Eigen::Index>(draw(*q.generator));
                      k = std::min<Eigen::Index>(k, static_cast<Eigen::Index>(q.budget_left()));
                      k = std::min<Eigen::Index>(k, static_cast<Eigen::Index>(q.pop.n));
                      if (k == 0)
                          return;
                      std::vector<Eigen::Index> idx(static_cast<size_t>(q.pop.n));
                      std::iota(idx.begin(), idx.end(), 0);
                      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                                        [&q](auto a, auto b) { return q.pop.f(a) < q.pop.f(b); });
                      idx.resize(static_cast<size_t>(k));
                      Matrix X = q.pop.X_transformed(Eigen::all, idx);
                      q.push_variable(
                          X,
                          [idx](parameters::Parameters &r, const evaluation::Evaluations &e) {
                              r.stats.evaluations += static_cast<size_t>(e.f.size());
                              for (size_t j = 0; j < idx.size(); ++j)
                                  r.pop.f(idx[j]) = 0.5 * (r.pop.f(idx[j]) + e.f(static_cast<Eigen::Index>(j)));
                          },
                          "test-reevaluation");
                  });
}

// #84: with a variable number of evaluations per generation, ask-tell (in any batches) gives the
// trajectory of run(), and the budget is never exceeded
static void test_variable_evaluations_module()
{
    for (int config = 0; config < 3; config++)
    {
        auto settings = ask_tell_settings(config);
        settings.budget = 2003; // not a multiple of anything
        settings.seed = 11;
        FunctionType f = rastrigin;

        ModularCMAES a(settings);
        add_random_reevaluation(*a.p);
        a(f);

        ModularCMAES b(settings);
        add_random_reevaluation(*b.p);
        size_t rounds = 0, generations = 0;
        while (!b.break_conditions())
        {
            bool done = false;
            while (!done)
            {
                const Matrix X = b.ask();
                Vector values(X.cols());
                for (Eigen::Index j = 0; j < X.cols(); ++j)
                    values(j) = rastrigin(X.col(j));
                done = b.tell(values);
                rounds++;
            }
            generations++;
        }
        CHECK(a.p->stats.evaluations == b.p->stats.evaluations);
        CHECK(a.p->adaptation->m == b.p->adaptation->m);
        CHECK(a.p->mutation->sigma == b.p->mutation->sigma);
        CHECK(a.p->stats.evaluations <= settings.budget);
        CHECK(rounds > generations); // some generations asked twice (the re-evaluations)
    }
}

// #84: a subscriber with a fixed number of points per generation (the mean, as HE-ES's f(m)) is
// reserved by the budget check; evaluations in SELECTED cannot run in select() (no objective)
static void test_fixed_points_and_decomposed_steps()
{
    auto settings = ask_tell_settings(2);
    settings.budget = 1000;
    ModularCMAES es(settings);
    es.p->pipeline.on(
        evaluation::GenerationStage::GENERATION_STARTING, "test-mean",
        [](parameters::Parameters &q) {
            q.queue.push(q.coordinate_mapping->transform(q.adaptation->m),
                         [](parameters::Parameters &r, const evaluation::Evaluations &) {
                             r.stats.evaluations++;
                         },
                         "test-mean");
        },
        [](const parameters::Parameters &) { return size_t{1}; });
    CHECK(es.p->pipeline.fixed_points(*es.p) == 1);
    FunctionType f = rastrigin;
    es(f);
    CHECK(es.p->stats.evaluations <= settings.budget);
    CHECK(es.p->stats.evaluations + es.p->lambda + 1 > settings.budget);

    ModularCMAES d(ask_tell_settings(2));
    d.p->pipeline.on(evaluation::GenerationStage::SELECTED, "test-after-selection",
                     [](parameters::Parameters &q) {
                         q.queue.push(q.coordinate_mapping->transform(q.adaptation->m),
                                      [](parameters::Parameters &, const evaluation::Evaluations &) {},
                                      "test-after-selection");
                     });
    d.mutate(f);
    bool raised = false;
    try
    {
        d.select();
    }
    catch (const std::logic_error &)
    {
        raised = true;
    }
    CHECK(raised);
}

int main()
{
    test_uniform_is_open_interval();
    test_uniform_values_unchanged();
    test_ask_tell_matches_classic_loop();
    test_tell_errors();
    test_evaluation_queue();
    test_push_variable_refuses();
    test_variable_evaluations_module();
    test_fixed_points_and_decomposed_steps();
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
