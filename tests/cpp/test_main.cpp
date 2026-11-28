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
#include "serialization.hpp"

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
        matrix_adaptation::NaturalGradientAdaptation a(d, Vector::Zero(d));
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
        CHECK(a.adapt_matrix_inner(w, pop, n / 2, settings, stats));
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

// #84: the values of a batch go, in queue order, to the module in place when they arrive (also one
// that replaced the module that queued the points); the population continues where it was
static void test_evaluation_queue()
{
    parameters::Modules m;
    m.ssa = parameters::StepSizeAdaptation::TPA;
    m.sequential_selection = true;
    parameters::Settings settings(2, m);
    settings.verbose = false;
    settings.seed = 5;
    ModularCMAES es(settings);

    // The population's first batch (individuals 0..first_stop(0)), then TPA's two points
    const Matrix X = es.ask();
    const Eigen::Index first = X.cols() - 2;
    CHECK(first >= 1 && first < static_cast<Eigen::Index>(es.p->lambda));

    // A new TPA replaces the one that queued the points: it receives their values
    es.p->mutation = mutation::get(es.p->settings.modules, 2.0, 2.0, es.p->generator);
    Vector f = Vector::Constant(X.cols(), 1.0);
    f(first) = 0.5; // the point ahead is better: TPA's signal is a success
    CHECK(!es.tell(f));
    const auto tpa = std::dynamic_pointer_cast<mutation::TPA>(es.p->mutation);
    CHECK(tpa && tpa->rank_tpa == tpa->a_tpa + tpa->b_tpa);
    CHECK(es.p->stats.evaluations == static_cast<size_t>(first) + 2);

    // The population was not stopped (the first generation has no reference): the rest of it,
    // batch by batch, and the population continues where it was
    for (bool done = false; !done;)
        done = es.tell(Vector::Constant(es.ask().cols(), 1.0));
    CHECK(es.p->stats.evaluations == es.p->lambda + 2);
}

// #90: apply_transpose_into is the transpose of the sampling transformation of compute_y_into,
// v^T (A z) = (A^T v)^T z, for every matrix adaptation after it has adapted its matrix
static void test_apply_transpose()
{
    using parameters::MatrixAdaptationType;
    FunctionType f = [](Eigen::Ref<const Vector> x) { return functions::ellipse(x); };
    for (const auto type : {MatrixAdaptationType::NONE,
                            MatrixAdaptationType::COVARIANCE,
                            MatrixAdaptationType::MATRIX,
                            MatrixAdaptationType::SEPARABLE,
                            MatrixAdaptationType::CHOLESKY,
                            MatrixAdaptationType::CMSA,
                            MatrixAdaptationType::COVARIANCE_NO_EIGV,
                            MatrixAdaptationType::NATURAL_GRADIENT})
    {
        parameters::Modules m;
        m.matrix_adaptation = type;
        parameters::Settings settings(5, m);
        settings.verbose = false;
        settings.seed = 11;
        ModularCMAES es(settings);
        for (int g = 0; g < 30; ++g)
            es.step(f);
        std::mt19937 gen(3);
        std::normal_distribution<Float> normal;
        Vector z(5), v(5), y(5), u(5);
        for (int k = 0; k < 5; ++k)
        {
            for (Eigen::Index i = 0; i < 5; ++i)
            {
                z(i) = normal(gen);
                v(i) = normal(gen);
            }
            es.p->adaptation->compute_y_into(z, y);
            es.p->adaptation->apply_transpose_into(v, u);
            const Float scale = y.norm() * v.norm() + u.norm() * z.norm();
            CHECK(std::abs(v.dot(y) - u.dot(z)) <= 1e-12 * scale);
        }
    }
}

// #90: gradient injection. The gradient point of the first population is m - sigma c g / ||g||
// (C = I), with c = E||z||; ask-tell with inject_gradient gives the same run as run() with the
// gradient; inject_gradient is refused outside a new population, and a wrong gradient is refused
// before anything changes
static void test_gradient_injection()
{
    FunctionType f = [](Eigen::Ref<const Vector> x) { return functions::ellipse(x); };
    GradientType gradient = [](Eigen::Ref<const Vector> x) {
        Vector g(x.size());
        for (Eigen::Index i = 0; i < x.size(); ++i)
            g(i) = 2.0 * std::pow(1.0e6, static_cast<Float>(i) / (x.size() - 1.0)) * x(i);
        return g;
    };
    parameters::Settings settings(5, {}, std::nullopt, std::nullopt, size_t{3000}, 0.5);
    settings.verbose = false;
    settings.seed = 8;
    settings.x0 = Vector::Ones(5);

    ModularCMAES es(settings);
    bool threw = false;
    try
    {
        es.inject_gradient(Vector::Ones(5)); // nothing asked yet
    }
    catch (const std::logic_error &)
    {
        threw = true;
    }
    CHECK(threw);
    const Matrix X = es.ask();
    CHECK(es.accepts_gradient());
    threw = false;
    try
    {
        es.inject_gradient(Vector::Ones(3));
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    CHECK(threw && es.accepts_gradient());
    const Vector g = gradient(es.center_of_mass());
    es.inject_gradient(g);
    CHECK(!es.accepts_gradient());
    const Matrix X2 = es.ask();
    const Vector expected =
        es.p->adaptation->m - 0.5 * es.p->weights.expected_length_z * g / g.norm();
    CHECK((X2.col(0) - expected).cwiseAbs().maxCoeff() < 1e-14);
    CHECK(X2.rightCols(X2.cols() - 1) == X.rightCols(X.cols() - 1));

    ModularCMAES a(settings), b(settings);
    a(f, gradient);
    while (!b.break_conditions())
        for (bool done = false; !done;)
        {
            b.ask();
            if (b.accepts_gradient())
                b.inject_gradient(gradient(b.center_of_mass()));
            const Matrix Xb = b.ask();
            Vector fb(Xb.cols());
            for (Eigen::Index j = 0; j < Xb.cols(); ++j)
                fb(j) = f(Xb.col(j));
            done = b.tell(fb);
        }
    CHECK(a.p->stats.evaluations == b.p->stats.evaluations);
    CHECK(a.p->adaptation->m == b.p->adaptation->m);
    CHECK(a.p->stats.global_best.y == b.p->stats.global_best.y);
}

// (c) A restart that starts now fits the budget when its next point leaves room for the smallest
// first generation of the next run, the points every generation reserves (TPA's two) and the
// evaluation of the next run's center of mass (#111)
static void test_reserved_points()
{
    parameters::Modules m;
    m.ssa = parameters::StepSizeAdaptation::TPA;
    parameters::Settings settings(2, m, std::nullopt, std::nullopt, size_t{100});
    settings.verbose = false;
    parameters::Parameters p(settings);
    CHECK(p.reserved_points() == 2);
    const size_t room = parameters::Parameters::min_lambda_after_restart + 2 + 1;
    p.stats.evaluations = 100 - room - 1;
    CHECK(p.restart_fits());
    p.stats.evaluations = 100 - room;
    CHECK(!p.restart_fits());
}

// (c) The hill-valley tests of repelling (a variable number of points, at a restart) leave the
// first generation of the next run, with TPA's two points, inside the budget: for every budget,
// run() stays within it, and ask-tell gives the same run
static void test_budget_with_restarts()
{
    // Flat: the restart criteria fire often, and every hill-valley test has all its 10 points, so a
    // chain that stops after another number was cut short by the budget
    FunctionType f = [](Eigen::Ref<const Vector>) { return 1.0; };
    size_t cut_short = 0;
    for (size_t budget = 60; budget <= 600; budget += 3)
    {
        parameters::Modules m;
        m.ssa = parameters::StepSizeAdaptation::TPA;
        m.restart_strategy = parameters::RestartStrategyType::IPOP;
        m.repelling_type = parameters::RepellingType::COVERAGE;
        parameters::Settings settings(2, m, std::nullopt, std::nullopt, budget, 2.0);
        settings.verbose = false;
        settings.seed = 4;

        ModularCMAES a(settings);
        a(f);
        CHECK(a.p->stats.evaluations <= budget);

        ModularCMAES b(settings);
        size_t chain = 0;
        bool previous_single = false;
        while (!b.break_conditions())
            for (bool done = false; !done;)
            {
                const Matrix X = b.ask();
                // A restart asks for the final mean (one point); single points right after it are
                // the hill-valley points (the populations here have at least 4)
                const bool single = X.cols() == 1;
                if (single && previous_single)
                    chain++;
                else if (chain != 0)
                {
                    cut_short += chain % 10 != 0;
                    chain = 0;
                }
                previous_single = single;
                done = b.tell(Vector::Ones(X.cols()));
            }
        CHECK(a.p->stats.evaluations == b.p->stats.evaluations);
        CHECK(a.p->adaptation->m == b.p->adaptation->m);
    }
    CHECK(cut_short > 0); // the budget did cut tests short
}

// #117: LAPACK asked for (constants) but not in the build: the decompositions use Eigen instead
// of failing at every lazy update (a restart each time)
static void test_lapack_fallback()
{
    if (lapack::available())
    {
        std::printf("(LAPACK in this build: test_lapack_fallback skipped)\n");
        return;
    }
    const int eigen_solver = constants::eigen_solver;
    const int cholesky_solver = constants::cholesky_solver;
    constants::eigen_solver = 1;
    constants::cholesky_solver = 1;
    for (const auto type : {parameters::MatrixAdaptationType::COVARIANCE,
                            parameters::MatrixAdaptationType::CMSA})
    {
        parameters::Modules m;
        m.matrix_adaptation = type;
        parameters::Settings settings(4, m);
        settings.seed = 1; // not random_device (#141)
        ModularCMAES es(settings);
        FunctionType sphere = [](Eigen::Ref<const Vector> x) { return x.squaredNorm(); };
        // 200 generations: 60 reached 1e-6 in only ~95% of seeds; over seeds 1..300 200 give <= 2e-21
        for (int g = 0; g < 200; g++)
            es.step(sphere);
        CHECK(es.p->stats.centers.empty()); // no restart: every decomposition succeeded
        CHECK(es.p->stats.global_best.y < 1e-6);
    }
    constants::eigen_solver = eigen_solver;
    constants::cholesky_solver = cholesky_solver;
}

// #140: BIPOP's population sizes over the restarts are those of pycma 4.6.0's fmin with
// bipop=True and an integer popsize, transcribed below with pycma's names
// (cma/evolution_strategy.py:5355-5381, 5566-5570), with IPOP's limit of 2^9 lambda0 on its
// multiplier, which pycma does not have. The random stream is fixed: a copy of the optimizer's
// generator replays the numbers BIPOP::update draws, u for lambda_small at every restart, then u'
// for sigma at a small one (pycma draws its two in the other order). The runs last a scripted
// number of generations; they include fourteen large runs, the 9th to the 14th at the limit, and
// the small runs after them stay within it.
static void test_bipop_populations_match_pycma()
{
    parameters::Modules m;
    m.restart_strategy = parameters::RestartStrategyType::BIPOP;
    parameters::Settings settings(5, m);
    settings.verbose = false;
    settings.seed = 140;
    parameters::Parameters p(settings);
    const auto bipop = std::dynamic_pointer_cast<restart::BIPOP>(p.restart_strategy);
    CHECK(bipop != nullptr);
    rng::Generator replay = *p.generator;
    rng::uniform<Float> uniform; // BIPOP's draws (#164)
    std::mt19937 lengths(140);
    std::uniform_int_distribution<size_t> generations(1, 100);

    const size_t popsize0 = settings.lambda0; // 8
    const size_t max_lambda_factor = restart::default_max_lambda_factor;
    CHECK(bipop->max_lambda_factor == max_lambda_factor);
    CHECK(max_lambda_factor == 512);
    size_t at_the_limit = 0;
    size_t irun = 0, runs_with_small = 0, n_large = 0, n_small = 0, largest = 0;
    Float sum_small = 0, sum_large = 0;
    bool small = true; // the first run: poptype = 'small' (5355-5360)
    size_t popsize = popsize0;
    Float maxiter = 1e9; // maxiter0, no cap
    while (n_large < 14 && irun < 5000)
    {
        // The run ends after its generations and the evaluation of its center of mass; a capped
        // run stops when countiter >= maxiter (4527-4529), after ceil(maxiter) generations
        const Float g =
            std::min(static_cast<Float>(generations(lengths)), std::max(1.0, std::ceil(maxiter)));
        const size_t this_evals = popsize * static_cast<size_t>(g) + 1;
        (small ? sum_small : sum_large) += static_cast<Float>(this_evals); // 5566-5570
        irun++;

        p.stats.evaluations += this_evals;
        const Float sigma = bipop->update(p);

        const Float u = uniform(replay);
        Float sigma_factor = 1.0;
        small = sum_small < std::max<Float>(1.0, sum_large); // 5362, bipop = 1
        if (small)
        {
            runs_with_small++; // 5367: before the multiplier, which is then the last large run's
            sigma_factor = std::pow(0.01, uniform(replay)); // 5369
            const size_t popsize_multiplier =
                std::min(size_t{1} << (irun - runs_with_small), max_lambda_factor); // 5370
            // 5371: int(popsize0 * popsize_multiplier**(np.random.uniform()**2)), u * u for u**2
            popsize = static_cast<size_t>(static_cast<Float>(popsize0) *
                                          std::pow(static_cast<Float>(popsize_multiplier), u * u));
            maxiter = 0.5 * sum_large / static_cast<Float>(popsize); // 5372
            n_small++;
        }
        else
        {
            const size_t popsize_multiplier =
                std::min(size_t{1} << (irun - runs_with_small), max_lambda_factor); // 5380
            popsize = popsize0 * popsize_multiplier;                                // 5381
            maxiter = 1e9;
            n_large++;
            at_the_limit += popsize_multiplier == max_lambda_factor;
        }
        largest = std::max(largest, popsize);
        CHECK(popsize <= max_lambda_factor * popsize0);

        CHECK(bipop->large_run == !small);
        CHECK(p.lambda == popsize);
        CHECK(p.mu == popsize / 2); // pycma's default mu, popsize // 2 (lambda0 is even here)
        CHECK(std::abs(sigma - settings.sigma0 * sigma_factor) <= 1e-12 * sigma);
        // ExceededMaxIter takes the smaller of max_iter and its own limit (pycma: maxiter0), and
        // stops after max_iter generations: pycma's ceil(maxiter) (#139 D3)
        CHECK(bipop->max_iter == (small ? static_cast<size_t>(std::ceil(maxiter))
                                        : std::numeric_limits<size_t>::max()));
    }
    CHECK(n_large == 14 && n_small > 14);
    CHECK(at_the_limit == 6); // the 9th to the 14th large run
    CHECK(largest == max_lambda_factor * popsize0);
}

// #89: C++ code can assign Settings::C0 without set_C0; the optimizer checks it again, so a matrix
// of the wrong shape, an asymmetric one or one with NONE is std::invalid_argument, not a crash
static void test_c0_assigned_directly()
{
    const auto rejected = [](parameters::Settings settings) {
        try
        {
            parameters::Parameters p(settings);
        }
        catch (const std::invalid_argument &)
        {
            return true;
        }
        return false;
    };
    parameters::Settings settings(3);
    settings.C0 = Matrix::Identity(2, 2);
    CHECK(rejected(settings));
    Matrix asymmetric = Matrix::Identity(3, 3);
    asymmetric(0, 1) = 0.5;
    settings.C0 = asymmetric;
    CHECK(rejected(settings));
    settings.C0 = Matrix::Identity(3, 3);
    settings.modules.matrix_adaptation = parameters::MatrixAdaptationType::NONE;
    CHECK(rejected(settings));

    // A valid C0: the sampling factor of every adaptation reproduces it
    Matrix c0(3, 3);
    c0 << 4, 1, 0.5, 1, 2, 0.25, 0.5, 0.25, 1;
    for (const auto type : {parameters::MatrixAdaptationType::COVARIANCE,
                            parameters::MatrixAdaptationType::COVARIANCE_NO_EIGV,
                            parameters::MatrixAdaptationType::MATRIX,
                            parameters::MatrixAdaptationType::CHOLESKY,
                            parameters::MatrixAdaptationType::CMSA,
                            parameters::MatrixAdaptationType::NATURAL_GRADIENT})
    {
        parameters::Modules m;
        m.matrix_adaptation = type;
        parameters::Settings s(3, m);
        s.C0 = c0;
        parameters::Parameters p(s);
        Matrix f(3, 3);
        for (Eigen::Index i = 0; i < 3; ++i)
            f.col(i) = p.adaptation->compute_y(Vector::Unit(3, i));
        CHECK((f * f.transpose() - c0).cwiseAbs().maxCoeff() < 1e-12);
    }
}

// #136: a field assigned directly (C++ only; Python goes through the setters, which check each
// field) is checked when an optimizer is created from the settings
static void test_settings_checked_at_creation()
{
    const auto rejected = [](const parameters::Settings &settings) {
        try
        {
            parameters::Parameters p(settings);
        }
        catch (const std::invalid_argument &)
        {
            return true;
        }
        return false;
    };
    parameters::Settings valid(3);
    CHECK(!rejected(valid));
    {
        auto s = valid;
        s.mu0 = 0; // its own check
        CHECK(rejected(s));
    }
    {
        auto s = valid;
        s.sigma0 = std::numeric_limits<Float>::quiet_NaN();
        CHECK(rejected(s));
    }
    {
        auto s = valid;
        s.lb(1) = 6.0; // between fields: lb above ub
        CHECK(rejected(s));
    }
    {
        auto s = valid;
        s.lb(0) = std::numeric_limits<Float>::infinity(); // its own check (#133)
        CHECK(rejected(s));
    }
    {
        auto s = valid;
        s.lambda0 = 2; // mu0 = 3 above lambda0
        CHECK(rejected(s));
        s.set_lambda0(2); // the setter recomputes the default mu0
        CHECK(s.mu0 == 1 && !rejected(s));
    }
    {
        auto s = valid;
        s.set_lb(Vector::Constant(3, 6.0)); // the setters take any order
        CHECK(rejected(s));
        s.set_ub(Vector::Constant(3, 10.0));
        CHECK(!rejected(s) && s.center.isApproxToConstant(8.0));
    }
}

// The bounded integers and the shuffle (#116) use only the raw mt19937 output, which the C++
// standard fixes: the values are the same on every platform (CI runs this test on Linux and
// Windows too)
static void test_portable_random_integers()
{
    // The expected values were computed independently from numpy's MT19937 (legacy seeding, the
    // same state as std::mt19937(seed)) with the same rejection rule
    std::mt19937 gen(42);
    std::vector<uint32_t> drawn;
    for (int i = 0; i < 8; ++i)
        drawn.push_back(rng::below(10, gen));
    CHECK((drawn == std::vector<uint32_t>{2, 7, 6, 4, 6, 5, 0, 4}));

    std::mt19937 gen2(7);
    std::vector<int> order(10);
    std::iota(order.begin(), order.end(), 0);
    rng::shuffle(order, gen2);
    CHECK((order == std::vector<int>{9, 0, 8, 3, 2, 6, 7, 1, 4, 5}));

    // below(n) is uniform: counts of 60000 draws in 6 cells are within 5 standard deviations
    std::mt19937 gen3(1);
    std::vector<int> counts(6, 0);
    for (int i = 0; i < 60000; ++i)
        counts[rng::below(6, gen3)]++;
    for (const int c : counts)
        CHECK(std::abs(c - 10000) < 5 * 91);
    // A range of 1 needs no draw; a power of 2 never rejects
    std::mt19937 gen4(3), gen5(3);
    CHECK(rng::below(1, gen4) == 0);
    gen5.discard(1);
    CHECK(rng::below(8, gen4) == static_cast<uint32_t>(gen5() % 8));
    CHECK(rng::random_integer(5, 5, gen4) == 5);
}

// sampling::expected_norm against the closed forms: E||z|| of a standard normal vector,
// sqrt(2) Gamma((d + 1) / 2) / Gamma(d / 2), and of z_i^2 exponential, Gamma(d + 1/2) / Gamma(d)
static void test_expected_norm()
{
    for (const size_t d : {size_t{1}, size_t{2}, size_t{3}, size_t{10}, size_t{100}, size_t{1000}})
    {
        const Float dd = static_cast<Float>(d);
        const Float normal =
            std::sqrt(2.0) * std::exp(std::lgamma((dd + 1) / 2) - std::lgamma(dd / 2));
        const Float gauss =
            sampling::expected_norm(d,
                                    sampling::phi_psi_from_density([](const Float z) {
                                        return 2.0 * std::exp(-z * z / 2) / std::sqrt(2 * M_PI);
                                    }),
                                    1.0);
        CHECK(std::abs(gauss / normal - 1) < 1e-5);

        const Float gamma = std::exp(std::lgamma(dd + 0.5) - std::lgamma(dd));
        const Float exponential = sampling::expected_norm(
            d,
            sampling::phi_psi_from_density([](const Float z) { return 2 * z * std::exp(-z * z); }),
            1.0);
        CHECK(std::abs(exponential / gamma - 1) < 1e-8);

        // A uniform coordinate in (-a, a): the closed form of phi
        const Float uniform = sampling::expected_norm(
            d, [](const Float t) { return sampling::phi_psi_uniform(std::sqrt(3.0), t); }, 1.0);
        if (d == 1)
            CHECK(std::abs(uniform - std::sqrt(3.0) / 2) < 1e-10);
        CHECK(uniform < std::sqrt(dd));
    }
}

// #91: the state of the Mersenne Twister in the standard's representation, recovered from its
// outputs, gives a generator that continues with the same numbers, at every position in a block
static void test_generator_state()
{
    for (const size_t skip : {0, 1, 2, 396, 397, 623, 624, 625, 1000, 1247, 1248, 5000})
    {
        rng::Generator g(static_cast<unsigned>(skip + 7));
        g.discard(skip);
        const auto state = serialization::generator_state(g);
        rng::Generator copy;
        serialization::set_generator_state(copy, state);
        CHECK(serialization::generator_state(copy) == state);
        bool same = true;
        for (int k = 0; k < 3000; ++k)
            same = same && g() == copy();
        CHECK(same);
    }
}

namespace
{
//! One generation of es by ask-tell on f; the points of every generation are appended to trace
bool generation(ModularCMAES &es, FunctionType &f, std::vector<double> &trace)
{
    for (;;)
    {
        const Matrix X = es.ask();
        Vector values(X.cols());
        for (Eigen::Index j = 0; j < X.cols(); ++j)
            values(j) = f(X.col(j));
        trace.insert(trace.end(), X.data(), X.data() + X.size());
        if (es.tell(values))
            return !es.break_conditions();
    }
}
} // namespace

// #91: a run that is saved and loaded continues bit for bit: after whole generations, and between
// ask() and tell() (a pending batch)
static void test_save_and_load_continue()
{
    using namespace parameters;
    std::vector<Modules> configs(6);
    configs[1].ssa = StepSizeAdaptation::TPA;
    configs[1].sequential_selection = true;
    configs[1].restart_strategy = RestartStrategyType::BIPOP;
    configs[2].sampler = BaseSampler::SOBOL;
    configs[2].matrix_adaptation = MatrixAdaptationType::MATRIX;
    configs[2].elitist = true;
    configs[3].sampler = BaseSampler::HALTON;
    configs[3].orthogonal = true;
    configs[3].mirrored = Mirror::PAIRWISE;
    configs[3].repelling_type = RepellingType::ADAPTIVE;
    configs[3].restart_strategy = RestartStrategyType::IPOP;
    configs[4].bound_correction = CorrectionMethod::COTN;
    configs[4].ssa = StepSizeAdaptation::TPA_INPLACE;
    configs[4].matrix_adaptation = MatrixAdaptationType::CHOLESKY;
    configs[5].ssa = StepSizeAdaptation::XNES;
    configs[5].matrix_adaptation = MatrixAdaptationType::NATURAL_GRADIENT;
    configs[5].sample_sigma = true;
    FunctionType f = functions::rastrigin;
    for (const auto &modules : configs)
    {
        Settings settings(4, modules);
        settings.budget = 3000;
        settings.seed = 3;
        ModularCMAES reference(settings), es(settings);
        std::vector<double> a, b;
        for (int t = 0; t < 15; ++t)
        {
            generation(reference, f, a);
            generation(es, f, b);
        }
        // Mid-generation: the points of ask() wait for their values in the saved state
        reference.ask();
        es.ask();
        ModularCMAES resumed = serialization::load<ModularCMAES>(serialization::save(es));
        CHECK(resumed.has_pending());
        CHECK(serialization::save(resumed) == serialization::save(es));
        bool running = true;
        while (running)
        {
            const bool r1 = generation(reference, f, a);
            const bool r2 = generation(resumed, f, b);
            CHECK(r1 == r2);
            running = r1 && r2;
            // A save between generations as well
            resumed = serialization::load<ModularCMAES>(serialization::save(resumed));
        }
        CHECK(a == b);
        CHECK(reference.p->stats.evaluations == resumed.p->stats.evaluations);
        CHECK(serialization::save(reference) == serialization::save(resumed));
    }
}

// #91: a state that cannot be loaded raises serialization::Error, with the reason
static void test_load_errors()
{
    ModularCMAES es(parameters::Settings(3));
    const std::string state = serialization::save(es);
    const auto message = [](const std::string &bytes) -> std::string {
        try
        {
            serialization::load<ModularCMAES>(bytes);
        }
        catch (const serialization::Error &e)
        {
            return e.what();
        }
        return "";
    };
    CHECK(message(state).empty());
    CHECK(message("hello").find("not a modcma state") != std::string::npos);
    CHECK(message(state.substr(0, state.size() / 2)).find("checksum") != std::string::npos);
    std::string flipped = state;
    flipped[flipped.size() / 2] ^= 1;
    CHECK(message(flipped).find("checksum") != std::string::npos);
    std::string newer = state;
    newer[8] = static_cast<char>(serialization::format_version + 1);
    CHECK(message(newer).find("newer modcma") != std::string::npos);
    bool kind = false;
    try
    {
        serialization::load<parameters::Settings>(state);
    }
    catch (const serialization::Error &e)
    {
        kind =
            std::string(e.what()).find("holds a ModularCMAES, not a Settings") != std::string::npos;
    }
    CHECK(kind);
}

int main()
{
    test_generator_state();
    test_save_and_load_continue();
    test_load_errors();
    test_portable_random_integers();
    test_expected_norm();
    test_uniform_is_open_interval();
    test_uniform_values_unchanged();
    test_evaluation_queue();
    test_reserved_points();
    test_budget_with_restarts();
    test_split_median();
    test_lapack_eigen();
    test_lapack_fallback();
    test_natural_gradient_low_rank_update();
    test_functions();
    test_c0_assigned_directly();
    test_settings_checked_at_creation();
    test_bipop_populations_match_pycma();
    test_apply_transpose();
    test_gradient_injection();
    if (failures)
        std::fprintf(stderr, "%d check(s) failed\n", failures);
    else
        std::printf("all checks passed\n");
    return failures ? 1 : 0;
}
