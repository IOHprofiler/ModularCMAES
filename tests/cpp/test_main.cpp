// C++ unit tests for behaviour that cannot be reached from Python (run with ctest).
// Deliberately without a test framework: each test is a function, CHECK reports and counts failures.
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "common.hpp"
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

int main()
{
    test_uniform_is_open_interval();
    test_uniform_values_unchanged();
    if (failures)
        std::fprintf(stderr, "%d check(s) failed\n", failures);
    else
        std::printf("all checks passed\n");
    return failures ? 1 : 0;
}
