// The state of the library's objects as bytes (#91); the format is described in
// serialization.hpp.
//
// Every class with state has a put() (write) and a get() or make() (read) here, which list its
// members in the order they are written. A member that is not written is named in a "Not saved"
// comment: a buffer that is overwritten before it is read (scratch), or a value that is computed
// again. A new member of a class needs a line in both functions (or in the comment), and, when it
// changes what is written, a new format_version.
//
// Polymorphic members (the modules) are written as a tag, the name of their type, followed by
// their members; the reader creates the object of that type. Types the library does not know (a
// sampler or a restart criterion written in Python) cannot be written, except the restart criteria
// through External (the Python bindings pickle them).
#include "serialization.hpp"

#include <cstdlib>
#include <cstring>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#if defined(__GNUG__)
#include <cxxabi.h>
#endif

namespace serialization
{
namespace
{
constexpr char magic[8] = {'M', 'O', 'D', 'C', 'M', 'A', 'S', 'T'};
//! The magic number and the format version
constexpr size_t header_size = sizeof(magic) + 4;
//! The CRC-32 of everything before it
constexpr size_t trailer_size = 4;

//! CRC-32 (IEEE 802.3, the CRC of zlib's crc32), reflected, with the slicing-by-8 tables
uint32_t crc32(const char *data, const size_t n)
{
    static const auto tables = [] {
        std::array<std::array<uint32_t, 256>, 8> t{};
        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[0][i] = c;
        }
        for (size_t k = 1; k < 8; ++k)
            for (uint32_t i = 0; i < 256; ++i)
                t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xFFu];
        return t;
    }();
    const auto &t = tables;
    const auto byte = [data](const size_t i) {
        return static_cast<uint32_t>(static_cast<uint8_t>(data[i]));
    };
    uint32_t c = 0xFFFFFFFFu;
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        const uint32_t lo =
            c ^ (byte(i) | byte(i + 1) << 8 | byte(i + 2) << 16 | byte(i + 3) << 24);
        const uint32_t hi = byte(i + 4) | byte(i + 5) << 8 | byte(i + 6) << 16 | byte(i + 7) << 24;
        c = t[7][lo & 0xFFu] ^ t[6][(lo >> 8) & 0xFFu] ^ t[5][(lo >> 16) & 0xFFu] ^ t[4][lo >> 24] ^
            t[3][hi & 0xFFu] ^ t[2][(hi >> 8) & 0xFFu] ^ t[1][(hi >> 16) & 0xFFu] ^ t[0][hi >> 24];
    }
    for (; i < n; ++i)
        c = t[0][(c ^ byte(i)) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

bool little_endian()
{
    const uint16_t one = 1;
    uint8_t first = 0;
    std::memcpy(&first, &one, 1);
    return first == 1;
}

//! The name of a C++ type, for messages
std::string type_name(const std::type_info &type)
{
#if defined(__GNUG__)
    int status = 0;
    char *name = abi::__cxa_demangle(type.name(), nullptr, nullptr, &status);
    if (status == 0 && name != nullptr)
    {
        std::string result(name);
        std::free(name);
        return result;
    }
#endif
    return type.name();
}

// ------------------------------------------------------------------------------------------------
// The Mersenne Twister (std::mt19937) in the representation of the C++ standard
// ([rand.eng.mers]: the words X_{i-n}, ..., X_{i-1}). The standard libraries write other
// representations with operator<< (libstdc++ adds its position in the block), so the words are
// recovered from the generator's output, which is the same everywhere, and set with a seed
// sequence, whose words the standard defines as the state.
// ------------------------------------------------------------------------------------------------
constexpr size_t mt_n = 624, mt_m = 397;
constexpr uint32_t mt_a = 0x9908B0DFu, mt_upper = 0x80000000u, mt_lower = 0x7FFFFFFFu;

uint32_t temper(uint32_t y)
{
    y ^= y >> 11;
    y ^= (y << 7) & 0x9D2C5680u;
    y ^= (y << 15) & 0xEFC60000u;
    y ^= y >> 18;
    return y;
}

//! The inverse of temper()
uint32_t untemper(uint32_t y)
{
    y ^= y >> 18;
    // The mask has no bits below 17, so one step recovers the bits shifted in
    y ^= (y << 15) & 0xEFC60000u;
    // Each step recovers 7 more of the low bits
    uint32_t r = y;
    for (int i = 0; i < 4; ++i)
        r = y ^ ((r << 7) & 0x9D2C5680u);
    y = r;
    // Each step recovers 11 more of the high bits
    r = y;
    for (int i = 0; i < 2; ++i)
        r = y ^ (r >> 11);
    return r;
}

//! The next n outputs of a generator in the standard representation `state`
std::array<uint32_t, mt_n> mt_outputs(const std::array<uint32_t, mt_n> &state)
{
    std::array<uint32_t, 2 * mt_n> x{};
    std::copy(state.begin(), state.end(), x.begin());
    std::array<uint32_t, mt_n> out{};
    for (size_t j = 0; j < mt_n; ++j)
    {
        const uint32_t y = (x[j] & mt_upper) | (x[j + 1] & mt_lower);
        x[j + mt_n] = x[j + mt_m] ^ (y >> 1) ^ ((y & 1u) ? mt_a : 0u);
        out[j] = temper(x[j + mt_n]);
    }
    return out;
}

/**
 * A seed sequence that gives the words of a state: the constructor and seed() of the Mersenne
 * Twister with a seed sequence set X_{-n}, ..., X_{-1} to the n words that generate() writes
 * ([rand.eng.mers]: k = 1 for w = 32), which is the standard representation of the state, in
 * every standard library
 */
struct StateSequence
{
    using result_type = uint32_t;
    const std::array<uint32_t, mt_n> &state;

    template <typename It> void generate(It begin, It end) const
    {
        if (static_cast<size_t>(end - begin) != mt_n)
            throw std::logic_error("serialization: the generator asked for " +
                                   std::to_string(end - begin) + " words");
        std::copy(state.begin(), state.end(), begin);
    }
};
} // namespace

std::array<uint32_t, 624> generator_state(const rng::Generator &generator)
{
    // The next n words X_i..X_{i+n-1} (untempered outputs) follow from the state by
    // X_{k+n} = X_{k+m} ^ twist(upper bit of X_k, lower 31 bits of X_{k+1}), and twist() can be
    // inverted: from k = n - 1 down to 0, X_{k+m} is known (a later word, or recovered at the steps
    // k + m and k + m - 1), which gives the upper bit of X_k and the lower bits of X_{k+1}. The
    // lower bits of X_{i-n} are not used by the generator; they are 0.
    rng::Generator copy = generator;
    std::array<uint32_t, mt_n> outputs{};
    std::array<uint32_t, 2 * mt_n> x{};
    for (size_t j = 0; j < mt_n; ++j)
    {
        outputs[j] = static_cast<uint32_t>(copy());
        x[mt_n + j] = untemper(outputs[j]);
    }
    for (size_t k = mt_n; k-- > 0;)
    {
        uint32_t t = x[k + mt_n] ^ x[k + mt_m];
        const uint32_t odd = t >> 31;
        if (odd != 0)
            t ^= mt_a;
        const uint32_t y = (t << 1) | odd;
        x[k] = (x[k] & mt_lower) | (y & mt_upper);
        if (k + 1 < mt_n)
            x[k + 1] = (x[k + 1] & mt_upper) | (y & mt_lower);
    }
    std::array<uint32_t, mt_n> state{};
    std::copy(x.begin(), x.begin() + mt_n, state.begin());
    if (mt_outputs(state) != outputs)
        throw std::logic_error(
            "serialization: the state of the random generator was not recovered");
    return state;
}

void set_generator_state(rng::Generator &generator, const std::array<uint32_t, 624> &state)
{
    StateSequence sequence{state};
    generator.seed(sequence);
    // Check that the generator continues as the state says
    rng::Generator copy = generator;
    std::array<uint32_t, mt_n> outputs{};
    for (auto &output : outputs)
        output = static_cast<uint32_t>(copy());
    if (outputs != mt_outputs(state))
        throw Error("modcma state: the random generator could not be restored with this C++ "
                    "standard library");
}

External &external()
{
    static External functions;
    return functions;
}

// ------------------------------------------------------------------------------------------------
// Writer and Reader: the encoding of numbers, vectors and matrices
// ------------------------------------------------------------------------------------------------
namespace detail
{
class Writer
{
  public:
    std::string out;
    //! The generators written so far, by address: each one's state is written once
    std::unordered_map<const rng::Generator *, uint64_t> generators;

    void u8(const uint8_t v) { out.push_back(static_cast<char>(v)); }

    void u32(const uint32_t v)
    {
        char bytes[4];
        for (int i = 0; i < 4; ++i)
            bytes[i] = static_cast<char>(static_cast<uint8_t>(v >> (8 * i)));
        out.append(bytes, sizeof bytes);
    }

    void u64(const uint64_t v)
    {
        char bytes[8];
        for (int i = 0; i < 8; ++i)
            bytes[i] = static_cast<char>(static_cast<uint8_t>(v >> (8 * i)));
        out.append(bytes, sizeof bytes);
    }

    void i64(const int64_t v) { u64(static_cast<uint64_t>(v)); }

    void f64(const double v)
    {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof bits);
        u64(bits);
    }

    void flag(const bool v) { u8(v ? 1 : 0); }

    //! As u64; the largest size_t (the sentinels "no limit", "every individual") as the largest
    //! u64, so that a state moves between 32- and 64-bit platforms
    void size(const size_t v)
    {
        u64(v == std::numeric_limits<size_t>::max() ? std::numeric_limits<uint64_t>::max()
                                                    : static_cast<uint64_t>(v));
    }

    void index(const Eigen::Index v) { i64(static_cast<int64_t>(v)); }

    void str(const std::string &s)
    {
        size(s.size());
        out.append(s);
    }

    void doubles(const double *data, const size_t n)
    {
        if (little_endian())
            out.append(reinterpret_cast<const char *>(data), n * sizeof(double));
        else
            for (size_t i = 0; i < n; ++i)
                f64(data[i]);
    }

    void vec(const Vector &v)
    {
        size(static_cast<size_t>(v.size()));
        doubles(v.data(), static_cast<size_t>(v.size()));
    }

    void mat(const Matrix &m)
    {
        size(static_cast<size_t>(m.rows()));
        size(static_cast<size_t>(m.cols()));
        doubles(m.data(), static_cast<size_t>(m.size()));
    }

    template <typename T, typename F> void opt(const std::optional<T> &value, F &&put)
    {
        flag(value.has_value());
        if (value)
            put(*value);
    }

    template <typename T, typename F> void list(const std::vector<T> &values, F &&put)
    {
        size(values.size());
        for (const auto &value : values)
            put(value);
    }

    void floats(const std::vector<Float> &values)
    {
        size(values.size());
        doubles(values.data(), values.size());
    }

    /**
     * A shared generator: 0 none, 1 the global generator (rng::set_seed; its state is not
     * written), 2 and the number of a generator written before, 3 and the state of a new one
     */
    void generator(const std::shared_ptr<rng::Generator> &g)
    {
        if (!g)
        {
            u8(0);
            return;
        }
        if (g.get() == &rng::GENERATOR)
        {
            u8(1);
            return;
        }
        if (const auto it = generators.find(g.get()); it != generators.end())
        {
            u8(2);
            u64(it->second);
            return;
        }
        generators.emplace(g.get(), generators.size());
        u8(3);
        for (const uint32_t word : generator_state(*g))
            u32(word);
    }
};

class Reader
{
  public:
    //! The object's bytes (between the header and the checksum)
    std::string_view in;
    size_t pos = 0;
    //! The format version of the state
    uint32_t version = format_version;
    //! The generators read so far, in order
    std::vector<std::shared_ptr<rng::Generator>> generators;

    [[noreturn]] static void fail(const std::string &what) { throw Error("modcma state: " + what); }

    //! A state that holds the impossible: a value out of range, inconsistent sizes
    static void check(const bool condition, const std::string &what)
    {
        if (!condition)
            fail("corrupted (" + what + ")");
    }

    void need(const size_t n) const
    {
        if (n > in.size() - pos)
            fail("truncated: the data ends in the middle of the object");
    }

    uint8_t u8()
    {
        need(1);
        return static_cast<uint8_t>(in[pos++]);
    }

    uint32_t u32()
    {
        need(4);
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i)
            v |= static_cast<uint32_t>(static_cast<uint8_t>(in[pos++])) << (8 * i);
        return v;
    }

    uint64_t u64()
    {
        need(8);
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i)
            v |= static_cast<uint64_t>(static_cast<uint8_t>(in[pos++])) << (8 * i);
        return v;
    }

    int64_t i64() { return static_cast<int64_t>(u64()); }

    double f64()
    {
        const uint64_t bits = u64();
        double v = 0;
        std::memcpy(&v, &bits, sizeof v);
        return v;
    }

    bool flag()
    {
        const uint8_t v = u8();
        check(v <= 1, "a boolean is " + std::to_string(v));
        return v == 1;
    }

    size_t size()
    {
        const uint64_t v = u64();
        if (v == std::numeric_limits<uint64_t>::max())
            return std::numeric_limits<size_t>::max();
        check(v < std::numeric_limits<size_t>::max(), "a size does not fit");
        return static_cast<size_t>(v);
    }

    Eigen::Index index() { return static_cast<Eigen::Index>(i64()); }

    int to_int(const int64_t v)
    {
        check(v >= std::numeric_limits<int>::min() && v <= std::numeric_limits<int>::max(),
              "an integer does not fit");
        return static_cast<int>(v);
    }

    //! A count of elements of at least element_bytes bytes each, which the data can hold
    size_t count(const size_t element_bytes)
    {
        const uint64_t n = u64();
        check(n <= (in.size() - pos) / std::max<size_t>(1, element_bytes),
              "a length of " + std::to_string(n) + " exceeds the data");
        return static_cast<size_t>(n);
    }

    std::string str()
    {
        const size_t n = count(1);
        std::string s(in.substr(pos, n));
        pos += n;
        return s;
    }

    void doubles(double *data, const size_t n)
    {
        need(n * sizeof(double));
        if (little_endian())
        {
            std::memcpy(data, in.data() + pos, n * sizeof(double));
            pos += n * sizeof(double);
        }
        else
            for (size_t i = 0; i < n; ++i)
                data[i] = f64();
    }

    Vector vec()
    {
        const size_t n = count(sizeof(double));
        Vector v(static_cast<Eigen::Index>(n));
        doubles(v.data(), n);
        return v;
    }

    //! A vector of the given length
    Vector vec(const size_t n, const char *name)
    {
        Vector v = vec();
        check(static_cast<size_t>(v.size()) == n,
              std::string(name) + " has length " + std::to_string(v.size()) + ", not " +
                  std::to_string(n));
        return v;
    }

    Matrix mat()
    {
        const uint64_t rows = u64(), cols = u64();
        const size_t available = (in.size() - pos) / sizeof(double);
        check(rows <= available && (rows == 0 || cols <= available / rows),
              "a matrix of " + std::to_string(rows) + " x " + std::to_string(cols) +
                  " exceeds the data");
        Matrix m(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
        doubles(m.data(), static_cast<size_t>(m.size()));
        return m;
    }

    //! A matrix of the given shape
    Matrix mat(const size_t rows, const size_t cols, const char *name)
    {
        Matrix m = mat();
        check(static_cast<size_t>(m.rows()) == rows && static_cast<size_t>(m.cols()) == cols,
              std::string(name) + " is " + std::to_string(m.rows()) + " x " +
                  std::to_string(m.cols()) + ", not " + std::to_string(rows) + " x " +
                  std::to_string(cols));
        return m;
    }

    template <typename F> auto opt(F &&get) -> std::optional<decltype(get())>
    {
        if (!flag())
            return std::nullopt;
        return get();
    }

    template <typename F> auto list(F &&get) -> std::vector<decltype(get())>
    {
        const size_t n = count(1);
        std::vector<decltype(get())> values;
        values.reserve(n);
        for (size_t i = 0; i < n; ++i)
            values.push_back(get());
        return values;
    }

    std::vector<Float> floats()
    {
        const size_t n = count(sizeof(double));
        std::vector<Float> values(n);
        doubles(values.data(), n);
        return values;
    }

    //! An enumerator in [0, last]
    template <typename E> E enumeration(const E last, const char *name)
    {
        const uint8_t v = u8();
        check(v <= static_cast<uint8_t>(last),
              std::string(name) + " has the unknown value " + std::to_string(v));
        return static_cast<E>(v);
    }

    std::shared_ptr<rng::Generator> generator()
    {
        switch (u8())
        {
        case 0:
            return nullptr;
        case 1:
            return rng::global();
        case 2: {
            const uint64_t id = u64();
            check(id < generators.size(), "a generator that was not written before");
            return generators[static_cast<size_t>(id)];
        }
        case 3: {
            std::array<uint32_t, 624> state{};
            for (auto &word : state)
                word = u32();
            auto g = std::make_shared<rng::Generator>();
            set_generator_state(*g, state);
            generators.push_back(g);
            return g;
        }
        default:
            fail("corrupted (an unknown kind of generator)");
        }
    }
};

//! The types of a polymorphic base, by tag and by C++ type
template <typename Base> struct Registry
{
    struct Entry
    {
        std::type_index type;
        std::string tag;
        void (*write)(Writer &, const Base &);
        std::shared_ptr<Base> (*read)(Reader &);
    };
    std::vector<Entry> entries;

    const Entry *by_type(const std::type_info &type) const
    {
        for (const auto &entry : entries)
            if (entry.type == std::type_index(type))
                return &entry;
        return nullptr;
    }

    const Entry *by_tag(const std::string &tag) const
    {
        for (const auto &entry : entries)
            if (entry.tag == tag)
                return &entry;
        return nullptr;
    }
};

template <typename T> struct Tag
{
};

//! The name of the kind of object a state holds (its header)
template <typename T> struct Kind;
#define MODCMA_KIND(T, name)                                                                       \
    template <> struct Kind<T>                                                                     \
    {                                                                                              \
        static constexpr const char *value = name;                                                 \
    };                                                                                             \
    template <> struct Kind<std::shared_ptr<T>>                                                    \
    {                                                                                              \
        static constexpr const char *value = name;                                                 \
    };
MODCMA_KIND(ModularCMAES, "ModularCMAES")
MODCMA_KIND(parameters::Parameters, "Parameters")
MODCMA_KIND(parameters::Settings, "Settings")
MODCMA_KIND(parameters::Modules, "Modules")
MODCMA_KIND(Population, "Population")
MODCMA_KIND(parameters::Stats, "Stats")
MODCMA_KIND(Solution, "Solution")
MODCMA_KIND(parameters::Center, "Center")
MODCMA_KIND(parameters::Weights, "Weights")
MODCMA_KIND(MetricCopy, "MetricCopy")
MODCMA_KIND(restart::Criteria, "Criteria")
MODCMA_KIND(selection::Strategy, "selection.Strategy")
MODCMA_KIND(repelling::TabooPoint, "TabooPoint")
MODCMA_KIND(mapping::CoordinateMapping, "CoordinateMapping")
MODCMA_KIND(sampling::Sampler, "sampling.Sampler")
MODCMA_KIND(matrix_adaptation::Adaptation, "matrix_adaptation.Adaptation")
MODCMA_KIND(mutation::Strategy, "mutation.Strategy")
MODCMA_KIND(mutation::ThresholdConvergence, "mutation.ThresholdConvergence")
MODCMA_KIND(mutation::SequentialSelection, "mutation.SequentialSelection")
MODCMA_KIND(mutation::SigmaSampler, "mutation.SigmaSampler")
MODCMA_KIND(selection::Pairwise, "selection.Pairwise")
MODCMA_KIND(selection::Elitsm, "selection.Elitism")
MODCMA_KIND(restart::Criterion, "restart.Criterion")
MODCMA_KIND(restart::Strategy, "restart.Strategy")
MODCMA_KIND(bounds::BoundCorrection, "bounds.BoundCorrection")
MODCMA_KIND(repelling::Repelling, "repelling.Repelling")
MODCMA_KIND(center::Placement, "center.Placement")
#undef MODCMA_KIND
} // namespace detail

using detail::Reader;
using detail::Registry;
using detail::Tag;
using detail::Writer;

struct Access
{
    // --------------------------------------------------------------------------------------------
    // Polymorphic members: a tag, then the members of the type ("" for a null pointer)
    // --------------------------------------------------------------------------------------------
    // The types of each polymorphic base (defined after the class)
    static const Registry<sampling::Sampler> &registry(Tag<sampling::Sampler>);
    static const Registry<matrix_adaptation::Adaptation> &
        registry(Tag<matrix_adaptation::Adaptation>);
    static const Registry<mutation::Strategy> &registry(Tag<mutation::Strategy>);
    static const Registry<mutation::ThresholdConvergence> &
        registry(Tag<mutation::ThresholdConvergence>);
    static const Registry<mutation::SequentialSelection> &
        registry(Tag<mutation::SequentialSelection>);
    static const Registry<mutation::SigmaSampler> &registry(Tag<mutation::SigmaSampler>);
    static const Registry<selection::Pairwise> &registry(Tag<selection::Pairwise>);
    static const Registry<selection::Elitsm> &registry(Tag<selection::Elitsm>);
    static const Registry<restart::Criterion> &registry(Tag<restart::Criterion>);
    static const Registry<restart::Strategy> &registry(Tag<restart::Strategy>);
    static const Registry<bounds::BoundCorrection> &registry(Tag<bounds::BoundCorrection>);
    static const Registry<repelling::Repelling> &registry(Tag<repelling::Repelling>);
    static const Registry<center::Placement> &registry(Tag<center::Placement>);

    template <typename T, typename Base>
    static typename Registry<Base>::Entry entry(const char *tag)
    {
        return {std::type_index(typeid(T)),
                tag,
                [](Writer &w, const Base &object) { put(w, static_cast<const T &>(object)); },
                [](Reader &r) -> std::shared_ptr<Base> { return make(r, Tag<T>{}); }};
    }

    template <typename Base> static void write_poly(Writer &w, const Base *object, const char *what)
    {
        if (object == nullptr)
        {
            w.str("");
            return;
        }
        const auto *entry = registry(Tag<Base>{}).by_type(typeid(*object));
        if (entry == nullptr)
        {
            if constexpr (std::is_same_v<Base, restart::Criterion>)
            {
                if (external().save_criterion)
                {
                    w.str("external");
                    w.str(external().save_criterion(*object));
                    return;
                }
            }
            throw std::invalid_argument("cannot save a " + std::string(what) + " of type " +
                                        type_name(typeid(*object)) +
                                        ": it is not a type of the library (one defined in "
                                        "Python?), and the library cannot save its state");
        }
        w.str(entry->tag);
        entry->write(w, *object);
    }

    template <typename Base>
    static void write_poly(Writer &w, const std::shared_ptr<Base> &object, const char *what)
    {
        write_poly<Base>(w, object.get(), what);
    }

    template <typename Base> static std::shared_ptr<Base> read_poly(Reader &r, const char *what)
    {
        const std::string tag = r.str();
        if (tag.empty())
            return nullptr;
        if constexpr (std::is_same_v<Base, restart::Criterion>)
        {
            if (tag == "external")
            {
                const std::string bytes = r.str();
                if (!external().load_criterion)
                    Reader::fail("the state holds a restart criterion defined in Python, which "
                                 "only the Python package can load");
                auto criterion = external().load_criterion(bytes);
                Reader::check(criterion != nullptr, "an external restart criterion");
                return criterion;
            }
        }
        const auto *entry = registry(Tag<Base>{}).by_tag(tag);
        if (entry == nullptr)
            Reader::fail("unknown " + std::string(what) + " type '" + tag +
                         "' (written by a newer modcma?)");
        return entry->read(r);
    }

    // --------------------------------------------------------------------------------------------
    // Small objects
    // --------------------------------------------------------------------------------------------
    static void put(Writer &w, const rng::normal<Float> &n)
    {
        // rng (uniform) has no state
        w.f64(n.mu);
        w.f64(n.sigma);
        w.f64(n.r2);
        w.flag(n.generate);
    }

    static void get(Reader &r, rng::normal<Float> &n)
    {
        n.mu = r.f64();
        n.sigma = r.f64();
        n.r2 = r.f64();
        n.generate = r.flag();
    }

    static void put(Writer &w, const rng::Xoshiro256pp &g)
    {
        for (const uint64_t word : g.s)
            w.u64(word);
    }

    static void get(Reader &r, rng::Xoshiro256pp &g)
    {
        for (auto &word : g.s)
            word = r.u64();
    }

    static void put(Writer &w, const rng::Shuffler &s)
    {
        for (const size_t v :
             {s.start, s.stop, s.n, s.seed, s.offset, s.multiplier, s.modulus, s.found})
            w.size(v);
    }

    static void get(Reader &r, rng::Shuffler &s)
    {
        for (size_t *v :
             {&s.start, &s.stop, &s.n, &s.seed, &s.offset, &s.multiplier, &s.modulus, &s.found})
            *v = r.size();
        Reader::check(s.stop >= s.start && s.n == s.stop - s.start && s.modulus >= s.n &&
                          (s.n == 0 || s.seed < s.n),
                      "the shuffler of a cached sequence");
    }

    static void put(Writer &w, const Solution &s)
    {
        w.vec(s.x);
        w.f64(s.y);
        w.size(s.t);
        w.size(s.e);
    }

    static void get(Reader &r, Solution &s)
    {
        s.x = r.vec();
        s.y = r.f64();
        s.t = r.size();
        s.e = r.size();
    }

    static Solution get_solution(Reader &r)
    {
        Solution s;
        get(r, s);
        return s;
    }

    static void put(Writer &w, const MetricCopy &m)
    {
        w.vec(m.d);
        w.opt(m.B, [&](const Matrix &B) { w.mat(B); });
    }

    static void get(Reader &r, MetricCopy &m)
    {
        m.d = r.vec();
        m.B = r.opt([&] { return r.mat(); });
        Reader::check(!m.B || (m.B->rows() == m.d.size() && m.B->cols() == m.d.size()),
                      "the eigenvectors of a metric");
    }

    static void put(Writer &w, const parameters::Center &c)
    {
        put(w, static_cast<const Solution &>(c));
        w.f64(c.sigma);
        put(w, c.metric);
        w.list(c.reasons, [&](const std::string &s) { w.str(s); });
    }

    static void get(Reader &r, parameters::Center &c)
    {
        get(r, static_cast<Solution &>(c));
        c.sigma = r.f64();
        get(r, c.metric);
        c.reasons = r.list([&] { return r.str(); });
    }

    // --------------------------------------------------------------------------------------------
    // Settings
    // --------------------------------------------------------------------------------------------
    static void put(Writer &w, const parameters::Modules &m)
    {
        for (const bool b : {m.elitist,
                             m.active,
                             m.orthogonal,
                             m.sequential_selection,
                             m.threshold_convergence,
                             m.sample_sigma})
            w.flag(b);
        for (const int e : {static_cast<int>(m.weights),
                            static_cast<int>(m.sampler),
                            static_cast<int>(m.mirrored),
                            static_cast<int>(m.ssa),
                            static_cast<int>(m.bound_correction),
                            static_cast<int>(m.restart_strategy),
                            static_cast<int>(m.matrix_adaptation),
                            static_cast<int>(m.center_placement),
                            static_cast<int>(m.sample_transformation),
                            static_cast<int>(m.repelling_type),
                            static_cast<int>(m.repelling_metric)})
            w.u8(static_cast<uint8_t>(e));
    }

    static void get(Reader &r, parameters::Modules &m)
    {
        using namespace parameters;
        for (bool *b : {&m.elitist,
                        &m.active,
                        &m.orthogonal,
                        &m.sequential_selection,
                        &m.threshold_convergence,
                        &m.sample_sigma})
            *b = r.flag();
        m.weights = r.enumeration(RecombinationWeights::UTILITY, "RecombinationWeights");
        m.sampler = r.enumeration(BaseSampler::TESTER, "BaseSampler");
        m.mirrored = r.enumeration(Mirror::PAIRWISE, "Mirror");
        m.ssa = r.enumeration(StepSizeAdaptation::TPA_INPLACE, "StepSizeAdaptation");
        m.bound_correction = r.enumeration(CorrectionMethod::RESAMPLE, "CorrectionMethod");
        m.restart_strategy = r.enumeration(RestartStrategyType::BIPOP, "RestartStrategyType");
        m.matrix_adaptation =
            r.enumeration(MatrixAdaptationType::NATURAL_GRADIENT, "MatrixAdaptationType");
        m.center_placement = r.enumeration(CenterPlacement::MAXIMIN_TABOO, "CenterPlacement");
        m.sample_transformation =
            r.enumeration(SampleTranformerType::DOUBLE_WEIBULL, "SampleTranformerType");
        m.repelling_type = r.enumeration(RepellingType::ADAPTIVE, "RepellingType");
        m.repelling_metric = r.enumeration(RepellingMetric::ARCHIVED, "RepellingMetric");
    }

    static void put_float(Writer &w, const std::optional<Float> &v)
    {
        w.opt(v, [&](const Float x) { w.f64(x); });
    }

    static std::optional<Float> get_float(Reader &r)
    {
        return r.opt([&] { return r.f64(); });
    }

    static void put(Writer &w, const parameters::Settings &s)
    {
        w.size(s.dim);
        put(w, s.modules);
        put_float(w, s.target);
        w.opt(s.max_generations, [&](const size_t v) { w.size(v); });
        w.size(s.budget);
        w.f64(s.sigma0);
        w.size(s.lambda0);
        w.size(s.mu0);
        w.opt(s.x0, [&](const Vector &v) { w.vec(v); });
        w.vec(s.lb);
        w.vec(s.ub);
        w.size(static_cast<size_t>(s.integer_variables.size()));
        for (const int i : s.integer_variables)
            w.i64(i);
        w.vec(s.db);
        w.vec(s.center);
        w.f64(s.diameter);
        w.f64(s.volume);
        w.flag(s.has_bounds);
        for (const auto *rate : {&s.cs, &s.cc, &s.cmu, &s.c1, &s.damps, &s.acov})
            put_float(w, *rate);
        w.flag(s.verbose);
        w.flag(s.one_plus_one);
        w.opt(s.seed, [&](const int v) { w.i64(v); });
        w.flag(s.fast_sampling);
        w.opt(s.C0, [&](const Matrix &m) { w.mat(m); });
        w.flag(s.mu0_given);
    }

    static parameters::Settings get_settings(Reader &r)
    {
        // Every member is overwritten
        parameters::Settings s(1);
        s.dim = r.size();
        Reader::check(s.dim >= 1, "dim is 0");
        get(r, s.modules);
        s.target = get_float(r);
        s.max_generations = r.opt([&] { return r.size(); });
        s.budget = r.size();
        s.sigma0 = r.f64();
        s.lambda0 = r.size();
        s.mu0 = r.size();
        s.x0 = r.opt([&] { return r.vec(s.dim, "x0"); });
        s.lb = r.vec(s.dim, "lb");
        s.ub = r.vec(s.dim, "ub");
        const size_t n_integer = r.count(8);
        s.integer_variables.resize(static_cast<Eigen::Index>(n_integer));
        for (auto &i : s.integer_variables)
            i = r.to_int(r.i64());
        s.db = r.vec(s.dim, "db");
        s.center = r.vec(s.dim, "center");
        s.diameter = r.f64();
        s.volume = r.f64();
        s.has_bounds = r.flag();
        for (auto *rate : {&s.cs, &s.cc, &s.cmu, &s.c1, &s.damps, &s.acov})
            *rate = get_float(r);
        s.verbose = r.flag();
        s.one_plus_one = r.flag();
        s.seed = r.opt([&] { return r.to_int(r.i64()); });
        s.fast_sampling = r.flag();
        s.C0 = r.opt([&] { return r.mat(s.dim, s.dim, "C0"); });
        s.mu0_given = r.flag();
        return s;
    }

    // --------------------------------------------------------------------------------------------
    // Statistics, population, weights
    // --------------------------------------------------------------------------------------------
    static void put(Writer &w, const parameters::Stats &s)
    {
        w.size(s.t);
        w.size(s.t_run);
        w.size(s.evaluations);
        w.f64(s.current_avg);
        w.list(s.solutions, [&](const Solution &x) { put(w, x); });
        w.list(s.centers, [&](const parameters::Center &c) { put(w, c); });
        put(w, s.current_best);
        put(w, s.global_best);
        w.flag(s.has_improved);
        w.f64(s.success_ratio);
        w.f64(s.cp);
        w.size(s.last_update);
        w.size(s.n_updates);
    }

    static void get(Reader &r, parameters::Stats &s)
    {
        s.t = r.size();
        s.t_run = r.size();
        s.evaluations = r.size();
        s.current_avg = r.f64();
        s.solutions = r.list([&] { return get_solution(r); });
        s.centers = r.list([&] {
            parameters::Center c;
            get(r, c);
            return c;
        });
        get(r, s.current_best);
        get(r, s.global_best);
        s.has_improved = r.flag();
        s.success_ratio = r.f64();
        s.cp = r.f64();
        s.last_update = r.size();
        s.n_updates = r.size();
    }

    // Not saved: the buffers of sort() and reorder() (order, scratch, scratch_vector). The
    // matrices are saved whole, with the columns past n (their capacity; the points dropped by
    // the elitist selection, n_pool).
    static void put(Writer &w, const Population &p)
    {
        for (const Matrix *m : {&p.X, &p.X_transformed, &p.Z, &p.Y, &p.S})
            w.mat(*m);
        w.vec(p.f);
        w.vec(p.t);
        w.size(p.d);
        w.size(p.n);
        w.size(p.n_pool);
        w.size(p.n_ranked);
    }

    static void get(Reader &r, Population &p)
    {
        for (Matrix *m : {&p.X, &p.X_transformed, &p.Z, &p.Y, &p.S})
            *m = r.mat();
        p.f = r.vec();
        p.t = r.vec();
        p.d = r.size();
        p.n = r.size();
        p.n_pool = r.size();
        p.n_ranked = r.size();
        const auto rows = p.X.rows(), cols = p.X.cols();
        for (const Matrix *m : {&p.X_transformed, &p.Z, &p.Y, &p.S})
            Reader::check(m->rows() == rows && m->cols() == cols,
                          "the matrices of a population differ in shape");
        Reader::check(p.f.size() == cols && p.t.size() == cols && p.n <= p.capacity() &&
                          p.n_pool <= p.capacity() && static_cast<size_t>(rows) == p.d,
                      "the sizes of a population");
    }

    static void put(Writer &w, const parameters::Weights &x)
    {
        for (const Vector *v : {&x.weights, &x.positive, &x.negative, &x.negative_reference})
            w.vec(*v);
        for (const Float f : {x.mueff, x.mueff_neg, x.mueff_mean})
            w.f64(f);
        w.flag(x.utility);
        w.flag(x.pairwise);
        for (const Float f : {x.c1,
                              x.cmu,
                              x.cc,
                              x.cs,
                              x.cs_path,
                              x.damps,
                              x.acov,
                              x.sqrt_cc_mueff,
                              x.sqrt_cs_mueff,
                              x.lazy_update_interval,
                              x.expected_length_z,
                              x.expected_length_ps,
                              x.beta,
                              x.int_lb_sigma})
            w.f64(f);
    }

    static void get(Reader &r, parameters::Weights &x)
    {
        for (Vector *v : {&x.weights, &x.positive, &x.negative, &x.negative_reference})
            *v = r.vec();
        for (Float *f : {&x.mueff, &x.mueff_neg, &x.mueff_mean})
            *f = r.f64();
        x.utility = r.flag();
        x.pairwise = r.flag();
        for (Float *f : {&x.c1,
                         &x.cmu,
                         &x.cc,
                         &x.cs,
                         &x.cs_path,
                         &x.damps,
                         &x.acov,
                         &x.sqrt_cc_mueff,
                         &x.sqrt_cs_mueff,
                         &x.lazy_update_interval,
                         &x.expected_length_z,
                         &x.expected_length_ps,
                         &x.beta,
                         &x.int_lb_sigma})
            *f = r.f64();
    }

    //! A Weights object to read into (every member is overwritten)
    static parameters::Weights blank_weights()
    {
        return parameters::Weights(parameters::Settings(1), 1.0);
    }

    // --------------------------------------------------------------------------------------------
    // Samplers
    // --------------------------------------------------------------------------------------------
    static void put(Writer &w, const sampling::Tester &s)
    {
        w.size(s.d);
        w.size(s.i);
    }

    static std::shared_ptr<sampling::Tester> make(Reader &r, Tag<sampling::Tester>)
    {
        auto s = std::make_shared<sampling::Tester>(r.size());
        s->i = r.size();
        return s;
    }

    static void put(Writer &w, const sampling::Gaussian &s)
    {
        w.size(s.d);
        put(w, s.dist);
        w.generator(s.generator);
    }

    static void get_into(Reader &r, sampling::Gaussian &s)
    {
        s.d = r.size();
        get(r, s.dist);
        s.generator = r.generator();
    }

    static std::shared_ptr<sampling::Gaussian> make(Reader &r, Tag<sampling::Gaussian>)
    {
        auto s = std::make_shared<sampling::Gaussian>(0);
        get_into(r, *s);
        return s;
    }

    static void put(Writer &w, const sampling::Uniform &s)
    {
        // rng::uniform has no state
        w.size(s.d);
        w.generator(s.generator);
    }

    static void get_into(Reader &r, sampling::Uniform &s)
    {
        s.d = r.size();
        s.generator = r.generator();
    }

    static std::shared_ptr<sampling::Uniform> make(Reader &r, Tag<sampling::Uniform>)
    {
        auto s = std::make_shared<sampling::Uniform>(0);
        get_into(r, *s);
        return s;
    }

    static void put(Writer &w, const sampling::FastGaussian &s)
    {
        w.size(s.d);
        put(w, s.generator);
    }

    static std::shared_ptr<sampling::FastGaussian> make(Reader &r, Tag<sampling::FastGaussian>)
    {
        // Seeded from a generator of its own (the state is overwritten): not from the global one
        rng::Generator unused;
        auto s = std::make_shared<sampling::FastGaussian>(r.size(), unused);
        get(r, s->generator);
        return s;
    }

    static void put(Writer &w, const sampling::Mirrored &s)
    {
        w.size(s.d);
        write_poly(w, s.sampler, "sampler");
        w.vec(s.previous);
        w.flag(s.mirror);
    }

    static std::shared_ptr<sampling::Sampler> required_sampler(Reader &r)
    {
        auto s = read_poly<sampling::Sampler>(r, "sampler");
        Reader::check(s != nullptr, "a sampler without the sampler it wraps");
        return s;
    }

    static std::shared_ptr<sampling::Mirrored> make(Reader &r, Tag<sampling::Mirrored>)
    {
        const size_t d = r.size();
        auto s = std::make_shared<sampling::Mirrored>(required_sampler(r));
        s->d = d;
        s->previous = r.vec();
        s->mirror = r.flag();
        return s;
    }

    // Not saved: the QR decomposition, the lengths and the workspace of next_block(), which
    // computes them for every block
    static void put(Writer &w, const sampling::Orthogonal &s)
    {
        w.size(s.d);
        write_poly(w, s.sampler, "sampler");
        w.size(s.n);
        w.size(s.remaining);
        w.index(s.block_size);
        w.index(s.current);
        w.mat(s.samples);
    }

    static std::shared_ptr<sampling::Orthogonal> make(Reader &r, Tag<sampling::Orthogonal>)
    {
        const size_t d = r.size();
        auto inner = required_sampler(r);
        Reader::check(inner->d == d, "the dimension of an orthogonal sampler");
        const size_t n = r.size();
        auto s = std::make_shared<sampling::Orthogonal>(inner, n);
        s->n = n;
        s->remaining = r.size();
        s->block_size = r.index();
        s->current = r.index();
        s->samples = r.mat(d, d, "the block of an orthogonal sampler");
        Reader::check(s->block_size >= 0 && static_cast<size_t>(s->block_size) <= d &&
                          s->current >= 0,
                      "the block of an orthogonal sampler");
        return s;
    }

    static void put(Writer &w, const sampling::Halton &s)
    {
        w.size(s.d);
        w.i64(s.index_);
        w.flag(s.scramble_);
        w.list(s.primes_, [&](const int p) { w.i64(p); });
        w.list(s.permutations_, [&](const std::vector<std::vector<int>> &per_prime) {
            w.list(per_prime, [&](const std::vector<int> &permutation) {
                w.list(permutation, [&](const int v) { w.i64(v); });
            });
        });
    }

    static std::shared_ptr<sampling::Halton> make(Reader &r, Tag<sampling::Halton>)
    {
        // Created without scrambling: no random numbers are drawn
        const size_t d = r.size();
        auto s = std::make_shared<sampling::Halton>(d, false, rng::global());
        s->index_ = r.to_int(r.i64());
        s->scramble_ = r.flag();
        s->primes_ = r.list([&] { return r.to_int(r.i64()); });
        s->permutations_ = r.list(
            [&] { return r.list([&] { return r.list([&] { return r.to_int(r.i64()); }); }); });
        Reader::check(s->primes_.size() == d && s->permutations_.size() == d,
                      "the primes of a Halton sampler");
        for (size_t j = 0; j < d; ++j)
        {
            const int base = s->primes_[j];
            Reader::check(base >= 2, "the primes of a Halton sampler");
            for (const auto &permutation : s->permutations_[j])
            {
                Reader::check(permutation.size() == static_cast<size_t>(base),
                              "a permutation of a Halton sampler");
                for (const int v : permutation)
                    Reader::check(v >= 0 && v < base, "a permutation of a Halton sampler");
            }
        }
        return s;
    }

    // Not saved: the Sobol points of the cache, which are computed again (the sampler draws them
    // in the order of the shuffler)
    static void put(Writer &w, const sampling::Sobol &s)
    {
        w.size(s.d);
        w.size(s.cache.dim);
        w.size(s.cache.n_samples);
        put(w, s.cache.shuffler);
    }

    static std::shared_ptr<sampling::Sobol> make(Reader &r, Tag<sampling::Sobol>)
    {
        const size_t d = r.size();
        const size_t dim = r.size();
        const size_t n_samples = r.size();
        Reader::check(d == dim && d >= 1 && n_samples >= 1 &&
                          n_samples <= std::numeric_limits<size_t>::max() / d,
                      "the cache of a Sobol sampler");
        // The constructor computes the points for the current constants (from a generator of its
        // own, whose shuffler is overwritten); again when the state's number of points differs
        auto unused = std::make_shared<rng::Generator>();
        auto s = std::make_shared<sampling::Sobol>(d, unused);
        auto &cache = s->cache;
        if (cache.n_samples != n_samples)
        {
            cache.n_samples = n_samples;
            cache.cache.assign(n_samples * d, 0.0);
            long long seed = 2;
            for (size_t i = 0; i < n_samples; i++)
                i8_sobol(static_cast<int>(d), &seed, cache.cache.data() + i * d);
        }
        get(r, cache.shuffler);
        Reader::check(cache.shuffler.start + cache.shuffler.n <= n_samples,
                      "the shuffler of a Sobol sampler");
        return s;
    }

    static void put(Writer &w, const sampling::CachedSampler &s)
    {
        w.size(s.d);
        write_poly(w, s.sampler, "sampler");
        w.list(s.cache, [&](const Vector &v) { w.vec(v); });
        w.size(s.index);
        w.size(s.n_samples);
    }

    static std::shared_ptr<sampling::CachedSampler> make(Reader &r, Tag<sampling::CachedSampler>)
    {
        const size_t d = r.size();
        auto s = std::make_shared<sampling::CachedSampler>(required_sampler(r));
        s->d = d;
        s->cache = r.list([&] { return r.vec(d, "a cached sample"); });
        s->index = r.size();
        s->n_samples = r.size();
        Reader::check(s->n_samples >= 1 && s->cache.size() == std::min(s->index, s->n_samples),
                      "the cache of a cached sampler");
        return s;
    }

    static void put_transformer(Writer &w, const sampling::SampleTransformer &s)
    {
        w.size(s.d);
        write_poly(w, s.sampler, "sampler");
        put_float(w, s.length);
    }

    static void get_transformer(Reader &r, sampling::SampleTransformer &s)
    {
        s.d = r.size();
        s.sampler = required_sampler(r);
        s.length = get_float(r);
    }

    template <typename T> static std::shared_ptr<T> make_transformer(Reader &r)
    {
        // Created around a placeholder, replaced by the sampler of the state
        auto s = std::make_shared<T>(std::make_shared<sampling::Tester>(1));
        get_transformer(r, *s);
        return s;
    }

    static void put(Writer &w, const sampling::IdentityTransformer &s) { put_transformer(w, s); }
    static void put(Writer &w, const sampling::UniformScaler &s) { put_transformer(w, s); }
    static void put(Writer &w, const sampling::LaplaceTransformer &s) { put_transformer(w, s); }
    // Not saved: scratch, a buffer of transform_inplace()
    static void put(Writer &w, const sampling::LogisticTransformer &s) { put_transformer(w, s); }
    static void put(Writer &w, const sampling::CauchyTransformer &s) { put_transformer(w, s); }
    static void put(Writer &w, const sampling::DoubleWeibullTransformer &s)
    {
        put_transformer(w, s);
    }

    static void put(Writer &w, const sampling::GaussianTransformer &s)
    {
        // The Box-Muller state for odd dimensions (constants::use_box_muller)
        put_transformer(w, s);
        w.vec(s.u_extra);
        w.index(s.n_extra_used);
        w.flag(s.needs_new_sample);
        w.f64(s.extra_sample);
    }

    static void get_gaussian(Reader &r, sampling::GaussianTransformer &s)
    {
        get_transformer(r, s);
        s.u_extra = r.vec();
        s.n_extra_used = r.index();
        s.needs_new_sample = r.flag();
        s.extra_sample = r.f64();
        Reader::check(s.n_extra_used >= 0, "the Box-Muller state of a Gaussian transformer");
    }

#define MODCMA_MAKE_TRANSFORMER(T)                                                                 \
    static std::shared_ptr<sampling::T> make(Reader &r, Tag<sampling::T>)                          \
    {                                                                                              \
        return make_transformer<sampling::T>(r);                                                   \
    }
    MODCMA_MAKE_TRANSFORMER(IdentityTransformer)
    MODCMA_MAKE_TRANSFORMER(UniformScaler)
    MODCMA_MAKE_TRANSFORMER(LaplaceTransformer)
    MODCMA_MAKE_TRANSFORMER(LogisticTransformer)
    MODCMA_MAKE_TRANSFORMER(CauchyTransformer)
    MODCMA_MAKE_TRANSFORMER(DoubleWeibullTransformer)
#undef MODCMA_MAKE_TRANSFORMER

    static std::shared_ptr<sampling::GaussianTransformer> make(Reader &r,
                                                               Tag<sampling::GaussianTransformer>)
    {
        auto s =
            std::make_shared<sampling::GaussianTransformer>(std::make_shared<sampling::Tester>(1));
        get_gaussian(r, *s);
        return s;
    }

    // --------------------------------------------------------------------------------------------
    // Matrix adaptation
    // --------------------------------------------------------------------------------------------
    // Not saved: the buffers of the updates (scratch_vector, weights_scratch, positive_scratch,
    // square_scratch, batch_scratch, batch_scratch2, steps_scratch, steps_scratch2), written
    // before they are read
    static void put_adaptation(Writer &w, const matrix_adaptation::Adaptation &a)
    {
        w.size(static_cast<size_t>(a.m.size()));
        for (const Vector *v : {&a.m, &a.m_old, &a.dm, &a.ps, &a.dz, &a.coordinate_wise_variances})
            w.vec(*v);
        w.f64(a.dd);
        w.size(a.factor_version);
        w.vec(a.metric_d);
        w.mat(a.metric_B);
        w.size(a.metric_version);
    }

    static void get_adaptation(Reader &r, matrix_adaptation::Adaptation &a, const size_t dim)
    {
        a.m = r.vec(dim, "the mean");
        a.m_old = r.vec(dim, "the previous mean");
        a.dm = r.vec(dim, "the mean shift");
        a.ps = r.vec(dim, "the path ps");
        a.dz = r.vec();
        a.coordinate_wise_variances = r.vec(dim, "the variances");
        a.dd = r.f64();
        a.factor_version = r.size();
        a.metric_d = r.vec();
        a.metric_B = r.mat();
        a.metric_version = r.size();
    }

    template <typename T> static std::shared_ptr<T> make_adaptation(Reader &r, size_t &dim)
    {
        dim = r.size();
        Reader::check(dim >= 1, "the dimension of a matrix adaptation");
        auto a = std::make_shared<T>(dim, Vector::Zero(static_cast<Eigen::Index>(dim)));
        get_adaptation(r, *a, dim);
        return a;
    }

    static void put(Writer &w, const matrix_adaptation::None &a)
    {
        put_adaptation(w, a);
        w.vec(a.ones);
    }

    static std::shared_ptr<matrix_adaptation::None> make(Reader &r, Tag<matrix_adaptation::None>)
    {
        size_t d = 0;
        auto a = make_adaptation<matrix_adaptation::None>(r, d);
        a->ones = r.vec(d, "the axes of NONE");
        return a;
    }

    // Not saved: the eigensolver and the LAPACK workspace of the decomposition
    static void put(Writer &w, const matrix_adaptation::CovarianceAdaptation &a)
    {
        put_adaptation(w, a);
        w.vec(a.pc);
        w.vec(a.d);
        w.mat(a.B);
        w.mat(a.C);
        w.mat(a.A);
        w.flag(a.hs);
    }

    template <typename T> static std::shared_ptr<T> make_covariance(Reader &r)
    {
        size_t d = 0;
        auto a = make_adaptation<T>(r, d);
        a->pc = r.vec(d, "the path pc");
        a->d = r.vec(d, "the axes D");
        a->B = r.mat(d, d, "B");
        a->C = r.mat(d, d, "C");
        a->A = r.mat(d, d, "A");
        a->hs = r.flag();
        return a;
    }

    static std::shared_ptr<matrix_adaptation::CovarianceAdaptation>
    make(Reader &r, Tag<matrix_adaptation::CovarianceAdaptation>)
    {
        return make_covariance<matrix_adaptation::CovarianceAdaptation>(r);
    }

    static std::shared_ptr<matrix_adaptation::CovarianceNoEigvAdaptation>
    make(Reader &r, Tag<matrix_adaptation::CovarianceNoEigvAdaptation>)
    {
        return make_covariance<matrix_adaptation::CovarianceNoEigvAdaptation>(r);
    }

    static void put(Writer &w, const matrix_adaptation::SeparableAdaptation &a)
    {
        put_adaptation(w, a);
        w.vec(a.pc);
        w.vec(a.d);
        w.vec(a.c);
        w.flag(a.hs);
    }

    static std::shared_ptr<matrix_adaptation::SeparableAdaptation>
    make(Reader &r, Tag<matrix_adaptation::SeparableAdaptation>)
    {
        size_t d = 0;
        auto a = make_adaptation<matrix_adaptation::SeparableAdaptation>(r, d);
        a->pc = r.vec(d, "the path pc");
        a->d = r.vec(d, "the axes D");
        a->c = r.vec(d, "the variances c");
        a->hs = r.flag();
        return a;
    }

    // Not saved: the buffers of the Woodbury update (wood_*); the exact recomputation of M_inv
    // counts the generations of the run, stats.t_run
    static void put(Writer &w, const matrix_adaptation::MatrixAdaptation &a)
    {
        put_adaptation(w, a);
        w.mat(a.M);
        w.mat(a.M_inv);
        w.flag(a.hs);
        w.size(a.inverse_fallbacks);
        w.flag(a.outdated_M_inv);
    }

    static std::shared_ptr<matrix_adaptation::MatrixAdaptation>
    make(Reader &r, Tag<matrix_adaptation::MatrixAdaptation>)
    {
        size_t d = 0;
        auto a = make_adaptation<matrix_adaptation::MatrixAdaptation>(r, d);
        a->M = r.mat(d, d, "M");
        a->M_inv = r.mat(d, d, "M_inv");
        a->hs = r.flag();
        a->inverse_fallbacks = r.size();
        a->outdated_M_inv = r.flag();
        return a;
    }

    static void put(Writer &w, const matrix_adaptation::CholeskyAdaptation &a)
    {
        put_adaptation(w, a);
        w.mat(a.A);
        w.vec(a.pc);
        w.flag(a.hs);
        w.size(a.skipped_downdates);
    }

    static std::shared_ptr<matrix_adaptation::CholeskyAdaptation>
    make(Reader &r, Tag<matrix_adaptation::CholeskyAdaptation>)
    {
        size_t d = 0;
        auto a = make_adaptation<matrix_adaptation::CholeskyAdaptation>(r, d);
        a->A = r.mat(d, d, "A");
        a->pc = r.vec(d, "the path pc");
        a->hs = r.flag();
        a->skipped_downdates = r.size();
        return a;
    }

    static void put(Writer &w, const matrix_adaptation::SelfAdaptation &a)
    {
        put_adaptation(w, a);
        w.mat(a.A);
        w.mat(a.C);
    }

    static std::shared_ptr<matrix_adaptation::SelfAdaptation>
    make(Reader &r, Tag<matrix_adaptation::SelfAdaptation>)
    {
        size_t d = 0;
        auto a = make_adaptation<matrix_adaptation::SelfAdaptation>(r, d);
        a->A = r.mat(d, d, "A");
        a->C = r.mat(d, d, "C");
        return a;
    }

    static void put(Writer &w, const matrix_adaptation::NaturalGradientAdaptation &a)
    {
        put_adaptation(w, a);
        w.mat(a.A);
        w.mat(a.A_inv);
        w.f64(a.sigma_g);
        w.mat(a.gradient_samples);
        w.vec(a.gradient_weights);
        w.f64(a.gradient_shift);
        w.mat(a.gradient_matrix);
        w.flag(a.outdated_A_inv);
    }

    static std::shared_ptr<matrix_adaptation::NaturalGradientAdaptation>
    make(Reader &r, Tag<matrix_adaptation::NaturalGradientAdaptation>)
    {
        size_t d = 0;
        auto a = make_adaptation<matrix_adaptation::NaturalGradientAdaptation>(r, d);
        a->A = r.mat(d, d, "A");
        a->A_inv = r.mat(d, d, "A_inv");
        a->sigma_g = r.f64();
        a->gradient_samples = r.mat();
        a->gradient_weights = r.vec();
        a->gradient_shift = r.f64();
        a->gradient_matrix = r.mat();
        a->outdated_A_inv = r.flag();
        Reader::check(static_cast<size_t>(a->gradient_samples.rows()) == d &&
                          a->gradient_weights.size() == a->gradient_samples.cols(),
                      "the natural gradient");
        return a;
    }

    // --------------------------------------------------------------------------------------------
    // Mutation (step-size adaptation)
    // --------------------------------------------------------------------------------------------
    static void put(Writer &w, const mutation::ThresholdConvergence &t)
    {
        w.f64(t.init_threshold);
        w.f64(t.decay_factor);
    }

    template <typename T> static std::shared_ptr<T> make_threshold(Reader &r)
    {
        auto t = std::make_shared<T>();
        t->init_threshold = r.f64();
        t->decay_factor = r.f64();
        return t;
    }

    static std::shared_ptr<mutation::ThresholdConvergence> make(Reader &r,
                                                                Tag<mutation::ThresholdConvergence>)
    {
        return make_threshold<mutation::ThresholdConvergence>(r);
    }

    static std::shared_ptr<mutation::NoThresholdConvergence>
    make(Reader &r, Tag<mutation::NoThresholdConvergence>)
    {
        return make_threshold<mutation::NoThresholdConvergence>(r);
    }

    static void put(Writer &w, const mutation::SequentialSelection &s)
    {
        w.f64(s.seq_cutoff_factor);
    }

    template <typename T> static std::shared_ptr<T> make_sequential(Reader &r)
    {
        // Mirror::NONE keeps the factor as it is given
        return std::make_shared<T>(parameters::Mirror::NONE, r.f64());
    }

    static std::shared_ptr<mutation::SequentialSelection> make(Reader &r,
                                                               Tag<mutation::SequentialSelection>)
    {
        return make_sequential<mutation::SequentialSelection>(r);
    }

    static std::shared_ptr<mutation::NoSequentialSelection>
    make(Reader &r, Tag<mutation::NoSequentialSelection>)
    {
        return make_sequential<mutation::NoSequentialSelection>(r);
    }

    static void put(Writer &w, const mutation::SigmaSampler &s)
    {
        // Its sampler is a member, a GaussianTransformer of a one-dimensional sampler
        put(w, s.sampler);
    }

    template <typename T> static std::shared_ptr<T> make_sigma_sampler(Reader &r)
    {
        auto s = std::make_shared<T>(1.0, rng::global());
        get_gaussian(r, s->sampler);
        return s;
    }

    static std::shared_ptr<mutation::SigmaSampler> make(Reader &r, Tag<mutation::SigmaSampler>)
    {
        return make_sigma_sampler<mutation::SigmaSampler>(r);
    }

    static std::shared_ptr<mutation::NoSigmaSampler> make(Reader &r, Tag<mutation::NoSigmaSampler>)
    {
        return make_sigma_sampler<mutation::NoSigmaSampler>(r);
    }

    static void put(Writer &w, const mutation::Strategy &s)
    {
        write_poly(w, s.tc, "threshold convergence");
        write_poly(w, s.sq, "sequential selection");
        write_poly(w, s.ss, "sigma sampler");
        w.f64(s.sigma);
        w.f64(s.s);
        w.f64(s.generation_threshold);
        w.flag(s.partner_sampled);
    }

    template <typename T> static std::shared_ptr<T> make_strategy(Reader &r)
    {
        auto tc = read_poly<mutation::ThresholdConvergence>(r, "threshold convergence");
        auto sq = read_poly<mutation::SequentialSelection>(r, "sequential selection");
        auto ss = read_poly<mutation::SigmaSampler>(r, "sigma sampler");
        Reader::check(tc && sq && ss, "a step-size adaptation without its parts");
        const Float sigma = r.f64();
        auto s = std::make_shared<T>(tc, sq, ss, sigma);
        mutation::Strategy &base = *s;
        base.s = r.f64();
        base.generation_threshold = r.f64();
        base.partner_sampled = r.flag();
        return s;
    }

#define MODCMA_MAKE_STRATEGY(T)                                                                    \
    static std::shared_ptr<mutation::T> make(Reader &r, Tag<mutation::T>)                          \
    {                                                                                              \
        return make_strategy<mutation::T>(r);                                                      \
    }
    MODCMA_MAKE_STRATEGY(Strategy)
    MODCMA_MAKE_STRATEGY(CSA)
    MODCMA_MAKE_STRATEGY(MXNES)
    MODCMA_MAKE_STRATEGY(LPXNES)
    MODCMA_MAKE_STRATEGY(SR)
#undef MODCMA_MAKE_STRATEGY

    static void put(Writer &w, const mutation::TPA &s)
    {
        put(w, static_cast<const mutation::Strategy &>(s));
        for (const Float f : {s.a_tpa, s.b_tpa, s.rank_tpa, s.sigma_old})
            w.f64(f);
    }

    static std::shared_ptr<mutation::TPA> make(Reader &r, Tag<mutation::TPA>)
    {
        auto s = make_strategy<mutation::TPA>(r);
        for (Float *f : {&s->a_tpa, &s->b_tpa, &s->rank_tpa, &s->sigma_old})
            *f = r.f64();
        return s;
    }

    static void put(Writer &w, const mutation::MSR &s)
    {
        put(w, static_cast<const mutation::Strategy &>(s));
        w.f64(s.quantile);
    }

    static std::shared_ptr<mutation::MSR> make(Reader &r, Tag<mutation::MSR>)
    {
        auto s = make_strategy<mutation::MSR>(r);
        s->quantile = r.f64();
        return s;
    }

    // Not saved: combined, idx and ranks, the buffers of adapt()
    static void put(Writer &w, const mutation::PSR &s)
    {
        put(w, static_cast<const mutation::Strategy &>(s));
        w.f64(s.success_ratio);
    }

    static std::shared_ptr<mutation::PSR> make(Reader &r, Tag<mutation::PSR>)
    {
        auto s = make_strategy<mutation::PSR>(r);
        s->success_ratio = r.f64();
        return s;
    }

    // Not saved: order and weights, the buffers of on_population_complete()
    static void put(Writer &w, const mutation::XNES &s)
    {
        put(w, static_cast<const mutation::Strategy &>(s));
        put_float(w, s.offspring_statistic);
    }

    static std::shared_ptr<mutation::XNES> make(Reader &r, Tag<mutation::XNES>)
    {
        auto s = make_strategy<mutation::XNES>(r);
        s->offspring_statistic = get_float(r);
        return s;
    }

    static void put(Writer &w, const mutation::SA &s)
    {
        put(w, static_cast<const mutation::Strategy &>(s));
        w.f64(s.mean_sigma);
    }

    static std::shared_ptr<mutation::SA> make(Reader &r, Tag<mutation::SA>)
    {
        auto s = make_strategy<mutation::SA>(r);
        s->mean_sigma = r.f64();
        return s;
    }

    static void put(Writer &w, const mutation::TPAInPlace &s)
    {
        put(w, static_cast<const mutation::Strategy &>(s));
        w.f64(s.rank_difference);
        w.flag(s.test_points);
        w.vec(s.direction);
        w.flag(s.pending);
    }

    static std::shared_ptr<mutation::TPAInPlace> make(Reader &r, Tag<mutation::TPAInPlace>)
    {
        auto s = make_strategy<mutation::TPAInPlace>(r);
        s->rank_difference = r.f64();
        s->test_points = r.flag();
        s->direction = r.vec();
        s->pending = r.flag();
        return s;
    }

    // --------------------------------------------------------------------------------------------
    // Selection, center placement: no state but their types
    // --------------------------------------------------------------------------------------------
    // Not saved: order and loser, the buffers of Pairwise::sort()
    static void put(Writer &, const selection::Pairwise &) {}
    static void put(Writer &, const selection::NoPairwise &) {}
    static void put(Writer &, const selection::Elitsm &) {}
    static void put(Writer &, const selection::NoElitsm &) {}

    template <typename T> static std::shared_ptr<T> make_plain(Reader &)
    {
        return std::make_shared<T>();
    }

#define MODCMA_MAKE_PLAIN(T)                                                                       \
    static std::shared_ptr<T> make(Reader &r, Tag<T>) { return make_plain<T>(r); }
    MODCMA_MAKE_PLAIN(selection::Pairwise)
    MODCMA_MAKE_PLAIN(selection::NoPairwise)
    MODCMA_MAKE_PLAIN(selection::Elitsm)
    MODCMA_MAKE_PLAIN(selection::NoElitsm)
    MODCMA_MAKE_PLAIN(center::X0)
    MODCMA_MAKE_PLAIN(center::Uniform)
    MODCMA_MAKE_PLAIN(center::Zero)
    MODCMA_MAKE_PLAIN(center::Center)
    MODCMA_MAKE_PLAIN(center::NoveltyWeighted)
    MODCMA_MAKE_PLAIN(center::MaximinTaboo)
#undef MODCMA_MAKE_PLAIN

    static void put(Writer &, const center::Placement &) {}

    static void put(Writer &w, const selection::Strategy &s)
    {
        write_poly(w, s.pairwise, "pairwise selection");
        write_poly(w, s.elitsm, "elitism");
    }

    static void get(Reader &r, selection::Strategy &s)
    {
        s.pairwise = read_poly<selection::Pairwise>(r, "pairwise selection");
        s.elitsm = read_poly<selection::Elitsm>(r, "elitism");
    }

    static std::shared_ptr<selection::Strategy> make_selection(Reader &r)
    {
        auto s = std::make_shared<selection::Strategy>(parameters::Modules{});
        get(r, *s);
        return s;
    }

    // --------------------------------------------------------------------------------------------
    // Restart criteria and strategies
    // --------------------------------------------------------------------------------------------
    static void put_criterion(Writer &w, const restart::Criterion &c)
    {
        w.flag(c.met);
        w.str(c.name);
        w.size(c.last_restart);
        w.flag(c.was_reset);
    }

    template <typename T> static std::shared_ptr<T> make_criterion(Reader &r)
    {
        auto c = std::make_shared<T>();
        c->met = r.flag();
        c->name = r.str();
        c->last_restart = r.size();
        c->was_reset = r.flag();
        return c;
    }

#define MODCMA_PLAIN_CRITERION(T)                                                                  \
    static void put(Writer &w, const restart::T &c) { put_criterion(w, c); }                       \
    static std::shared_ptr<restart::T> make(Reader &r, Tag<restart::T>)                            \
    {                                                                                              \
        return make_criterion<restart::T>(r);                                                      \
    }
    MODCMA_PLAIN_CRITERION(MinSigma)
    MODCMA_PLAIN_CRITERION(UnableToAdapt)
    MODCMA_PLAIN_CRITERION(ConditionC)
    MODCMA_PLAIN_CRITERION(NoEffectAxis)
    MODCMA_PLAIN_CRITERION(NoEffectCoord)
    MODCMA_PLAIN_CRITERION(TooMuchRepelling)
#undef MODCMA_PLAIN_CRITERION

    static void put(Writer &w, const restart::ExceededMaxIter &c)
    {
        put_criterion(w, c);
        w.size(c.max_iter);
    }

    static std::shared_ptr<restart::ExceededMaxIter> make(Reader &r, Tag<restart::ExceededMaxIter>)
    {
        auto c = make_criterion<restart::ExceededMaxIter>(r);
        c->max_iter = r.size();
        return c;
    }

    static void put(Writer &w, const restart::NoImprovement &c)
    {
        put_criterion(w, c);
        w.size(c.n_bin);
        w.floats(c.best_fitnesses);
    }

    static std::shared_ptr<restart::NoImprovement> make(Reader &r, Tag<restart::NoImprovement>)
    {
        auto c = make_criterion<restart::NoImprovement>(r);
        c->n_bin = r.size();
        c->best_fitnesses = r.floats();
        return c;
    }

    static void put(Writer &w, const restart::MaxSigma &c)
    {
        put_criterion(w, c);
        w.f64(c.start);
    }

    static std::shared_ptr<restart::MaxSigma> make(Reader &r, Tag<restart::MaxSigma>)
    {
        auto c = make_criterion<restart::MaxSigma>(r);
        c->start = r.f64();
        return c;
    }

    static void put(Writer &w, const restart::FlatFitness &c)
    {
        put_criterion(w, c);
        w.size(c.max_flat_fitness);
        w.size(c.flat_fitness_index);
        w.size(static_cast<size_t>(c.flat_fitnesses.size()));
        for (const int v : c.flat_fitnesses)
            w.i64(v);
    }

    static std::shared_ptr<restart::FlatFitness> make(Reader &r, Tag<restart::FlatFitness>)
    {
        auto c = make_criterion<restart::FlatFitness>(r);
        c->max_flat_fitness = r.size();
        c->flat_fitness_index = r.size();
        c->flat_fitnesses.resize(static_cast<Eigen::Index>(r.count(8)));
        for (auto &v : c->flat_fitnesses)
            v = r.to_int(r.i64());
        return c;
    }

    static void put(Writer &w, const restart::TolX &c)
    {
        put_criterion(w, c);
        w.vec(c.tolx_vector);
        w.f64(c.sigma_start);
    }

    static std::shared_ptr<restart::TolX> make(Reader &r, Tag<restart::TolX>)
    {
        auto c = make_criterion<restart::TolX>(r);
        c->tolx_vector = r.vec();
        c->sigma_start = r.f64();
        return c;
    }

    static void put(Writer &w, const restart::MaxDSigma &c)
    {
        put_criterion(w, c);
        w.f64(c.sigma_start);
    }

    static std::shared_ptr<restart::MaxDSigma> make(Reader &r, Tag<restart::MaxDSigma>)
    {
        auto c = make_criterion<restart::MaxDSigma>(r);
        c->sigma_start = r.f64();
        return c;
    }

    static void put(Writer &w, const restart::MinDSigma &c)
    {
        put_criterion(w, c);
        w.f64(c.sigma_start);
    }

    static std::shared_ptr<restart::MinDSigma> make(Reader &r, Tag<restart::MinDSigma>)
    {
        auto c = make_criterion<restart::MinDSigma>(r);
        c->sigma_start = r.f64();
        return c;
    }

    // The sequence and its split; the two halves (multisets) are built again from them
    static void put(Writer &w, const restart::SplitMedian &m)
    {
        w.floats(m.values_);
        w.size(m.split_);
    }

    static void get(Reader &r, restart::SplitMedian &m)
    {
        m.assign(r.floats());
        const size_t split = r.size();
        Reader::check(split <= m.values_.size(), "the split of a running median");
        m.set_split(split);
    }

    static void put(Writer &w, const restart::Stagnation &c)
    {
        put_criterion(w, c);
        w.size(c.n_stagnation);
        put(w, c.median_fitnesses);
        put(w, c.best_fitnesses);
    }

    static std::shared_ptr<restart::Stagnation> make(Reader &r, Tag<restart::Stagnation>)
    {
        auto c = make_criterion<restart::Stagnation>(r);
        c->n_stagnation = r.size();
        get(r, c->median_fitnesses);
        get(r, c->best_fitnesses);
        return c;
    }

    static void put(Writer &w, const restart::Criteria &c)
    {
        w.list(c.items, [&](const std::shared_ptr<restart::Criterion> &x) {
            write_poly(w, x, "restart criterion");
        });
    }

    static restart::Criteria get_criteria(Reader &r)
    {
        return restart::Criteria(
            r.list([&] { return read_poly<restart::Criterion>(r, "restart criterion"); }));
    }

    static void put(Writer &w, const restart::Strategy &s) { w.size(s.max_iter); }

    static std::shared_ptr<restart::Strategy> make(Reader &r, Tag<restart::Strategy>)
    {
        auto s = std::make_shared<restart::Strategy>();
        s->max_iter = r.size();
        return s;
    }

    static void put(Writer &w, const restart::IPOP &s)
    {
        w.size(s.max_iter);
        w.f64(s.ipop_factor);
        w.size(s.max_lambda_factor);
    }

    static std::shared_ptr<restart::IPOP> make(Reader &r, Tag<restart::IPOP>)
    {
        auto s = std::make_shared<restart::IPOP>();
        s->max_iter = r.size();
        s->ipop_factor = r.f64();
        s->max_lambda_factor = r.size();
        return s;
    }

    static void put(Writer &w, const restart::BIPOP &s)
    {
        w.size(s.max_iter);
        for (const size_t v : {s.lambda_init,
                               s.max_lambda_factor,
                               s.lambda_large,
                               s.lambda_small,
                               s.used_small,
                               s.used_large,
                               s.used_budget})
            w.size(v);
        w.flag(s.large_run);
    }

    static std::shared_ptr<restart::BIPOP> make(Reader &r, Tag<restart::BIPOP>)
    {
        const size_t max_iter = r.size();
        auto s = std::make_shared<restart::BIPOP>(r.size());
        s->max_iter = max_iter;
        for (size_t *v : {&s->max_lambda_factor,
                          &s->lambda_large,
                          &s->lambda_small,
                          &s->used_small,
                          &s->used_large,
                          &s->used_budget})
            *v = r.size();
        s->large_run = r.flag();
        return s;
    }

    // --------------------------------------------------------------------------------------------
    // Bound correction
    // --------------------------------------------------------------------------------------------
    static void put(Writer &w, const bounds::BoundCorrection &b) { w.size(b.n_out_of_bounds); }

    template <typename T> static std::shared_ptr<T> make_bounds(Reader &r)
    {
        auto b = std::make_shared<T>();
        b->n_out_of_bounds = r.size();
        return b;
    }

#define MODCMA_MAKE_BOUNDS(T)                                                                      \
    static std::shared_ptr<bounds::T> make(Reader &r, Tag<bounds::T>)                              \
    {                                                                                              \
        return make_bounds<bounds::T>(r);                                                          \
    }
    MODCMA_MAKE_BOUNDS(NoCorrection)
    MODCMA_MAKE_BOUNDS(Resample)
    MODCMA_MAKE_BOUNDS(Mirror)
    MODCMA_MAKE_BOUNDS(Saturate)
    MODCMA_MAKE_BOUNDS(Toroidal)
#undef MODCMA_MAKE_BOUNDS

    static void put(Writer &w, const bounds::COTN &b)
    {
        w.size(b.n_out_of_bounds);
        put(w, b.sampler);
    }

    static std::shared_ptr<bounds::COTN> make(Reader &r, Tag<bounds::COTN>)
    {
        auto b = std::make_shared<bounds::COTN>(1, rng::global());
        b->n_out_of_bounds = r.size();
        get_into(r, b->sampler);
        return b;
    }

    static void put(Writer &w, const bounds::UniformResample &b)
    {
        w.size(b.n_out_of_bounds);
        put(w, b.sampler);
    }

    static std::shared_ptr<bounds::UniformResample> make(Reader &r, Tag<bounds::UniformResample>)
    {
        auto b = std::make_shared<bounds::UniformResample>(1, rng::global());
        b->n_out_of_bounds = r.size();
        get_into(r, b->sampler);
        return b;
    }

    // --------------------------------------------------------------------------------------------
    // Repelling
    // --------------------------------------------------------------------------------------------
    static void put(Writer &w, const repelling::TabooPoint &t)
    {
        put(w, t.solution);
        for (const Float f : {t.radius, t.shrinkage})
            w.f64(f);
        w.i64(t.n_rep);
        for (const Float f : {t.criticality, t.min_radius, t.max_radius, t.duplicate_evaluations})
            w.f64(f);
        w.i64(t.rejected_count);
        w.i64(t.checked_count);
        w.f64(t.last_rejection_rate);
        w.mat(t.metric);
    }

    static repelling::TabooPoint get_taboo_point(Reader &r)
    {
        repelling::TabooPoint t(Solution{}, 0.0);
        get(r, t.solution);
        t.radius = r.f64();
        t.shrinkage = r.f64();
        t.n_rep = r.to_int(r.i64());
        for (Float *f : {&t.criticality, &t.min_radius, &t.max_radius, &t.duplicate_evaluations})
            *f = r.f64();
        t.rejected_count = r.to_int(r.i64());
        t.checked_count = r.to_int(r.i64());
        t.last_rejection_rate = r.f64();
        t.metric = r.mat();
        return t;
    }

    static void put_repelling(Writer &w, const repelling::Repelling &x)
    {
        w.list(x.archive, [&](const repelling::TabooPoint &t) { put(w, t); });
        w.i64(x.attempts);
        w.f64(x.coverage);
        // The archive update after a restart, between its hill-valley points
        const auto &q = x.query;
        w.flag(q.active);
        w.opt(q.candidate, [&](const Solution &s) { put(w, s); });
        w.size(q.j);
        w.size(q.k);
        w.flag(q.accept);
        w.vec(q.x);
    }

    static void get_repelling(Reader &r, repelling::Repelling &x)
    {
        x.archive = r.list([&] { return get_taboo_point(r); });
        x.attempts = r.to_int(r.i64());
        x.coverage = r.f64();
        auto &q = x.query;
        q.active = r.flag();
        q.candidate = r.opt([&] { return get_solution(r); });
        q.j = r.size();
        q.k = r.size();
        q.accept = r.flag();
        q.x = r.vec();
        Reader::check(!q.active || (q.candidate && q.j < x.archive.size()),
                      "the archive update of repelling");
    }

    static void put(Writer &w, const repelling::NoRepelling &x) { put_repelling(w, x); }
    static void put(Writer &w, const repelling::CoverageRepelling &x) { put_repelling(w, x); }

    template <typename T> static std::shared_ptr<T> make_repelling(Reader &r)
    {
        auto x = std::make_shared<T>();
        get_repelling(r, *x);
        return x;
    }

    static std::shared_ptr<repelling::NoRepelling> make(Reader &r, Tag<repelling::NoRepelling>)
    {
        return make_repelling<repelling::NoRepelling>(r);
    }

    static std::shared_ptr<repelling::CoverageRepelling> make(Reader &r,
                                                              Tag<repelling::CoverageRepelling>)
    {
        return make_repelling<repelling::CoverageRepelling>(r);
    }

    static void put(Writer &w, const repelling::AdaptiveRepelling &x)
    {
        put_repelling(w, x);
        for (const Float f : {x.initial_radius_factor,
                              x.min_radius_factor,
                              x.max_radius_factor,
                              x.shrink_factor,
                              x.target_rejection_rate,
                              x.max_rejection_rate})
            w.f64(f);
        w.i64(x.last_restart_evaluations);
        for (const Float f : {x.average_restart_evaluations,
                              x.restart_eval_alpha,
                              x.grow_eta,
                              x.max_waste_ratio,
                              x.local_target_rejection_rate,
                              x.local_max_rejection_rate})
            w.f64(f);
        w.i64(x.min_checks_before_local_shrink);
        w.f64(x.restart_evals);
    }

    static std::shared_ptr<repelling::AdaptiveRepelling> make(Reader &r,
                                                              Tag<repelling::AdaptiveRepelling>)
    {
        auto x = make_repelling<repelling::AdaptiveRepelling>(r);
        for (Float *f : {&x->initial_radius_factor,
                         &x->min_radius_factor,
                         &x->max_radius_factor,
                         &x->shrink_factor,
                         &x->target_rejection_rate,
                         &x->max_rejection_rate})
            *f = r.f64();
        x->last_restart_evaluations = r.to_int(r.i64());
        for (Float *f : {&x->average_restart_evaluations,
                         &x->restart_eval_alpha,
                         &x->grow_eta,
                         &x->max_waste_ratio,
                         &x->local_target_rejection_rate,
                         &x->local_max_rejection_rate})
            *f = r.f64();
        x->min_checks_before_local_shrink = r.to_int(r.i64());
        x->restart_evals = r.f64();
        return x;
    }

    // --------------------------------------------------------------------------------------------
    // Coordinate mapping
    // --------------------------------------------------------------------------------------------
    static void put(Writer &w, const mapping::CoordinateMapping &m)
    {
        w.list(m.integer_idx, [&](const Eigen::Index i) { w.index(i); });
    }

    static std::shared_ptr<mapping::CoordinateMapping> make_mapping(Reader &r)
    {
        auto m = std::make_shared<mapping::CoordinateMapping>(Indices{}, 0);
        m->integer_idx = r.list([&] { return r.index(); });
        for (const auto i : m->integer_idx)
            Reader::check(i >= 0, "an integer variable");
        return m;
    }

    // --------------------------------------------------------------------------------------------
    // Parameters and ModularCMAES
    // --------------------------------------------------------------------------------------------
    static void put(Writer &w, const parameters::Parameters &p)
    {
        // The generator first: the modules refer to it
        w.generator(p.generator);
        w.flag(p.successful_adaptation);
        w.size(p.lambda);
        w.size(p.mu);
        put(w, p.settings);
        put(w, p.stats);
        write_poly(w, p.sampler, "sampler");
        put(w, p.weights);
        put(w, p.pop);
        put(w, p.old_pop);
        put(w, p.criteria);
        write_poly(w, p.adaptation, "matrix adaptation");
        write_poly(w, p.mutation, "step-size adaptation");
        w.flag(p.selection != nullptr);
        if (p.selection)
            put(w, *p.selection);
        write_poly(w, p.restart_strategy, "restart strategy");
        write_poly(w, p.bounds, "bound correction");
        write_poly(w, p.repelling, "repelling");
        write_poly(w, p.center_placement, "center placement");
        w.flag(p.coordinate_mapping != nullptr);
        if (p.coordinate_mapping)
            put(w, *p.coordinate_mapping);
        w.f64(p.restart_sigma);
    }

    static std::shared_ptr<parameters::Parameters> make_parameters(Reader &r)
    {
        auto generator = r.generator();
        const bool successful_adaptation = r.flag();
        const size_t lambda = r.size();
        const size_t mu = r.size();
        auto p = std::shared_ptr<parameters::Parameters>(
            new parameters::Parameters(get_settings(r), blank_weights()));
        p->generator = std::move(generator);
        p->successful_adaptation = successful_adaptation;
        p->lambda = lambda;
        p->mu = mu;
        get(r, p->stats);
        p->sampler = read_poly<sampling::Sampler>(r, "sampler");
        get(r, p->weights);
        get(r, p->pop);
        get(r, p->old_pop);
        p->criteria = get_criteria(r);
        p->adaptation = read_poly<matrix_adaptation::Adaptation>(r, "matrix adaptation");
        p->mutation = read_poly<mutation::Strategy>(r, "step-size adaptation");
        if (r.flag())
            p->selection = make_selection(r);
        p->restart_strategy = read_poly<restart::Strategy>(r, "restart strategy");
        p->bounds = read_poly<bounds::BoundCorrection>(r, "bound correction");
        p->repelling = read_poly<repelling::Repelling>(r, "repelling");
        p->center_placement = read_poly<center::Placement>(r, "center placement");
        if (r.flag())
            p->coordinate_mapping = make_mapping(r);
        p->restart_sigma = r.f64();

        const size_t dim = p->settings.dim;
        Reader::check(!p->adaptation || static_cast<size_t>(p->adaptation->m.size()) == dim,
                      "the dimension of the matrix adaptation");
        Reader::check(!p->sampler || p->sampler->d == dim, "the dimension of the sampler");
        Reader::check(p->pop.d == dim && p->old_pop.d == dim, "the dimension of the populations");
        if (p->coordinate_mapping)
            for (const auto i : p->coordinate_mapping->integer_idx)
                Reader::check(static_cast<size_t>(i) < dim, "an integer variable");
        return p;
    }

    // Not saved: values, the buffer of the objective values in step() and mutate()
    static void put(Writer &w, const ModularCMAES &es)
    {
        if (!es.p)
            throw std::invalid_argument("cannot save an optimizer without parameters");
        put(w, *es.p);
        w.u8(static_cast<uint8_t>(es.next));
        w.mat(es.batch);
        w.index(es.n_pending);
        w.list(es.requests, [&](const ModularCMAES::Request &q) {
            w.u8(static_cast<uint8_t>(q.source));
            w.index(q.begin);
            w.index(q.size);
            w.index(q.index);
        });
        w.index(es.next_individual);
        w.flag(es.ended);
        w.flag(es.steps_skipped);
        w.list(es.end_reasons, [&](const std::string &s) { w.str(s); });
        w.flag(es.gradient_open);
    }

    static ModularCMAES make_cmaes(Reader &r)
    {
        ModularCMAES es(make_parameters(r));
        es.next = r.enumeration(ModularCMAES::Step::UPDATE, "the step of the generation");
        es.batch = r.mat();
        es.n_pending = r.index();
        es.requests = r.list([&] {
            ModularCMAES::Request q{};
            q.source = r.enumeration(ModularCMAES::Source::HILL_VALLEY, "a request");
            q.begin = r.index();
            q.size = r.index();
            q.index = r.index();
            return q;
        });
        es.next_individual = r.index();
        es.ended = r.flag();
        es.steps_skipped = r.flag();
        es.end_reasons = r.list([&] { return r.str(); });
        es.gradient_open = r.flag();

        Reader::check(
            es.n_pending >= 0 && es.n_pending <= es.batch.cols() &&
                (es.n_pending == 0 || static_cast<size_t>(es.batch.rows()) == es.p->settings.dim),
            "the points of ask()");
        Eigen::Index end = 0;
        for (const auto &q : es.requests)
        {
            Reader::check(q.begin == end && q.size >= 1, "the requests of ask()");
            end += q.size;
        }
        Reader::check(end == es.n_pending, "the requests of ask()");
        Reader::check(!es.gradient_open || !es.requests.empty(), "the gradient injection");
        return es;
    }

    // --------------------------------------------------------------------------------------------
    // The objects of save() and load()
    // --------------------------------------------------------------------------------------------
    template <typename T> static void save_object(Writer &w, const T &object)
    {
        if constexpr (std::is_same_v<T, ModularCMAES> ||
                      std::is_same_v<T, parameters::Parameters> ||
                      std::is_same_v<T, parameters::Settings> ||
                      std::is_same_v<T, parameters::Modules> || std::is_same_v<T, Population> ||
                      std::is_same_v<T, parameters::Stats> || std::is_same_v<T, Solution> ||
                      std::is_same_v<T, parameters::Center> ||
                      std::is_same_v<T, parameters::Weights> || std::is_same_v<T, MetricCopy> ||
                      std::is_same_v<T, restart::Criteria> ||
                      std::is_same_v<T, selection::Strategy> ||
                      std::is_same_v<T, repelling::TabooPoint> ||
                      std::is_same_v<T, mapping::CoordinateMapping>)
            put(w, object);
        else
            write_poly<T>(w, &object, detail::Kind<T>::value);
    }

    template <typename T> static T load_object(Reader &r)
    {
        if constexpr (std::is_same_v<T, ModularCMAES>)
            return make_cmaes(r);
        else if constexpr (std::is_same_v<T, parameters::Parameters>)
            return std::move(*make_parameters(r));
        else if constexpr (std::is_same_v<T, parameters::Settings>)
            return get_settings(r);
        else if constexpr (std::is_same_v<T, parameters::Weights>)
        {
            auto weights = blank_weights();
            get(r, weights);
            return weights;
        }
        else if constexpr (std::is_same_v<T, restart::Criteria>)
            return get_criteria(r);
        else if constexpr (std::is_same_v<T, selection::Strategy>)
            return *make_selection(r);
        else if constexpr (std::is_same_v<T, repelling::TabooPoint>)
            return get_taboo_point(r);
        else if constexpr (std::is_same_v<T, mapping::CoordinateMapping>)
            return *make_mapping(r);
        else if constexpr (std::is_same_v<T, parameters::Modules> ||
                           std::is_same_v<T, Population> || std::is_same_v<T, parameters::Stats> ||
                           std::is_same_v<T, Solution> || std::is_same_v<T, parameters::Center> ||
                           std::is_same_v<T, MetricCopy>)
        {
            T object{};
            get(r, object);
            return object;
        }
        else
        {
            using Base = typename T::element_type;
            auto object = read_poly<Base>(r, detail::Kind<Base>::value);
            Reader::check(object != nullptr, "an empty object");
            return object;
        }
    }
};

// ------------------------------------------------------------------------------------------------
// The types of each polymorphic base
// ------------------------------------------------------------------------------------------------
const Registry<sampling::Sampler> &Access::registry(Tag<sampling::Sampler>)
{
    using namespace sampling;
    static const Registry<Sampler> r{{
        entry<Tester, Sampler>("Tester"),
        entry<Gaussian, Sampler>("Gaussian"),
        entry<Uniform, Sampler>("Uniform"),
        entry<FastGaussian, Sampler>("FastGaussian"),
        entry<Mirrored, Sampler>("Mirrored"),
        entry<Orthogonal, Sampler>("Orthogonal"),
        entry<Halton, Sampler>("Halton"),
        entry<Sobol, Sampler>("Sobol"),
        entry<CachedSampler, Sampler>("CachedSampler"),
        entry<IdentityTransformer, Sampler>("IdentityTransformer"),
        entry<GaussianTransformer, Sampler>("GaussianTransformer"),
        entry<UniformScaler, Sampler>("UniformScaler"),
        entry<LaplaceTransformer, Sampler>("LaplaceTransformer"),
        entry<LogisticTransformer, Sampler>("LogisticTransformer"),
        entry<CauchyTransformer, Sampler>("CauchyTransformer"),
        entry<DoubleWeibullTransformer, Sampler>("DoubleWeibullTransformer"),
    }};
    return r;
}

const Registry<matrix_adaptation::Adaptation> &Access::registry(Tag<matrix_adaptation::Adaptation>)
{
    using namespace matrix_adaptation;
    static const Registry<Adaptation> r{{
        entry<None, Adaptation>("None"),
        entry<CovarianceAdaptation, Adaptation>("CovarianceAdaptation"),
        entry<SeparableAdaptation, Adaptation>("SeparableAdaptation"),
        entry<MatrixAdaptation, Adaptation>("MatrixAdaptation"),
        entry<CholeskyAdaptation, Adaptation>("CholeskyAdaptation"),
        entry<SelfAdaptation, Adaptation>("SelfAdaptation"),
        entry<CovarianceNoEigvAdaptation, Adaptation>("CovarianceNoEigvAdaptation"),
        entry<NaturalGradientAdaptation, Adaptation>("NaturalGradientAdaptation"),
    }};
    return r;
}

const Registry<mutation::Strategy> &Access::registry(Tag<mutation::Strategy>)
{
    using namespace mutation;
    static const Registry<Strategy> r{{
        entry<Strategy, Strategy>("Strategy"),
        entry<CSA, Strategy>("CSA"),
        entry<TPA, Strategy>("TPA"),
        entry<MSR, Strategy>("MSR"),
        entry<PSR, Strategy>("PSR"),
        entry<XNES, Strategy>("XNES"),
        entry<MXNES, Strategy>("MXNES"),
        entry<LPXNES, Strategy>("LPXNES"),
        entry<SR, Strategy>("SR"),
        entry<SA, Strategy>("SA"),
        entry<TPAInPlace, Strategy>("TPAInPlace"),
    }};
    return r;
}

const Registry<mutation::ThresholdConvergence> &
Access::registry(Tag<mutation::ThresholdConvergence>)
{
    using namespace mutation;
    static const Registry<ThresholdConvergence> r{{
        entry<ThresholdConvergence, ThresholdConvergence>("ThresholdConvergence"),
        entry<NoThresholdConvergence, ThresholdConvergence>("NoThresholdConvergence"),
    }};
    return r;
}

const Registry<mutation::SequentialSelection> &Access::registry(Tag<mutation::SequentialSelection>)
{
    using namespace mutation;
    static const Registry<SequentialSelection> r{{
        entry<SequentialSelection, SequentialSelection>("SequentialSelection"),
        entry<NoSequentialSelection, SequentialSelection>("NoSequentialSelection"),
    }};
    return r;
}

const Registry<mutation::SigmaSampler> &Access::registry(Tag<mutation::SigmaSampler>)
{
    using namespace mutation;
    static const Registry<SigmaSampler> r{{
        entry<SigmaSampler, SigmaSampler>("SigmaSampler"),
        entry<NoSigmaSampler, SigmaSampler>("NoSigmaSampler"),
    }};
    return r;
}

const Registry<selection::Pairwise> &Access::registry(Tag<selection::Pairwise>)
{
    using namespace selection;
    static const Registry<Pairwise> r{{
        entry<Pairwise, Pairwise>("Pairwise"),
        entry<NoPairwise, Pairwise>("NoPairwise"),
    }};
    return r;
}

const Registry<selection::Elitsm> &Access::registry(Tag<selection::Elitsm>)
{
    using namespace selection;
    static const Registry<Elitsm> r{{
        entry<Elitsm, Elitsm>("Elitism"),
        entry<NoElitsm, Elitsm>("NoElitism"),
    }};
    return r;
}

const Registry<restart::Criterion> &Access::registry(Tag<restart::Criterion>)
{
    using namespace restart;
    static const Registry<Criterion> r{{
        entry<ExceededMaxIter, Criterion>("ExceededMaxIter"),
        entry<NoImprovement, Criterion>("NoImprovement"),
        entry<MaxSigma, Criterion>("MaxSigma"),
        entry<MinSigma, Criterion>("MinSigma"),
        entry<UnableToAdapt, Criterion>("UnableToAdapt"),
        entry<FlatFitness, Criterion>("FlatFitness"),
        entry<TolX, Criterion>("TolX"),
        entry<MaxDSigma, Criterion>("MaxDSigma"),
        entry<MinDSigma, Criterion>("MinDSigma"),
        entry<ConditionC, Criterion>("ConditionC"),
        entry<NoEffectAxis, Criterion>("NoEffectAxis"),
        entry<NoEffectCoord, Criterion>("NoEffectCoord"),
        entry<Stagnation, Criterion>("Stagnation"),
        entry<TooMuchRepelling, Criterion>("TooMuchRepelling"),
    }};
    return r;
}

const Registry<restart::Strategy> &Access::registry(Tag<restart::Strategy>)
{
    using namespace restart;
    static const Registry<Strategy> r{{
        entry<Strategy, Strategy>("Strategy"),
        entry<IPOP, Strategy>("IPOP"),
        entry<BIPOP, Strategy>("BIPOP"),
    }};
    return r;
}

const Registry<bounds::BoundCorrection> &Access::registry(Tag<bounds::BoundCorrection>)
{
    using namespace bounds;
    static const Registry<BoundCorrection> r{{
        entry<NoCorrection, BoundCorrection>("NoCorrection"),
        entry<Resample, BoundCorrection>("Resample"),
        entry<COTN, BoundCorrection>("COTN"),
        entry<Mirror, BoundCorrection>("Mirror"),
        entry<UniformResample, BoundCorrection>("UniformResample"),
        entry<Saturate, BoundCorrection>("Saturate"),
        entry<Toroidal, BoundCorrection>("Toroidal"),
    }};
    return r;
}

const Registry<repelling::Repelling> &Access::registry(Tag<repelling::Repelling>)
{
    using namespace repelling;
    static const Registry<Repelling> r{{
        entry<NoRepelling, Repelling>("NoRepelling"),
        entry<CoverageRepelling, Repelling>("CoverageRepelling"),
        entry<AdaptiveRepelling, Repelling>("AdaptiveRepelling"),
    }};
    return r;
}

const Registry<center::Placement> &Access::registry(Tag<center::Placement>)
{
    using namespace center;
    static const Registry<Placement> r{{
        entry<X0, Placement>("X0"),
        entry<Uniform, Placement>("Uniform"),
        entry<Zero, Placement>("Zero"),
        entry<Center, Placement>("Center"),
        entry<NoveltyWeighted, Placement>("NoveltyWeighted"),
        entry<MaximinTaboo, Placement>("MaximinTaboo"),
    }};
    return r;
}

// ------------------------------------------------------------------------------------------------
// save() and load()
// ------------------------------------------------------------------------------------------------
template <typename T> std::string save(const T &object)
{
    Writer w;
    w.out.reserve(16384);
    w.out.append(magic, sizeof magic);
    w.u32(format_version);
    w.str(detail::Kind<T>::value);
    Access::save_object(w, object);
    w.u32(crc32(w.out.data(), w.out.size()));
    return std::move(w.out);
}

template <typename T> T load(std::string_view state)
{
    const std::string expected = detail::Kind<T>::value;
    if (state.size() < sizeof magic || std::memcmp(state.data(), magic, sizeof magic) != 0)
        Reader::fail("not a modcma state (it does not start with the magic number of the format)");
    if (state.size() < header_size + trailer_size)
        Reader::fail("truncated: the state is shorter than its header");

    Reader r;
    r.in = state.substr(sizeof magic, 4);
    const uint32_t version = r.u32();
    if (version > format_version)
        Reader::fail("written by a newer modcma, in format version " + std::to_string(version) +
                     "; this modcma reads versions up to " + std::to_string(format_version) +
                     ": update modcma to load it");
    if (version == 0)
        Reader::fail("corrupted (format version 0)");

    const size_t body = state.size() - trailer_size;
    r.in = state.substr(body);
    r.pos = 0;
    if (r.u32() != crc32(state.data(), body))
        Reader::fail("corrupted or truncated: the checksum does not match");

    r.in = state.substr(header_size, body - header_size);
    r.pos = 0;
    r.version = version;
    const std::string kind = r.str();
    if (kind != expected)
        Reader::fail("the state holds a " + kind + ", not a " + expected);
    T object = Access::load_object<T>(r);
    if (r.pos != r.in.size())
        Reader::fail("corrupted: " + std::to_string(r.in.size() - r.pos) +
                     " bytes after the object");
    return object;
}

#define MODCMA_SERIALIZE_VALUE(T)                                                                  \
    template std::string save<T>(const T &);                                                       \
    template T load<T>(std::string_view);
#define MODCMA_SERIALIZE_MODULE(B)                                                                 \
    template std::string save<B>(const B &);                                                       \
    template std::shared_ptr<B> load<std::shared_ptr<B>>(std::string_view);

MODCMA_SERIALIZE_VALUE(ModularCMAES)
MODCMA_SERIALIZE_VALUE(parameters::Parameters)
MODCMA_SERIALIZE_VALUE(parameters::Settings)
MODCMA_SERIALIZE_VALUE(parameters::Modules)
MODCMA_SERIALIZE_VALUE(Population)
MODCMA_SERIALIZE_VALUE(parameters::Stats)
MODCMA_SERIALIZE_VALUE(Solution)
MODCMA_SERIALIZE_VALUE(parameters::Center)
MODCMA_SERIALIZE_VALUE(parameters::Weights)
MODCMA_SERIALIZE_VALUE(MetricCopy)
MODCMA_SERIALIZE_VALUE(restart::Criteria)
MODCMA_SERIALIZE_VALUE(selection::Strategy)
MODCMA_SERIALIZE_VALUE(repelling::TabooPoint)
MODCMA_SERIALIZE_VALUE(mapping::CoordinateMapping)
MODCMA_SERIALIZE_MODULE(sampling::Sampler)
MODCMA_SERIALIZE_MODULE(matrix_adaptation::Adaptation)
MODCMA_SERIALIZE_MODULE(mutation::Strategy)
MODCMA_SERIALIZE_MODULE(mutation::ThresholdConvergence)
MODCMA_SERIALIZE_MODULE(mutation::SequentialSelection)
MODCMA_SERIALIZE_MODULE(mutation::SigmaSampler)
MODCMA_SERIALIZE_MODULE(selection::Pairwise)
MODCMA_SERIALIZE_MODULE(selection::Elitsm)
MODCMA_SERIALIZE_MODULE(restart::Criterion)
MODCMA_SERIALIZE_MODULE(restart::Strategy)
MODCMA_SERIALIZE_MODULE(bounds::BoundCorrection)
MODCMA_SERIALIZE_MODULE(repelling::Repelling)
MODCMA_SERIALIZE_MODULE(center::Placement)
#undef MODCMA_SERIALIZE_VALUE
#undef MODCMA_SERIALIZE_MODULE
} // namespace serialization
