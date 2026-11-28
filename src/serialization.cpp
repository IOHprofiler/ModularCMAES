// The state of the library's objects as bytes (#91); the format is described in
// serialization.hpp.
//
// Every class with state has one function here, io(a, x), that lists its members in the order
// they are written. `a` is a Writer (save) or a Reader (load), with the same methods:
// a.f64(x.sigma) writes sigma, or reads it into sigma; a.check() tests what was read (only when
// loading). So a member is written and read by the same line. A member that is not written is named
// in a "Not saved" comment: a buffer that is overwritten before it is read (scratch), or a value
// that is computed again. A new member of a class needs a line in its io() (or in the comment),
// and, when it changes what is written, a new format_version.
//
// An object is created before its members are read into it (create()): most classes with
// placeholder arguments; the few whose constructor needs part of the state (the dimension of a
// matrix adaptation, ...) write that part first (header()).
//
// Polymorphic members (the modules) are written as a tag, the name of their type, followed by
// the object; the reader creates the object of that type (the registries at the end of Access).
// Types the library does not know (a sampler or a restart criterion written in Python) cannot be
// written, except the restart criteria through External (the Python bindings pickle them).
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
// Writer and Reader: the encoding of numbers, vectors and matrices. Both have the same methods for
// the members of an object, so that one function, Access::io(), lists them for both: with a Writer,
// a.f64(x) writes x; with a Reader, it reads x.
// ------------------------------------------------------------------------------------------------
namespace detail
{
class Writer
{
  public:
    //! io() checks the state only when it is loaded
    static constexpr bool loading = false;

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

    //! A signed integer (int, Eigen::Index) as 64 bits
    template <typename I> void i64(const I v)
    {
        static_assert(std::is_integral_v<I> && std::is_signed_v<I>);
        u64(static_cast<uint64_t>(static_cast<int64_t>(v)));
    }

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

    //! Its length, then its elements (the Reader checks the length, when one is given)
    void vec(const Vector &v, size_t = 0, const char * = nullptr)
    {
        size(static_cast<size_t>(v.size()));
        doubles(v.data(), static_cast<size_t>(v.size()));
    }

    //! Its shape, then its elements in column order
    void mat(const Matrix &m, size_t = 0, size_t = 0, const char * = nullptr)
    {
        size(static_cast<size_t>(m.rows()));
        size(static_cast<size_t>(m.cols()));
        doubles(m.data(), static_cast<size_t>(m.size()));
    }

    void floats(const std::vector<Float> &values)
    {
        size(values.size());
        doubles(values.data(), values.size());
    }

    //! The length, then every element as 64 bits
    void ints(const Eigen::ArrayXi &values)
    {
        size(static_cast<size_t>(values.size()));
        for (const int v : values)
            i64(v);
    }

    //! An enumerator as a byte (the Reader checks that it is at most last)
    template <typename E> void enumeration(const E e, E, const char *)
    {
        u8(static_cast<uint8_t>(e));
    }

    //! Whether it has a value, then the value (with put)
    template <typename T, typename F> void opt(std::optional<T> &value, F &&put)
    {
        flag(value.has_value());
        if (value)
            put(*value);
    }

    //! The length, then every element (with put). blank: the element the Reader starts from
    template <typename T, typename F> void list(std::vector<T> &values, F &&put, const T & = T{})
    {
        size(values.size());
        for (auto &value : values)
            put(value);
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

    static void check(bool, const char *) {}
};

class Reader
{
  public:
    static constexpr bool loading = true;

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

    static void check(const bool condition, const char *what)
    {
        if (!condition)
            fail("corrupted (" + std::string(what) + ")");
    }

    void need(const size_t n) const
    {
        if (n > in.size() - pos)
            fail("truncated: the data ends in the middle of the object");
    }

    // --------------------------------------------------------------------------------------------
    // The values
    // --------------------------------------------------------------------------------------------
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

    // --------------------------------------------------------------------------------------------
    // The members: the methods of the Writer, reading into their argument
    // --------------------------------------------------------------------------------------------
    void u64(uint64_t &v) { v = u64(); }

    template <typename I> void i64(I &v)
    {
        static_assert(std::is_integral_v<I> && std::is_signed_v<I>);
        const int64_t x = i64();
        if constexpr (sizeof(I) < sizeof(int64_t))
            check(x >= std::numeric_limits<I>::min() && x <= std::numeric_limits<I>::max(),
                  "an integer does not fit");
        v = static_cast<I>(x);
    }

    void f64(double &v) { v = f64(); }

    void flag(bool &v) { v = flag(); }

    void size(size_t &v) { v = size(); }

    void str(std::string &s) { s = str(); }

    void vec(Vector &v)
    {
        const size_t n = count(sizeof(double));
        v.resize(static_cast<Eigen::Index>(n));
        doubles(v.data(), n);
    }

    //! A vector of the given length
    void vec(Vector &v, const size_t n, const char *name)
    {
        vec(v);
        check(static_cast<size_t>(v.size()) == n,
              std::string(name) + " has length " + std::to_string(v.size()) + ", not " +
                  std::to_string(n));
    }

    void mat(Matrix &m)
    {
        const uint64_t rows = u64(), cols = u64();
        const size_t available = (in.size() - pos) / sizeof(double);
        check(rows <= available && (rows == 0 || cols <= available / rows),
              "a matrix of " + std::to_string(rows) + " x " + std::to_string(cols) +
                  " exceeds the data");
        m.resize(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
        doubles(m.data(), static_cast<size_t>(m.size()));
    }

    //! A matrix of the given shape
    void mat(Matrix &m, const size_t rows, const size_t cols, const char *name)
    {
        mat(m);
        check(static_cast<size_t>(m.rows()) == rows && static_cast<size_t>(m.cols()) == cols,
              std::string(name) + " is " + std::to_string(m.rows()) + " x " +
                  std::to_string(m.cols()) + ", not " + std::to_string(rows) + " x " +
                  std::to_string(cols));
    }

    void floats(std::vector<Float> &values)
    {
        values.resize(count(sizeof(double)));
        doubles(values.data(), values.size());
    }

    void ints(Eigen::ArrayXi &values)
    {
        values.resize(static_cast<Eigen::Index>(count(8)));
        for (auto &v : values)
            i64(v);
    }

    //! An enumerator in [0, last]
    template <typename E> void enumeration(E &e, const E last, const char *name)
    {
        const uint8_t v = u8();
        check(v <= static_cast<uint8_t>(last),
              std::string(name) + " has the unknown value " + std::to_string(v));
        e = static_cast<E>(v);
    }

    template <typename T, typename F> void opt(std::optional<T> &value, F &&get)
    {
        if (flag())
            get(value.emplace());
        else
            value.reset();
    }

    template <typename T, typename F>
    void list(std::vector<T> &values, F &&get, const T &blank = T{})
    {
        values.assign(count(1), blank);
        for (auto &value : values)
            get(value);
    }

    void generator(std::shared_ptr<rng::Generator> &g) { g = generator(); }
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

/**
 * The name of the kind of object a state holds (its header), and whether it is a module: the base
 * class of polymorphic objects, saved with the tag of the object's own type
 */
template <typename T> struct Kind;
#define MODCMA_KIND(T, name, is_module)                                                            \
    template <> struct Kind<T>                                                                     \
    {                                                                                              \
        static constexpr const char *value = name;                                                 \
        static constexpr bool module = is_module;                                                  \
    };                                                                                             \
    template <> struct Kind<std::shared_ptr<T>> : Kind<T>                                          \
    {                                                                                              \
    };
MODCMA_KIND(ModularCMAES, "ModularCMAES", false)
MODCMA_KIND(parameters::Parameters, "Parameters", false)
MODCMA_KIND(parameters::Settings, "Settings", false)
MODCMA_KIND(parameters::Modules, "Modules", false)
MODCMA_KIND(Population, "Population", false)
MODCMA_KIND(parameters::Stats, "Stats", false)
MODCMA_KIND(Solution, "Solution", false)
MODCMA_KIND(parameters::Center, "Center", false)
MODCMA_KIND(parameters::Weights, "Weights", false)
MODCMA_KIND(MetricCopy, "MetricCopy", false)
MODCMA_KIND(restart::Criteria, "Criteria", false)
MODCMA_KIND(selection::Strategy, "selection.Strategy", false)
MODCMA_KIND(repelling::TabooPoint, "TabooPoint", false)
MODCMA_KIND(mapping::CoordinateMapping, "CoordinateMapping", false)
MODCMA_KIND(sampling::Sampler, "sampling.Sampler", true)
MODCMA_KIND(matrix_adaptation::Adaptation, "matrix_adaptation.Adaptation", true)
MODCMA_KIND(mutation::Strategy, "mutation.Strategy", true)
MODCMA_KIND(mutation::ThresholdConvergence, "mutation.ThresholdConvergence", true)
MODCMA_KIND(mutation::SequentialSelection, "mutation.SequentialSelection", true)
MODCMA_KIND(mutation::SigmaSampler, "mutation.SigmaSampler", true)
MODCMA_KIND(selection::Pairwise, "selection.Pairwise", true)
MODCMA_KIND(selection::Elitsm, "selection.Elitism", true)
MODCMA_KIND(restart::Criterion, "restart.Criterion", true)
MODCMA_KIND(restart::Strategy, "restart.Strategy", true)
MODCMA_KIND(bounds::BoundCorrection, "bounds.BoundCorrection", true)
MODCMA_KIND(repelling::Repelling, "repelling.Repelling", true)
MODCMA_KIND(center::Placement, "center.Placement", true)
#undef MODCMA_KIND
} // namespace detail

using detail::Reader;
using detail::Registry;
using detail::Tag;
using detail::Writer;

struct Access
{
    // --------------------------------------------------------------------------------------------
    // An object is saved as what its constructor needs (header(); most classes need nothing),
    // then its members (io()). It is loaded by creating it from what header() wrote (create()),
    // then reading its members into it (io()).
    // --------------------------------------------------------------------------------------------
    template <typename T> static void save_object(Writer &w, const T &object)
    {
        // io() takes the object by reference for both directions; the Writer only reads it
        T &x = const_cast<T &>(object);
        header(w, x);
        io(w, x);
    }

    template <typename T> static std::shared_ptr<T> load_object(Reader &r)
    {
        std::shared_ptr<T> x = create<T>(r);
        io(r, *x);
        return x;
    }

    //! What the constructor of T needs, written before its members (create() reads it)
    template <typename T> static void header(Writer &w, T &x)
    {
        if constexpr (std::is_base_of_v<matrix_adaptation::Adaptation, T>)
            // The dimension: the constructor sizes the buffers of the updates
            w.size(static_cast<size_t>(x.m.size()));
        else if constexpr (std::is_same_v<T, sampling::Sobol>)
        {
            // The constructor computes the points of the cache
            w.size(x.d);
            w.size(x.cache.dim);
            w.size(x.cache.n_samples);
        }
        else if constexpr (std::is_same_v<T, sampling::Orthogonal>)
        {
            // The constructor sizes the QR decomposition and the lengths from the wrapped sampler
            w.size(x.d);
            poly(w, x.sampler, "sampler");
            w.size(x.n);
        }
    }

    /**
     * An object of type T to load into: io() overwrites every member that is saved. A class whose
     * constructor needs part of the state reads it here (header()); the others are created with
     * placeholder arguments.
     */
    template <typename T> static std::shared_ptr<T> create(Reader &r)
    {
        using namespace sampling;
        if constexpr (std::is_base_of_v<matrix_adaptation::Adaptation, T>)
        {
            const size_t dim = r.size();
            Reader::check(dim >= 1, "the dimension of a matrix adaptation");
            return std::make_shared<T>(dim, Vector::Zero(static_cast<Eigen::Index>(dim)));
        }
        else if constexpr (std::is_same_v<T, Sobol>)
        {
            const size_t d = r.size();
            const size_t dim = r.size();
            const size_t n_samples = r.size();
            Reader::check(d == dim && d >= 1 && n_samples >= 1 &&
                              n_samples <= std::numeric_limits<size_t>::max() / d,
                          "the cache of a Sobol sampler");
            // The constructor computes the points for the current constants (from a generator
            // of its own, whose shuffler is overwritten); again when the state's number of points
            // differs
            auto unused = std::make_shared<rng::Generator>();
            auto s = std::make_shared<Sobol>(d, unused);
            auto &cache = s->cache;
            if (cache.n_samples != n_samples)
            {
                cache.n_samples = n_samples;
                cache.cache.assign(n_samples * d, 0.0);
                long long seed = 2;
                for (size_t i = 0; i < n_samples; i++)
                    i8_sobol(static_cast<int>(d), &seed, cache.cache.data() + i * d);
            }
            return s;
        }
        else if constexpr (std::is_same_v<T, Orthogonal>)
        {
            const size_t d = r.size();
            std::shared_ptr<Sampler> inner;
            required(r, inner, "sampler");
            Reader::check(inner->d == d, "the dimension of an orthogonal sampler");
            const size_t n = r.size();
            auto s = std::make_shared<Orthogonal>(inner, n);
            s->n = n;
            return s;
        }
        else if constexpr (std::is_same_v<T, parameters::Parameters>)
            // Parameters without modules (the constructor is private)
            return std::shared_ptr<T>(new T(parameters::Settings(1), blank_weights()));
        else if constexpr (std::is_same_v<T, ModularCMAES>)
            return std::make_shared<T>(create<parameters::Parameters>(r));
        else if constexpr (std::is_base_of_v<SampleTransformer, T> || std::is_same_v<T, Mirrored>)
            // Around a placeholder sampler
            return std::make_shared<T>(std::make_shared<Tester>(1));
        else if constexpr (std::is_same_v<T, CachedSampler>)
            // With a cache of one sample: the other constructor reserves the whole cache
            return std::make_shared<T>(std::vector<Vector>{Vector::Zero(1)});
        else if constexpr (std::is_same_v<T, Halton>)
            // Without scrambling: no random numbers are drawn
            return std::make_shared<T>(1, false, rng::global());
        else if constexpr (std::is_same_v<T, FastGaussian>)
        {
            // Seeded from a generator of its own (the state is overwritten): not from the global
            // one
            rng::Generator unused;
            return std::make_shared<T>(1, unused);
        }
        else if constexpr (std::is_base_of_v<mutation::Strategy, T>)
            return std::make_shared<T>(nullptr, nullptr, nullptr, 0.0);
        else if constexpr (std::is_base_of_v<mutation::SequentialSelection, T>)
            // Mirror::NONE keeps the factor as it is given
            return std::make_shared<T>(parameters::Mirror::NONE, 0.0);
        else if constexpr (std::is_base_of_v<mutation::SigmaSampler, T>)
            return std::make_shared<T>(1.0, rng::global());
        else if constexpr (std::is_same_v<T, parameters::Weights>)
            return std::make_shared<T>(blank_weights());
        else if constexpr (std::is_same_v<T, repelling::TabooPoint>)
            return std::make_shared<T>(blank_taboo_point());
        else if constexpr (std::is_same_v<T, mapping::CoordinateMapping>)
            return std::make_shared<T>(Indices{}, 0);
        else if constexpr (std::is_same_v<T, restart::Criteria>)
            return std::make_shared<T>(restart::vCriteria{});
        else if constexpr (std::is_same_v<T, selection::Strategy>)
            return std::make_shared<T>(parameters::Modules{});
        else if constexpr (std::is_same_v<T, Tester> || std::is_same_v<T, Gaussian> ||
                           std::is_same_v<T, Uniform> || std::is_same_v<T, bounds::COTN> ||
                           std::is_same_v<T, bounds::UniformResample> ||
                           std::is_same_v<T, restart::BIPOP> ||
                           std::is_same_v<T, parameters::Settings>)
            // A dimension (or BIPOP's lambda) of 1
            return std::make_shared<T>(1);
        else
            return std::make_shared<T>();
    }

    static parameters::Weights blank_weights()
    {
        return parameters::Weights(parameters::Settings(1), 1.0);
    }

    static repelling::TabooPoint blank_taboo_point() { return {Solution{}, 0.0}; }

    // --------------------------------------------------------------------------------------------
    // Polymorphic members: a tag, the name of the object's type ("" for a null pointer), then the
    // object (save_object); the reader creates the object of that type (load_object)
    // --------------------------------------------------------------------------------------------
    template <typename T, typename Base>
    static typename Registry<Base>::Entry entry(const char *tag)
    {
        return {std::type_index(typeid(T)),
                tag,
                [](Writer &w, const Base &x) { save_object(w, static_cast<const T &>(x)); },
                [](Reader &r) -> std::shared_ptr<Base> { return load_object<T>(r); }};
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

    //! A polymorphic member (what: its name in messages)
    template <typename A, typename Base>
    static void poly(A &a, std::shared_ptr<Base> &x, const char *what)
    {
        if constexpr (A::loading)
            x = read_poly<Base>(a, what);
        else
            write_poly<Base>(a, x.get(), what);
    }

    //! A polymorphic member that is never null: a sampler that wraps another
    template <typename A>
    static void required(A &a, std::shared_ptr<sampling::Sampler> &x, const char *what)
    {
        poly(a, x, what);
        a.check(x != nullptr, "a sampler without the sampler it wraps");
    }

    //! A member that may be null (an object, not a module): a flag, then the object
    template <typename A, typename T> static void nullable(A &a, std::shared_ptr<T> &x)
    {
        bool present = x != nullptr;
        a.flag(present);
        if constexpr (A::loading)
            x = present ? create<T>(a) : nullptr;
        if (present)
            io(a, *x);
    }

    // --------------------------------------------------------------------------------------------
    // The members of each class, in the order they are written. A member that is not written is
    // named in a "Not saved" comment.
    // --------------------------------------------------------------------------------------------
    // rng::uniform has no state
    template <typename A> static void io(A &, rng::uniform<Float> &) {}

    template <typename A> static void io(A &a, rng::normal<Float> &n)
    {
        a.f64(n.mu);
        a.f64(n.sigma);
        a.f64(n.r2);
        a.flag(n.generate);
    }

    template <typename A> static void io(A &a, rng::Shuffler &s)
    {
        for (size_t *v :
             {&s.start, &s.stop, &s.n, &s.seed, &s.offset, &s.multiplier, &s.modulus, &s.found})
            a.size(*v);
        a.check(s.stop >= s.start && s.n == s.stop - s.start && s.modulus >= s.n &&
                    (s.n == 0 || s.seed < s.n),
                "the shuffler of a cached sequence");
    }

    template <typename A> static void io(A &a, Solution &s)
    {
        a.vec(s.x);
        a.f64(s.y);
        a.size(s.t);
        a.size(s.e);
    }

    template <typename A> static void io(A &a, MetricCopy &m)
    {
        a.vec(m.d);
        a.opt(m.B, [&](Matrix &B) { a.mat(B); });
        a.check(!m.B || (m.B->rows() == m.d.size() && m.B->cols() == m.d.size()),
                "the eigenvectors of a metric");
    }

    template <typename A> static void io(A &a, parameters::Center &c)
    {
        io(a, static_cast<Solution &>(c));
        a.f64(c.sigma);
        io(a, c.metric);
        a.list(c.reasons, [&](std::string &s) { a.str(s); });
    }

    // --------------------------------------------------------------------------------------------
    // Settings
    // --------------------------------------------------------------------------------------------
    template <typename A> static void io(A &a, parameters::Modules &m)
    {
        using namespace parameters;
        for (bool *b : {&m.elitist,
                        &m.active,
                        &m.orthogonal,
                        &m.sequential_selection,
                        &m.threshold_convergence,
                        &m.sample_sigma})
            a.flag(*b);
        // Each with its last enumerator: a value appended to an enum needs it here (else the
        // states with it do not load; test_c_pickle's test_every_module_option pickles every
        // option)
        a.enumeration(m.weights, RecombinationWeights::UTILITY, "RecombinationWeights");
        a.enumeration(m.sampler, BaseSampler::TESTER, "BaseSampler");
        a.enumeration(m.mirrored, Mirror::PAIRWISE, "Mirror");
        a.enumeration(m.ssa, StepSizeAdaptation::TPA_INPLACE, "StepSizeAdaptation");
        a.enumeration(m.bound_correction, CorrectionMethod::RESAMPLE, "CorrectionMethod");
        a.enumeration(m.restart_strategy, RestartStrategyType::BIPOP, "RestartStrategyType");
        a.enumeration(
            m.matrix_adaptation, MatrixAdaptationType::NATURAL_GRADIENT, "MatrixAdaptationType");
        a.enumeration(m.center_placement, CenterPlacement::MAXIMIN_TABOO, "CenterPlacement");
        a.enumeration(
            m.sample_transformation, SampleTranformerType::DOUBLE_WEIBULL, "SampleTranformerType");
        a.enumeration(m.repelling_type, RepellingType::ADAPTIVE, "RepellingType");
        a.enumeration(m.repelling_metric, RepellingMetric::ARCHIVED, "RepellingMetric");
    }

    template <typename A> static void opt_float(A &a, std::optional<Float> &v)
    {
        a.opt(v, [&](Float &x) { a.f64(x); });
    }

    template <typename A> static void io(A &a, parameters::Settings &s)
    {
        a.size(s.dim);
        a.check(s.dim >= 1, "dim is 0");
        io(a, s.modules);
        opt_float(a, s.target);
        a.opt(s.max_generations, [&](size_t &v) { a.size(v); });
        a.size(s.budget);
        a.f64(s.sigma0);
        a.size(s.lambda0);
        a.size(s.mu0);
        a.opt(s.x0, [&](Vector &v) { a.vec(v, s.dim, "x0"); });
        a.vec(s.lb, s.dim, "lb");
        a.vec(s.ub, s.dim, "ub");
        a.ints(s.integer_variables);
        a.vec(s.db, s.dim, "db");
        a.vec(s.center, s.dim, "center");
        a.f64(s.diameter);
        a.f64(s.volume);
        a.flag(s.has_bounds);
        for (auto *rate : {&s.cs, &s.cc, &s.cmu, &s.c1, &s.damps, &s.acov})
            opt_float(a, *rate);
        a.flag(s.verbose);
        a.flag(s.one_plus_one);
        a.opt(s.seed, [&](int &v) { a.i64(v); });
        a.flag(s.fast_sampling);
        a.opt(s.C0, [&](Matrix &m) { a.mat(m, s.dim, s.dim, "C0"); });
        a.flag(s.mu0_given);
    }

    // --------------------------------------------------------------------------------------------
    // Statistics, population, weights
    // --------------------------------------------------------------------------------------------
    template <typename A> static void io(A &a, parameters::Stats &s)
    {
        a.size(s.t);
        a.size(s.t_run);
        a.size(s.evaluations);
        a.f64(s.current_avg);
        a.list(s.solutions, [&](Solution &x) { io(a, x); });
        a.list(s.centers, [&](parameters::Center &c) { io(a, c); });
        io(a, s.current_best);
        io(a, s.global_best);
        a.flag(s.has_improved);
        a.f64(s.success_ratio);
        a.f64(s.cp);
        a.size(s.last_update);
        a.size(s.n_updates);
    }

    // Not saved: the buffers of sort() and reorder() (order, scratch, scratch_vector). The
    // matrices are saved whole, with the columns past n (their capacity; the points dropped by
    // the elitist selection, n_pool).
    template <typename A> static void io(A &a, Population &p)
    {
        for (Matrix *m : {&p.X, &p.X_transformed, &p.Z, &p.Y, &p.S})
            a.mat(*m);
        a.vec(p.f);
        a.vec(p.t);
        a.size(p.d);
        a.size(p.n);
        a.size(p.n_pool);
        a.size(p.n_ranked);
        if constexpr (A::loading)
        {
            const auto rows = p.X.rows(), cols = p.X.cols();
            for (const Matrix *m : {&p.X_transformed, &p.Z, &p.Y, &p.S})
                a.check(m->rows() == rows && m->cols() == cols,
                        "the matrices of a population differ in shape");
            a.check(p.f.size() == cols && p.t.size() == cols && p.n <= p.capacity() &&
                        p.n_pool <= p.capacity() && static_cast<size_t>(rows) == p.d,
                    "the sizes of a population");
        }
    }

    template <typename A> static void io(A &a, parameters::Weights &x)
    {
        for (Vector *v : {&x.weights, &x.positive, &x.negative, &x.negative_reference})
            a.vec(*v);
        for (Float *f : {&x.mueff, &x.mueff_neg, &x.mueff_mean})
            a.f64(*f);
        a.flag(x.utility);
        a.flag(x.pairwise);
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
            a.f64(*f);
    }

    // --------------------------------------------------------------------------------------------
    // Samplers
    // --------------------------------------------------------------------------------------------
    template <typename A> static void io(A &a, sampling::Tester &s)
    {
        a.size(s.d);
        a.size(s.i);
    }

    //! Gaussian and Uniform
    template <typename A, typename D> static void io(A &a, sampling::Random<D> &s)
    {
        a.size(s.d);
        io(a, s.dist);
        a.generator(s.generator);
    }

    template <typename A> static void io(A &a, sampling::FastGaussian &s)
    {
        a.size(s.d);
        for (auto &word : s.generator.s)
            a.u64(word);
    }

    template <typename A> static void io(A &a, sampling::Mirrored &s)
    {
        a.size(s.d);
        required(a, s.sampler, "sampler");
        a.vec(s.previous);
        a.flag(s.mirror);
    }

    // Not saved: the QR decomposition, the lengths and the workspace of next_block(), which
    // computes them for every block. In header(): d, the wrapped sampler, n
    template <typename A> static void io(A &a, sampling::Orthogonal &s)
    {
        a.size(s.remaining);
        a.i64(s.block_size);
        a.i64(s.current);
        a.mat(s.samples, s.d, s.d, "the block of an orthogonal sampler");
        a.check(s.block_size >= 0 && static_cast<size_t>(s.block_size) <= s.d && s.current >= 0,
                "the block of an orthogonal sampler");
    }

    template <typename A> static void io(A &a, sampling::Halton &s)
    {
        a.size(s.d);
        a.i64(s.index_);
        a.flag(s.scramble_);
        a.list(s.primes_, [&](int &p) { a.i64(p); });
        a.list(s.permutations_, [&](std::vector<std::vector<int>> &per_prime) {
            a.list(per_prime, [&](std::vector<int> &permutation) {
                a.list(permutation, [&](int &v) { a.i64(v); });
            });
        });
        if constexpr (A::loading)
        {
            a.check(s.primes_.size() == s.d && s.permutations_.size() == s.d,
                    "the primes of a Halton sampler");
            for (size_t j = 0; j < s.d; ++j)
            {
                const int base = s.primes_[j];
                a.check(base >= 2, "the primes of a Halton sampler");
                for (const auto &permutation : s.permutations_[j])
                {
                    a.check(permutation.size() == static_cast<size_t>(base),
                            "a permutation of a Halton sampler");
                    for (const int v : permutation)
                        a.check(v >= 0 && v < base, "a permutation of a Halton sampler");
                }
            }
        }
    }

    // Not saved: the Sobol points of the cache, which are computed again (the sampler draws them
    // in the order of the shuffler). In header(): d and the size of the cache
    template <typename A> static void io(A &a, sampling::Sobol &s)
    {
        io(a, s.cache.shuffler);
        a.check(s.cache.shuffler.start + s.cache.shuffler.n <= s.cache.n_samples,
                "the shuffler of a Sobol sampler");
    }

    template <typename A> static void io(A &a, sampling::CachedSampler &s)
    {
        a.size(s.d);
        required(a, s.sampler, "sampler");
        a.list(s.cache, [&](Vector &v) { a.vec(v, s.d, "a cached sample"); });
        a.size(s.index);
        a.size(s.n_samples);
        a.check(s.n_samples >= 1 && s.cache.size() == std::min(s.index, s.n_samples),
                "the cache of a cached sampler");
    }

    // Identity, UniformScaler, Laplace, Cauchy, DoubleWeibull; Logistic (not saved: scratch, a
    // buffer of transform_inplace())
    template <typename A> static void io(A &a, sampling::SampleTransformer &s)
    {
        a.size(s.d);
        required(a, s.sampler, "sampler");
        opt_float(a, s.length);
    }

    template <typename A> static void io(A &a, sampling::GaussianTransformer &s)
    {
        io(a, static_cast<sampling::SampleTransformer &>(s));
        // The Box-Muller state for odd dimensions (constants::use_box_muller)
        a.vec(s.u_extra);
        a.i64(s.n_extra_used);
        a.flag(s.needs_new_sample);
        a.f64(s.extra_sample);
        a.check(s.n_extra_used >= 0, "the Box-Muller state of a Gaussian transformer");
    }

    // --------------------------------------------------------------------------------------------
    // Matrix adaptation. In header(): the dimension
    // --------------------------------------------------------------------------------------------
    // Not saved: the buffers of the updates (scratch_vector, weights_scratch, positive_scratch,
    // square_scratch, batch_scratch, batch_scratch2, steps_scratch, steps_scratch2), written
    // before they are read
    template <typename A> static void io(A &a, matrix_adaptation::Adaptation &x)
    {
        // The dimension of header(): the constructor sized m
        const size_t d = static_cast<size_t>(x.m.size());
        a.vec(x.m, d, "the mean");
        a.vec(x.m_old, d, "the previous mean");
        a.vec(x.dm, d, "the mean shift");
        a.vec(x.ps, d, "the path ps");
        a.vec(x.dz);
        a.vec(x.coordinate_wise_variances, d, "the variances");
        a.f64(x.dd);
        a.size(x.factor_version);
        a.vec(x.metric_d);
        a.mat(x.metric_B);
        a.size(x.metric_version);
    }

    template <typename A> static void io(A &a, matrix_adaptation::None &x)
    {
        io(a, static_cast<matrix_adaptation::Adaptation &>(x));
        a.vec(x.ones, static_cast<size_t>(x.m.size()), "the axes of NONE");
    }

    // COVARIANCE, COVARIANCE_NO_EIGV. Not saved: the eigensolver and the LAPACK workspace of the
    // decomposition
    template <typename A> static void io(A &a, matrix_adaptation::CovarianceAdaptation &x)
    {
        io(a, static_cast<matrix_adaptation::Adaptation &>(x));
        const size_t d = static_cast<size_t>(x.m.size());
        a.vec(x.pc, d, "the path pc");
        a.vec(x.d, d, "the axes D");
        a.mat(x.B, d, d, "B");
        a.mat(x.C, d, d, "C");
        a.mat(x.A, d, d, "A");
        a.flag(x.hs);
    }

    template <typename A> static void io(A &a, matrix_adaptation::SeparableAdaptation &x)
    {
        io(a, static_cast<matrix_adaptation::Adaptation &>(x));
        const size_t d = static_cast<size_t>(x.m.size());
        a.vec(x.pc, d, "the path pc");
        a.vec(x.d, d, "the axes D");
        a.vec(x.c, d, "the variances c");
        a.flag(x.hs);
    }

    // Not saved: the buffers of the Woodbury update (wood_*); the exact recomputation of M_inv
    // counts the generations of the run, stats.t_run
    template <typename A> static void io(A &a, matrix_adaptation::MatrixAdaptation &x)
    {
        io(a, static_cast<matrix_adaptation::Adaptation &>(x));
        const size_t d = static_cast<size_t>(x.m.size());
        a.mat(x.M, d, d, "M");
        a.mat(x.M_inv, d, d, "M_inv");
        a.flag(x.hs);
        a.size(x.inverse_fallbacks);
        a.flag(x.outdated_M_inv);
    }

    template <typename A> static void io(A &a, matrix_adaptation::CholeskyAdaptation &x)
    {
        io(a, static_cast<matrix_adaptation::Adaptation &>(x));
        const size_t d = static_cast<size_t>(x.m.size());
        a.mat(x.A, d, d, "A");
        a.vec(x.pc, d, "the path pc");
        a.flag(x.hs);
        a.size(x.skipped_downdates);
    }

    template <typename A> static void io(A &a, matrix_adaptation::SelfAdaptation &x)
    {
        io(a, static_cast<matrix_adaptation::Adaptation &>(x));
        const size_t d = static_cast<size_t>(x.m.size());
        a.mat(x.A, d, d, "A");
        a.mat(x.C, d, d, "C");
    }

    template <typename A> static void io(A &a, matrix_adaptation::NaturalGradientAdaptation &x)
    {
        io(a, static_cast<matrix_adaptation::Adaptation &>(x));
        const size_t d = static_cast<size_t>(x.m.size());
        a.mat(x.A, d, d, "A");
        a.mat(x.A_inv, d, d, "A_inv");
        a.f64(x.sigma_g);
        a.mat(x.gradient_samples);
        a.vec(x.gradient_weights);
        a.f64(x.gradient_shift);
        a.mat(x.gradient_matrix);
        a.flag(x.outdated_A_inv);
        a.check(static_cast<size_t>(x.gradient_samples.rows()) == d &&
                    x.gradient_weights.size() == x.gradient_samples.cols(),
                "the natural gradient");
    }

    // --------------------------------------------------------------------------------------------
    // Mutation (step-size adaptation)
    // --------------------------------------------------------------------------------------------
    template <typename A> static void io(A &a, mutation::ThresholdConvergence &t)
    {
        a.f64(t.init_threshold);
        a.f64(t.decay_factor);
    }

    template <typename A> static void io(A &a, mutation::SequentialSelection &s)
    {
        a.f64(s.seq_cutoff_factor);
    }

    template <typename A> static void io(A &a, mutation::SigmaSampler &s)
    {
        // Its sampler is a member, a GaussianTransformer of a one-dimensional sampler
        io(a, s.sampler);
    }

    // CSA, MXNES, LPXNES, SR and the base class
    template <typename A> static void io(A &a, mutation::Strategy &s)
    {
        poly(a, s.tc, "threshold convergence");
        poly(a, s.sq, "sequential selection");
        poly(a, s.ss, "sigma sampler");
        a.check(s.tc && s.sq && s.ss, "a step-size adaptation without its parts");
        a.f64(s.sigma);
        a.f64(s.s);
        a.f64(s.generation_threshold);
        a.flag(s.partner_sampled);
    }

    template <typename A> static void io(A &a, mutation::TPA &s)
    {
        io(a, static_cast<mutation::Strategy &>(s));
        for (Float *f : {&s.a_tpa, &s.b_tpa, &s.rank_tpa, &s.sigma_old})
            a.f64(*f);
    }

    template <typename A> static void io(A &a, mutation::MSR &s)
    {
        io(a, static_cast<mutation::Strategy &>(s));
        a.f64(s.quantile);
    }

    // Not saved: combined, idx and ranks, the buffers of adapt()
    template <typename A> static void io(A &a, mutation::PSR &s)
    {
        io(a, static_cast<mutation::Strategy &>(s));
        a.f64(s.success_ratio);
    }

    // Not saved: order and weights, the buffers of on_population_complete()
    template <typename A> static void io(A &a, mutation::XNES &s)
    {
        io(a, static_cast<mutation::Strategy &>(s));
        opt_float(a, s.offspring_statistic);
    }

    template <typename A> static void io(A &a, mutation::SA &s)
    {
        io(a, static_cast<mutation::Strategy &>(s));
        a.f64(s.mean_sigma);
    }

    template <typename A> static void io(A &a, mutation::TPAInPlace &s)
    {
        io(a, static_cast<mutation::Strategy &>(s));
        a.f64(s.rank_difference);
        a.flag(s.test_points);
        a.vec(s.direction);
        a.flag(s.pending);
    }

    // --------------------------------------------------------------------------------------------
    // Selection, center placement: no state but their types
    // --------------------------------------------------------------------------------------------
    // Not saved: order and loser, the buffers of Pairwise::sort()
    template <typename A> static void io(A &, selection::Pairwise &) {}
    template <typename A> static void io(A &, selection::Elitsm &) {}
    template <typename A> static void io(A &, center::Placement &) {}

    template <typename A> static void io(A &a, selection::Strategy &s)
    {
        poly(a, s.pairwise, "pairwise selection");
        poly(a, s.elitsm, "elitism");
    }

    // --------------------------------------------------------------------------------------------
    // Restart criteria and strategies
    // --------------------------------------------------------------------------------------------
    // MinSigma, UnableToAdapt, ConditionC, NoEffectAxis, NoEffectCoord, TooMuchRepelling
    template <typename A> static void io(A &a, restart::Criterion &c)
    {
        a.flag(c.met);
        a.str(c.name);
        a.size(c.last_restart);
        a.flag(c.was_reset);
    }

    template <typename A> static void io(A &a, restart::ExceededMaxIter &c)
    {
        io(a, static_cast<restart::Criterion &>(c));
        a.size(c.max_iter);
    }

    template <typename A> static void io(A &a, restart::NoImprovement &c)
    {
        io(a, static_cast<restart::Criterion &>(c));
        a.size(c.n_bin);
        a.floats(c.best_fitnesses);
    }

    template <typename A> static void io(A &a, restart::MaxSigma &c)
    {
        io(a, static_cast<restart::Criterion &>(c));
        a.f64(c.start);
    }

    template <typename A> static void io(A &a, restart::FlatFitness &c)
    {
        io(a, static_cast<restart::Criterion &>(c));
        a.size(c.max_flat_fitness);
        a.size(c.flat_fitness_index);
        a.ints(c.flat_fitnesses);
    }

    template <typename A> static void io(A &a, restart::TolX &c)
    {
        io(a, static_cast<restart::Criterion &>(c));
        a.vec(c.tolx_vector);
        a.f64(c.sigma_start);
    }

    template <typename A> static void io(A &a, restart::MaxDSigma &c)
    {
        io(a, static_cast<restart::Criterion &>(c));
        a.f64(c.sigma_start);
    }

    template <typename A> static void io(A &a, restart::MinDSigma &c)
    {
        io(a, static_cast<restart::Criterion &>(c));
        a.f64(c.sigma_start);
    }

    // The sequence and its split; the two halves (multisets) are built again from them
    template <typename A> static void io(A &a, restart::SplitMedian &m)
    {
        a.floats(m.values_);
        a.size(m.split_);
        if constexpr (A::loading)
        {
            a.check(m.split_ <= m.values_.size(), "the split of a running median");
            const size_t split = m.split_;
            m.assign(std::vector<Float>(m.values_));
            m.set_split(split);
        }
    }

    template <typename A> static void io(A &a, restart::Stagnation &c)
    {
        io(a, static_cast<restart::Criterion &>(c));
        a.size(c.n_stagnation);
        io(a, c.median_fitnesses);
        io(a, c.best_fitnesses);
    }

    template <typename A> static void io(A &a, restart::Criteria &c)
    {
        a.list(c.items,
               [&](std::shared_ptr<restart::Criterion> &x) { poly(a, x, "restart criterion"); });
    }

    template <typename A> static void io(A &a, restart::Strategy &s) { a.size(s.max_iter); }

    template <typename A> static void io(A &a, restart::IPOP &s)
    {
        io(a, static_cast<restart::Strategy &>(s));
        a.f64(s.ipop_factor);
        a.size(s.max_lambda_factor);
    }

    template <typename A> static void io(A &a, restart::BIPOP &s)
    {
        io(a, static_cast<restart::Strategy &>(s));
        for (size_t *v : {&s.lambda_init,
                          &s.max_lambda_factor,
                          &s.lambda_large,
                          &s.lambda_small,
                          &s.used_small,
                          &s.used_large,
                          &s.used_budget})
            a.size(*v);
        a.flag(s.large_run);
    }

    // --------------------------------------------------------------------------------------------
    // Bound correction
    // --------------------------------------------------------------------------------------------
    // NONE, RESAMPLE, MIRROR, SATURATE, TOROIDAL
    template <typename A> static void io(A &a, bounds::BoundCorrection &b)
    {
        a.size(b.n_out_of_bounds);
    }

    template <typename A> static void io(A &a, bounds::COTN &b)
    {
        io(a, static_cast<bounds::BoundCorrection &>(b));
        io(a, b.sampler);
    }

    template <typename A> static void io(A &a, bounds::UniformResample &b)
    {
        io(a, static_cast<bounds::BoundCorrection &>(b));
        io(a, b.sampler);
    }

    // --------------------------------------------------------------------------------------------
    // Repelling
    // --------------------------------------------------------------------------------------------
    template <typename A> static void io(A &a, repelling::TabooPoint &t)
    {
        io(a, t.solution);
        a.f64(t.radius);
        a.f64(t.shrinkage);
        a.i64(t.n_rep);
        for (Float *f : {&t.criticality, &t.min_radius, &t.max_radius, &t.duplicate_evaluations})
            a.f64(*f);
        a.i64(t.rejected_count);
        a.i64(t.checked_count);
        a.f64(t.last_rejection_rate);
        a.mat(t.metric);
    }

    // NONE, COVERAGE
    template <typename A> static void io(A &a, repelling::Repelling &x)
    {
        a.list(x.archive, [&](repelling::TabooPoint &t) { io(a, t); }, blank_taboo_point());
        a.i64(x.attempts);
        a.f64(x.coverage);
        // The archive update after a restart, between its hill-valley points
        auto &q = x.query;
        a.flag(q.active);
        a.opt(q.candidate, [&](Solution &s) { io(a, s); });
        a.size(q.j);
        a.size(q.k);
        a.flag(q.accept);
        a.vec(q.x);
        a.check(!q.active || (q.candidate && q.j < x.archive.size()),
                "the archive update of repelling");
    }

    template <typename A> static void io(A &a, repelling::AdaptiveRepelling &x)
    {
        io(a, static_cast<repelling::Repelling &>(x));
        for (Float *f : {&x.initial_radius_factor,
                         &x.min_radius_factor,
                         &x.max_radius_factor,
                         &x.shrink_factor,
                         &x.target_rejection_rate,
                         &x.max_rejection_rate})
            a.f64(*f);
        // A size since #192 (an int before): the same bytes for every count
        a.size(x.last_restart_evaluations);
        for (Float *f : {&x.average_restart_evaluations,
                         &x.restart_eval_alpha,
                         &x.grow_eta,
                         &x.max_waste_ratio,
                         &x.local_target_rejection_rate,
                         &x.local_max_rejection_rate})
            a.f64(*f);
        a.i64(x.min_checks_before_local_shrink);
        a.f64(x.restart_evals);
    }

    // --------------------------------------------------------------------------------------------
    // Coordinate mapping
    // --------------------------------------------------------------------------------------------
    template <typename A> static void io(A &a, mapping::CoordinateMapping &m)
    {
        a.list(m.integer_idx, [&](Eigen::Index &i) {
            a.i64(i);
            a.check(i >= 0, "an integer variable");
        });
    }

    // --------------------------------------------------------------------------------------------
    // Parameters and ModularCMAES
    // --------------------------------------------------------------------------------------------
    template <typename A> static void io(A &a, parameters::Parameters &p)
    {
        // The generator first: the modules refer to it
        a.generator(p.generator);
        a.flag(p.successful_adaptation);
        a.size(p.lambda);
        a.size(p.mu);
        io(a, p.settings);
        io(a, p.stats);
        poly(a, p.sampler, "sampler");
        io(a, p.weights);
        io(a, p.pop);
        io(a, p.old_pop);
        io(a, p.criteria);
        poly(a, p.adaptation, "matrix adaptation");
        poly(a, p.mutation, "step-size adaptation");
        nullable(a, p.selection);
        poly(a, p.restart_strategy, "restart strategy");
        poly(a, p.bounds, "bound correction");
        poly(a, p.repelling, "repelling");
        poly(a, p.center_placement, "center placement");
        nullable(a, p.coordinate_mapping);
        a.f64(p.restart_sigma);

        if constexpr (A::loading)
        {
            const size_t dim = p.settings.dim;
            a.check(!p.adaptation || static_cast<size_t>(p.adaptation->m.size()) == dim,
                    "the dimension of the matrix adaptation");
            a.check(!p.sampler || p.sampler->d == dim, "the dimension of the sampler");
            a.check(p.pop.d == dim && p.old_pop.d == dim, "the dimension of the populations");
            if (p.coordinate_mapping)
                for (const auto i : p.coordinate_mapping->integer_idx)
                    a.check(static_cast<size_t>(i) < dim, "an integer variable");
        }
    }

    // Not saved: values, the buffer of the objective values in step() and mutate()
    template <typename A> static void io(A &a, ModularCMAES &es)
    {
        if (!es.p)
            throw std::invalid_argument("cannot save an optimizer without parameters");
        io(a, *es.p);
        a.enumeration(es.next, ModularCMAES::Step::UPDATE, "the step of the generation");
        a.mat(es.batch);
        a.i64(es.n_pending);
        a.list(es.requests, [&](ModularCMAES::Request &q) {
            a.enumeration(q.source, ModularCMAES::Source::HILL_VALLEY, "a request");
            a.i64(q.begin);
            a.i64(q.size);
            a.i64(q.index);
        });
        a.i64(es.next_individual);
        a.flag(es.ended);
        a.flag(es.steps_skipped);
        a.list(es.end_reasons, [&](std::string &s) { a.str(s); });
        a.flag(es.gradient_open);

        if constexpr (A::loading)
        {
            a.check(es.n_pending >= 0 && es.n_pending <= es.batch.cols() &&
                        (es.n_pending == 0 ||
                         static_cast<size_t>(es.batch.rows()) == es.p->settings.dim),
                    "the points of ask()");
            Eigen::Index end = 0;
            for (const auto &q : es.requests)
            {
                a.check(q.begin == end && q.size >= 1, "the requests of ask()");
                end += q.size;
            }
            a.check(end == es.n_pending, "the requests of ask()");
            a.check(!es.gradient_open || !es.requests.empty(), "the gradient injection");
        }
    }

    // --------------------------------------------------------------------------------------------
    // The types of each polymorphic base, with their tags: a new class of a module gets an entry
    // --------------------------------------------------------------------------------------------
    static const Registry<sampling::Sampler> &registry(Tag<sampling::Sampler>)
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

    static const Registry<matrix_adaptation::Adaptation> &
    registry(Tag<matrix_adaptation::Adaptation>)
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

    static const Registry<mutation::Strategy> &registry(Tag<mutation::Strategy>)
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

    static const Registry<mutation::ThresholdConvergence> &
    registry(Tag<mutation::ThresholdConvergence>)
    {
        using namespace mutation;
        static const Registry<ThresholdConvergence> r{{
            entry<ThresholdConvergence, ThresholdConvergence>("ThresholdConvergence"),
            entry<NoThresholdConvergence, ThresholdConvergence>("NoThresholdConvergence"),
        }};
        return r;
    }

    static const Registry<mutation::SequentialSelection> &
    registry(Tag<mutation::SequentialSelection>)
    {
        using namespace mutation;
        static const Registry<SequentialSelection> r{{
            entry<SequentialSelection, SequentialSelection>("SequentialSelection"),
            entry<NoSequentialSelection, SequentialSelection>("NoSequentialSelection"),
        }};
        return r;
    }

    static const Registry<mutation::SigmaSampler> &registry(Tag<mutation::SigmaSampler>)
    {
        using namespace mutation;
        static const Registry<SigmaSampler> r{{
            entry<SigmaSampler, SigmaSampler>("SigmaSampler"),
            entry<NoSigmaSampler, SigmaSampler>("NoSigmaSampler"),
        }};
        return r;
    }

    static const Registry<selection::Pairwise> &registry(Tag<selection::Pairwise>)
    {
        using namespace selection;
        static const Registry<Pairwise> r{{
            entry<Pairwise, Pairwise>("Pairwise"),
            entry<NoPairwise, Pairwise>("NoPairwise"),
        }};
        return r;
    }

    static const Registry<selection::Elitsm> &registry(Tag<selection::Elitsm>)
    {
        using namespace selection;
        static const Registry<Elitsm> r{{
            entry<Elitsm, Elitsm>("Elitism"),
            entry<NoElitsm, Elitsm>("NoElitism"),
        }};
        return r;
    }

    static const Registry<restart::Criterion> &registry(Tag<restart::Criterion>)
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

    static const Registry<restart::Strategy> &registry(Tag<restart::Strategy>)
    {
        using namespace restart;
        static const Registry<Strategy> r{{
            entry<Strategy, Strategy>("Strategy"),
            entry<IPOP, Strategy>("IPOP"),
            entry<BIPOP, Strategy>("BIPOP"),
        }};
        return r;
    }

    static const Registry<bounds::BoundCorrection> &registry(Tag<bounds::BoundCorrection>)
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

    static const Registry<repelling::Repelling> &registry(Tag<repelling::Repelling>)
    {
        using namespace repelling;
        static const Registry<Repelling> r{{
            entry<NoRepelling, Repelling>("NoRepelling"),
            entry<CoverageRepelling, Repelling>("CoverageRepelling"),
            entry<AdaptiveRepelling, Repelling>("AdaptiveRepelling"),
        }};
        return r;
    }

    static const Registry<center::Placement> &registry(Tag<center::Placement>)
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
};

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
    if constexpr (detail::Kind<T>::module)
        Access::write_poly<T>(w, &object, detail::Kind<T>::value);
    else
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
    T object = [&] {
        if constexpr (detail::Kind<T>::module)
        {
            using Base = typename T::element_type;
            auto module = Access::read_poly<Base>(r, detail::Kind<Base>::value);
            Reader::check(module != nullptr, "an empty object");
            return module;
        }
        else
            return std::move(*Access::load_object<T>(r));
    }();
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
