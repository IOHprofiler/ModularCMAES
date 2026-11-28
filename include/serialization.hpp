#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "c_maes.hpp"

/**
 * Saving and resuming a run (#91): the state of an optimizer, or of one of its objects, as bytes,
 * from which it continues bit for bit.
 *
 *     const std::string state = serialization::save(es);           // ModularCMAES es
 *     ModularCMAES resumed = serialization::load<ModularCMAES>(state);
 *
 * The state is everything that changes during a run: the settings, the random generator (also
 * the positions of the Sobol and Halton sequences, the cache of CachedSampler, the second number
 * of a Box-Muller pair, the xoshiro state of fast_sampling), the populations, the matrix
 * adaptation (with its cached metric and inverse factors), the step-size adaptation, the restart
 * criteria and strategy, the repelling archive, the statistics, and the generation in progress
 * (the points of ask() that wait for tell(), and the step that runs next). A run that is saved and
 * loaded, at any point (between ask() and tell() too), continues with the same numbers as the
 * run that was not interrupted, on the same platform (results are bit-exact only with the same
 * compiler and library, as everywhere in modcma).
 *
 * Not in the state:
 *  - the objective function (and the gradient function): pass them again;
 *  - the process-wide settings: the namespace constants (cache_samples, use_box_muller, ...) and
 *    the static tolerances of the restart criteria; a run continues with the values of the process
 *    that loads it;
 *  - the global generator (rng::set_seed): an object that draws from it still does after loading,
 *    from its state in the loading process;
 *  - the points of the Sobol sampler, which are computed again when it is loaded;
 *  - buffers that are overwritten before they are read (documented in serialization.cpp).
 *
 * The format is binary and versioned: a magic number, format_version, the kind of object, the
 * object, and a CRC-32 checksum. Numbers are little-endian, floating point numbers are stored
 * with all their bits; the state of the Mersenne Twister is stored in the representation of the
 * C++ standard (the last 624 words), so a state can be loaded with any C++ standard library.
 * load() raises Error for bytes that are not a modcma state, a newer format, a truncated or
 * corrupted state, or a state of another kind of object. Like pickle, it is meant for states
 * that you wrote yourself: a crafted state that passes the checksum may describe an inconsistent
 * object.
 */
namespace serialization
{
/**
 * The version of the format that save() writes. A change of what is written increases it; load()
 * then reads the older versions it still supports, and raises Error for a newer version.
 */
inline constexpr uint32_t format_version = 1;

//! A state that cannot be loaded: not a modcma state, a newer format, truncated or corrupted, or
//! another kind of object (a std::invalid_argument: ValueError in Python)
struct Error : std::invalid_argument
{
    using std::invalid_argument::invalid_argument;
};

/**
 * The state of an object as bytes. T is one of:
 *  - ModularCMAES, parameters::Parameters, parameters::Settings, parameters::Modules, Population,
 *    parameters::Stats, Solution, parameters::Center, parameters::Weights, MetricCopy,
 *    restart::Criteria, selection::Strategy, repelling::TabooPoint, mapping::CoordinateMapping;
 *  - the base class of a module, which saves the module's own type: sampling::Sampler,
 *    matrix_adaptation::Adaptation, mutation::Strategy, mutation::ThresholdConvergence,
 *    mutation::SequentialSelection, mutation::SigmaSampler, selection::Pairwise,
 *    selection::Elitsm, restart::Criterion, restart::Strategy, bounds::BoundCorrection,
 *    repelling::Repelling, center::Placement.
 * std::invalid_argument when the object holds something that cannot be saved: a sampler or a
 * restart criterion defined in Python (a criterion can, through the Python bindings: External).
 * The generators of an object are saved once each: objects that share one still do after load().
 */
template <typename T> std::string save(const T &object);

/**
 * The object of a state from save<T>: load<T>(state) for the types of save(), and
 * load<std::shared_ptr<Base>>(state) for a module saved as its base class Base (the object has
 * the module's own type). Error when the state cannot be loaded.
 */
template <typename T> T load(std::string_view state);

/**
 * How to save the objects that the library cannot save itself: the restart criteria defined in
 * Python (subclasses of restart::Criterion). The Python bindings set these functions (to pickle);
 * without them, save() raises std::invalid_argument for such a criterion.
 */
struct External
{
    //! The bytes of a criterion of a type that the library does not know
    std::function<std::string(const restart::Criterion &)> save_criterion;
    //! The criterion of those bytes
    std::function<std::shared_ptr<restart::Criterion>(std::string_view)> load_criterion;
};

//! The functions for external objects (set once, before any save or load)
External &external();

/**
 * The state of the Mersenne Twister as the 624 words of the C++ standard's textual
 * representation (the last 624 words it generated, before tempering), whatever the standard
 * library: computed from the generator's next 624 outputs. Two generators with the same words
 * generate the same numbers.
 */
std::array<uint32_t, 624> generator_state(const rng::Generator &generator);

//! Set a generator to the state of generator_state(); Error when this standard library does not
//! read it back
void set_generator_state(rng::Generator &generator, const std::array<uint32_t, 624> &state);
} // namespace serialization
