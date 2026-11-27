// Shared declarations of the nanobind bindings of the C++ ModularCMAES (modcma.c_maes.cmaescpp).
// Each source file defines the submodules of one area; module.cpp calls them in an order in which
// every base class and every type used for a default argument is bound first.
#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <typeinfo>

#include "c_maes.hpp"
#include "es.hpp"
#include "serialization.hpp"
#include "to_string.hpp"

/**
 * A DeprecationWarning for an old name of 1.x, reported at the Python line that used it (a C++
 * function has no frame of its own, so stacklevel 1 is the caller). Raises when warnings are errors.
 */
inline void warn_deprecated(const char *message)
{
    if (PyErr_WarnEx(PyExc_DeprecationWarning, message, 1) != 0)
        throw nanobind::python_error();
}

// The restart criteria of Parameters are a vector that Python code appends to
// (p.criteria.items.append(...)), so it is bound as a list-like object, not converted.
NB_MAKE_OPAQUE(restart::vCriteria)

namespace nanobind::detail
{
/**
 * A Python callable as an objective function (FunctionType). The C++ side passes the point by
 * reference (a column of the population); the generic std::function caster would hand Python a
 * view of that column, which later samples overwrite. This caster gives the function a new array
 * for every call, so a Python objective may keep the point it was given.
 */
template <> struct type_caster<FunctionType>
{
    NB_TYPE_CASTER(FunctionType,
                   const_name("collections.abc.Callable[[numpy.typing.NDArray[numpy.float64]], "
                              "float]"))

    bool from_python(handle src, uint8_t, cleanup_list *) noexcept
    {
        if (!PyCallable_Check(src.ptr()))
            return false;
        value = [f = borrow<object>(src)](Eigen::Ref<const Vector> x) -> Float {
            return cast<Float>(f(cast(Vector(x), rv_policy::move)));
        };
        return true;
    }

    static handle from_cpp(const FunctionType &, rv_policy, cleanup_list *) noexcept
    {
        return none().release();
    }
};
} // namespace nanobind::detail

namespace modcma_python
{
namespace nb = nanobind;

void define_constants(nb::module_ &m);
void define_options(nb::module_ &m);
void define_utils(nb::module_ &m);
void define_population(nb::module_ &m);
void define_samplers(nb::module_ &m);
void define_mutation(nb::module_ &m);
void define_restart_criteria(nb::module_ &m);
void define_restart_strategy(nb::module_ &m);
void define_matrix_adaptation(nb::module_ &m);
void define_center_placement(nb::module_ &m);
void define_repelling(nb::module_ &m);
void define_parameters(nb::module_ &m);
void define_bounds(nb::module_ &m);
void define_selection(nb::module_ &m);
void define_cmaes(nb::module_ &m);
void define_es(nb::module_ &m);

//! A submodule; created on first use (restart is shared by the criteria and the strategies)
inline nb::module_ submodule(nb::module_ &m, const char *name)
{
    if (nb::hasattr(m, name))
        return nb::borrow<nb::module_>(m.attr(name));
    return m.def_submodule(name);
}

//! A Python bool or a numpy bool (pybind11 accepted numpy.bool_, ConfigSpace returns it)
inline bool to_bool(nb::handle value)
{
    if (PyBool_Check(value.ptr()))
        return value.ptr() == Py_True;
    const nb::handle type = value.type();
    const auto module = nb::cast<std::string>(nb::str(nb::getattr(type, "__module__", nb::str(""))));
    const auto name = nb::cast<std::string>(nb::str(nb::getattr(type, "__name__", nb::str(""))));
    if (module == "numpy" && (name == "bool" || name == "bool_"))
        return PyObject_IsTrue(value.ptr()) == 1;
    throw nb::type_error(("expected a bool, got " + module + "." + name).c_str());
}

//! def_prop_rw(name, bool_getter(&C::member), bool_setter(&C::member)) for a bool member
template <typename C> auto bool_getter(bool C::*member)
{
    return [member](const C &c) { return c.*member; };
}

template <typename C> auto bool_setter(bool C::*member)
{
    return [member](C &c, nb::handle value) { c.*member = to_bool(value); };
}

//! A vector from any sequence of numbers (numpy arrays, lists, tuples)
inline Vector to_vector(const std::vector<Float> &values)
{
    return Eigen::Map<const Vector>(values.data(), static_cast<Eigen::Index>(values.size()));
}

inline std::optional<Vector> to_vector(const std::optional<std::vector<Float>> &values)
{
    if (!values)
        return std::nullopt;
    return to_vector(*values);
}

//! repr helper: "<Name a: 1 b: 2>"
template <typename F> std::string repr(const char *name, F &&body)
{
    std::stringstream ss;
    ss << std::boolalpha << "<" << name;
    body(ss);
    ss << ">";
    return ss.str();
}

/**
 * __getstate__ and __setstate__ of the bound class C, for pickle, copy.copy and copy.deepcopy
 * (#91). The state is the bytes of serialization::save<Saved>, with the instance's __dict__ when it
 * has attributes (a Python subclass): (bytes, dict). Saved is C, or, for a module (a polymorphic
 * class), its base class, so that the state records the module's type (and __setstate__ checks
 * it). A class that derives from a bound class gets its own,
 * so that __setstate__ creates the right type. Errors: TypeError for an object that cannot be
 * saved (a sampler defined in Python), ValueError for a state that cannot be loaded
 * (serialization::Error).
 */
template <typename C, typename Saved = C> void def_pickle()
{
    const nb::handle cls = nb::type<C>();
    nb::cpp_function_def(
        [](nb::handle self) -> nb::object {
            const C &object = nb::cast<const C &>(self);
            if constexpr (std::is_polymorphic_v<Saved>)
                if (typeid(object) != typeid(C))
                    throw nb::type_error("cannot pickle this object on its own: its C++ type has "
                                         "no Python class (pickle the optimizer instead)");
            std::string state;
            try
            {
                state = serialization::save<Saved>(object);
            }
            catch (const std::invalid_argument &e)
            {
                throw nb::type_error(e.what());
            }
            nb::bytes bytes(state.data(), state.size());
            nb::object dict = nb::getattr(self, "__dict__", nb::none());
            if (!dict.is_none() && nb::len(dict) != 0)
                return nb::make_tuple(bytes, dict);
            return std::move(bytes);
        },
        nb::scope(cls),
        nb::name("__getstate__"),
        nb::is_method(),
        "The state for pickle and copy: the bytes of the object's C++ serialization (format "
        "described in the guide's page on saving and resuming a run), with the attributes of a "
        "Python subclass.");
    nb::cpp_function_def(
        [](C &self, nb::handle state) {
            nb::object data = nb::borrow(state), dict;
            if (nb::isinstance<nb::tuple>(state))
            {
                if (nb::len(state) != 2)
                    throw nb::value_error("__setstate__: expected bytes or (bytes, dict)");
                data = state[0];
                dict = state[1];
            }
            if (!nb::isinstance<nb::bytes>(data))
                throw nb::type_error("__setstate__: the state must be bytes");
            const auto bytes = nb::borrow<nb::bytes>(data);
            const std::string_view view(bytes.c_str(), bytes.size());
            if constexpr (!std::is_polymorphic_v<Saved>)
                new (&self) C(serialization::load<C>(view));
            else
            {
                const auto loaded = serialization::load<std::shared_ptr<Saved>>(view);
                if (typeid(*loaded) != typeid(C))
                    throw nb::value_error("__setstate__: the state holds a module of another type");
                new (&self) C(static_cast<const C &>(*loaded));
            }
            if (dict.is_valid() && !dict.is_none())
                nb::find(self).attr("__dict__").attr("update")(dict);
        },
        nb::scope(cls),
        nb::name("__setstate__"),
        nb::is_method(),
        nb::arg("state"),
        "Restore the object from a state of __getstate__ (pickle and copy). ValueError when the "
        "state cannot be loaded: not a modcma state, written by a newer modcma, truncated or "
        "corrupted.");
}
} // namespace modcma_python
