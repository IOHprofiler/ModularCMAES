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

#include "c_maes.hpp"
#include "es.hpp"
#include "to_string.hpp"

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
} // namespace modcma_python
