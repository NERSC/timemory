// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#if !defined(TIMEMORY_PYCOMPONENTS_SOURCE)
#    define TIMEMORY_PYCOMPONENTS_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/timemory.hpp"

//======================================================================================//
//
namespace pyinternal
{
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
using pytuple_t = tim::lightweight_tuple<T>;
//
//--------------------------------------------------------------------------------------//
//
static inline std::string
get_class_name(std::string id)
{
    static const std::set<char> delim{
        '_',
        '-',
    };

    if(id.empty())
        return std::string{};

    id = tim::settings::tolower(id);

    // capitalize after every delimiter
    for(size_t i = 0; i < id.size(); ++i)
    {
        if(i == 0)
            id.at(i) = toupper(id.at(i));
        else
        {
            if(delim.find(id.at(i)) != delim.end() && i + 1 < id.length())
            {
                id.at(i + 1) = toupper(id.at(i + 1));
                ++i;
            }
        }
    }
    // remove all delimiters
    for(auto ditr : delim)
    {
        size_t _pos = 0;
        while((_pos = id.find(ditr)) != std::string::npos)
            id = id.erase(_pos, 1);
    }

    return id;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename V = typename T::value_type,
          std::enable_if_t<!(std::is_same<V, void>::value), int> = 0>
static inline auto
get(py::class_<pytuple_t<T>>& _pyclass)
    -> decltype(std::get<0>(std::declval<pytuple_t<T>>().get()), void())
{
    using bundle_t = pytuple_t<T>;
    auto _get      = [](bundle_t* obj) { return std::get<0>(obj->get()); };
    _pyclass.def("get", _get, "Get the current value");
    _pyclass.def_property_readonly_static(
        "has_value", [](py::object) { return true; },
        "Whether the component has an accessible value");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename V = typename T::value_type,
          std::enable_if_t<(std::is_same<V, void>::value), int> = 0>
static inline void
get(py::class_<pytuple_t<T>>& _pyclass)
{
    using bundle_t = pytuple_t<T>;
    auto _get      = [](bundle_t*) { return py::none{}; };
    _pyclass.def("get", _get, "Component does not return value");
    _pyclass.def_property_readonly_static(
        "has_value", [](py::object) { return false; },
        "Whether the component has an accessible value");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
record(py::class_<pytuple_t<T>>& _pyclass, int, int)
    -> decltype(T::record(std::declval<Args>()...), void())
{
    auto _record = [](Args... _args) { return T::record(_args...); };
    _pyclass.def_static("record", _record, "Get the record of a measurement");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
record(py::class_<pytuple_t<T>>& _pyclass, int, long) -> decltype(
    std::declval<pytuple_t<T>>().template get<T>()->record(std::declval<Args>()...),
    void())
{
    using bundle_t = pytuple_t<T>;
    auto _record   = [](bundle_t* obj, Args... args) {
        return obj->template get<T>()->record(args...);
    };
    _pyclass.def("record", _record, "Get the record of a measurement");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline void
record(py::class_<pytuple_t<T>>&, long, long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
configure(py::class_<pytuple_t<T>>& _pyclass, int, int, Args&&... args)
    -> decltype(T::configure(tim::api::python{}, std::forward<Args>(args)...), void())
{
    auto _configure = [](Args&&... _args) {
        T::configure(tim::api::python{}, std::forward<Args>(_args)...);
    };
    _pyclass.def_static("configure", _configure, "Configure the tool");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
configure(py::class_<pytuple_t<T>>& _pyclass, int, long, Args&&... args)
    -> decltype(std::declval<pytuple_t<T>>().template get<T>()->configure(
                    tim::api::python{}, std::forward<Args>(args)...),
                void())
{
    using bundle_t  = pytuple_t<T>;
    auto _configure = [](bundle_t* obj, Args&&... _args) {
        if(obj->template get<T>())
            obj->template get<T>()->configure(tim::api::python{},
                                              std::forward<Args>(_args)...);
    };
    _pyclass.def("configure", _configure, "Configure the tool");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
configure(py::class_<pytuple_t<T>>&, long, long, Args&&...)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
static inline auto
configure(py::class_<pytuple_t<T>>& _pyclass, int)
    -> decltype(T::configure(tim::api::python{}, _pyclass))
{
    T::configure(tim::api::python{}, _pyclass);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
static inline auto
configure(py::class_<pytuple_t<T>>&, long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <int OpT, typename... Args>
struct process_args;
//
template <int OpT>
struct process_args<OpT>
{
    template <typename U>
    static void generate(py::class_<U>&)
    {}
};
//
template <int OpT, typename Arg, typename... Args>
struct process_args<OpT, Arg, Args...>
{
    template <typename U, int Op = OpT,
              std::enable_if_t<(Op == TIMEMORY_CONSTRUCT), int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _init = [](Arg arg, Args... args) {
            auto obj = new U{};
            obj->construct(arg, args...);
            return obj;
        };
        _pycomp.def(py::init(_init), "Construct");
    }

    template <typename U, int Op = OpT, std::enable_if_t<(Op == TIMEMORY_START), int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return obj->start(arg, args...);
        };
        _pycomp.def("start", _func, "Start measurement");
    }

    template <typename U, int Op = OpT, std::enable_if_t<(Op == TIMEMORY_STOP), int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return obj->stop(arg, args...);
        };
        _pycomp.def("stop", _func, "Stop measurement");
    }

    template <typename U, int Op = OpT, std::enable_if_t<(Op == TIMEMORY_STORE), int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return obj->store(arg, args...);
        };
        _pycomp.def("store", _func, "Store measurement");
    }

    template <typename U, int Op = OpT,
              std::enable_if_t<(Op == TIMEMORY_RECORD), int> = 0>
    static void generate(py::class_<pytuple_t<U>>& _pycomp)
    {
        pyinternal::record<U, Arg, Args...>(_pycomp, 0, 0);
    }

    template <typename U, int Op = OpT,
              std::enable_if_t<(Op == TIMEMORY_MEASURE), int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return obj->measure(arg, args...);
        };
        _pycomp.def("measure", _func, "Take a measurement");
    }

    template <typename U, int Op = OpT,
              std::enable_if_t<(Op == TIMEMORY_MARK_BEGIN), int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return obj->mark_begin(arg, args...);
        };
        _pycomp.def("mark_begin", _func, "Mark a begin point");
    }

    template <typename U, int Op = OpT,
              std::enable_if_t<(Op == TIMEMORY_MARK_END), int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return obj->mark_end(arg, args...);
        };
        _pycomp.def("mark_end", _func, "Mark an end point");
    }
};
//
template <int OpT, typename... Args, typename... Tail>
struct process_args<OpT, tim::type_list<Args...>, Tail...>
{
    template <typename U>
    static void generate(py::class_<U>& _pycomp)
    {
        process_args<OpT, Args...>::generate(_pycomp);
        process_args<OpT, Tail...>::generate(_pycomp);
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class TupleT, typename T, size_t... Idx>
static void
operations(py::class_<TupleT<T>>& _pyclass, std::index_sequence<Idx...>)
{
    using tim::trait::python_args_t;
    TIMEMORY_FOLD_EXPRESSION(
        pyinternal::process_args<Idx, python_args_t<Idx, T>>::generate(_pyclass));
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, std::enable_if_t<(tim::component::enumerator<Idx>::value), int> = 0>
static void
generate(py::module& _pymod)
{
    using T = typename tim::component::enumerator<Idx>::type;
    if(std::is_same<T, tim::component::placeholder<tim::component::nothing>>::value)
        return;
    using property_t = tim::component::properties<T>;
    using bundle_t   = pytuple_t<T>;
    std::string id   = get_class_name(property_t::enum_string());

    auto _init       = []() { return new bundle_t{}; };
    auto _sinit      = [](std::string key) { return new bundle_t(key); };
    auto _push       = [](bundle_t* obj) { obj->push(); };
    auto _pop        = [](bundle_t* obj) { obj->pop(); };
    auto _start      = [](bundle_t* obj) { obj->start(); };
    auto _stop       = [](bundle_t* obj) { obj->stop(); };
    auto _measure    = [](bundle_t* obj) { obj->measure(); };
    auto _reset      = [](bundle_t* obj) { obj->reset(); };
    auto _mark_begin = [](bundle_t* obj) { obj->mark_begin(); };
    auto _mark_end   = [](bundle_t* obj) { obj->mark_end(); };
    auto _hash       = [](bundle_t* obj) { return obj->hash(); };
    auto _key        = [](bundle_t* obj) { return obj->key(); };
    auto _laps       = [](bundle_t* obj) { return obj->laps(); };
    auto _rekey      = [](bundle_t* obj, std::string _key) { obj->rekey(_key); };

    auto _isub = [](bundle_t* lhs, bundle_t* rhs) {
        if(lhs && rhs)
            *lhs -= *rhs;
        return lhs;
    };
    auto _repr = [](bundle_t* obj) {
        std::stringstream ss;
        if(obj)
            ss << *obj;
        return ss.str();
    };

    py::class_<bundle_t> _pycomp(_pymod, id.c_str(), T::description().c_str());

    // these have direct mappings to the component
    _pycomp.def(py::init(_init), "Creates component");
    _pycomp.def(py::init(_sinit), "Creates component with a label");
    _pycomp.def("push", _push, "Push into the call-graph");
    _pycomp.def("pop", _pop, "Pop off the call-graph");
    _pycomp.def("start", _start, "Start measurement");
    _pycomp.def("stop", _stop, "Stop measurement");
    _pycomp.def("measure", _measure, "Take a measurement");
    _pycomp.def("reset", _reset, "Reset the values");
    _pycomp.def("mark_begin", _mark_begin, "Mark an begin point");
    _pycomp.def("mark_end", _mark_end, "Mark an end point");

    // these require further evaluation
    pyinternal::get(_pycomp);
    pyinternal::record(_pycomp, 0, 0);
    pyinternal::configure(_pycomp, 0);
    pyinternal::configure(_pycomp, 0, 0, py::args{}, py::kwargs{});
    pyinternal::operations(_pycomp, std::make_index_sequence<TIMEMORY_OPERATION_END>{});

    // these are operations on the bundler
    _pycomp.def("hash", _hash, "Get the current hash");
    _pycomp.def("key", _key, "Get the identifier");
    _pycomp.def("rekey", _rekey, "Change the identifier");
    _pycomp.def("laps", _laps, "Get the number of laps");

    // operators
    _pycomp.def(py::self + py::self);
    _pycomp.def(py::self - py::self);
    _pycomp.def(py::self += py::self);
    _pycomp.def("__isub__", _isub, "Subtract rhs from lhs", py::is_operator());
    _pycomp.def("__repr__", _repr, "String representation");

    _pycomp.def_static("label", &T::label, "Get the label for the type");
    _pycomp.def_static("description", &T::description,
                       "Get the description for the type");
    _pycomp.def_property_readonly_static(
        "available", [](py::object) { return true; },
        "Whether the component is available");
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx,
          std::enable_if_t<!(tim::component::enumerator<Idx>::value), int> = 0>
static void
generate(py::module& _pymod)
{
    using T = typename tim::component::enumerator<Idx>::type;
    if(std::is_same<T, tim::component::placeholder<tim::component::nothing>>::value)
        return;
    using property_t  = tim::component::properties<T>;
    using bundle_t    = pytuple_t<T>;
    std::string id    = get_class_name(property_t::enum_string());
    std::string _desc = "not available";

    auto _init       = []() { return new bundle_t{}; };
    auto _sinit      = [](std::string) { return new bundle_t{}; };
    auto _push       = [](bundle_t*) {};
    auto _pop        = [](bundle_t*) {};
    auto _start      = [](bundle_t*) {};
    auto _stop       = [](bundle_t*) {};
    auto _measure    = [](bundle_t*) {};
    auto _reset      = [](bundle_t*) {};
    auto _mark_begin = [](bundle_t*) {};
    auto _mark_end   = [](bundle_t*) {};
    auto _hash       = [](bundle_t*) { return py::none{}; };
    auto _key        = [](bundle_t*) { return py::none{}; };
    auto _laps       = [](bundle_t*) { return py::none{}; };
    auto _rekey      = [](bundle_t*, std::string) {};
    auto _add        = [](bundle_t*, bundle_t*) { return py::none{}; };
    auto _sub        = [](bundle_t*, bundle_t*) { return py::none{}; };
    auto _iadd       = [](bundle_t*, bundle_t*) { return py::none{}; };
    auto _isub       = [](bundle_t*, bundle_t*) { return py::none{}; };
    auto _repr       = [](bundle_t*) { return std::string(""); };
    auto _get        = [](bundle_t*) { return py::none{}; };

    py::class_<bundle_t> _pycomp(_pymod, id.c_str(), _desc.c_str());

    _pycomp.def(py::init(_init), "Creates component");
    _pycomp.def(py::init(_sinit), "Creates component with a label");
    _pycomp.def("push", _push, "Push into the call-graph");
    _pycomp.def("pop", _pop, "Pop off the call-graph");
    _pycomp.def("start", _start, "Start measurement");
    _pycomp.def("stop", _stop, "Stop measurement");
    _pycomp.def("measure", _measure, "Take a measurement");
    _pycomp.def("reset", _reset, "Reset the values");
    _pycomp.def("mark_begin", _mark_begin, "Mark an begin point");
    _pycomp.def("mark_end", _mark_end, "Mark an end point");
    _pycomp.def("get", _get, "No value available");

    // these are operations on the bundler
    _pycomp.def("hash", _hash, "Get the current hash");
    _pycomp.def("key", _key, "Get the identifier");
    _pycomp.def("rekey", _rekey, "Change the identifier");
    _pycomp.def("laps", _laps, "Get the number of laps");

    // operators
    _pycomp.def("__add__", _add, "Get addition of two components", py::is_operator());
    _pycomp.def("__sub__", _sub, "Get difference between two components",
                py::is_operator());
    _pycomp.def("__iadd__", _iadd, "Add rhs to lhs", py::is_operator());
    _pycomp.def("__isub__", _isub, "Subtract rhs from lhs", py::is_operator());
    _pycomp.def("__repr__", _repr, "String representation");

    _pycomp.def_property_readonly_static(
        "available", [](py::object) { return false; },
        "Whether the component is available");
    _pycomp.def_property_readonly_static(
        "has_value", [](py::object) { return false; },
        "Whether the component has an accessible value");
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t... Idx>
static void
components(py::module& _pymod, std::index_sequence<Idx...>)
{
    TIMEMORY_FOLD_EXPRESSION(pyinternal::generate<Idx>(_pymod));
}
}  // namespace pyinternal
//
//======================================================================================//
//
namespace pycomponents
{
py::module
generate(py::module& _pymod)
{
    py::module _pycomp = _pymod.def_submodule(
        "component",
        "Stand-alone classes for the components. Unless push() and pop() are called on "
        "these objects, they will not store any data in the timemory call-graph (if "
        "applicable)");
    pyinternal::components(_pycomp, std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
    return _pycomp;
}
}  // namespace pycomponents
//
//======================================================================================//
