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
    _pyclass.def_static("has_value", []() { return true; },
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
    _pyclass.def_static("has_value", []() { return false; },
                        "Whether the component has an accessible value");
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

    auto _init    = [](const std::string& key) { return new bundle_t(key); };
    auto _start   = [](bundle_t* obj) { obj->start(); };
    auto _stop    = [](bundle_t* obj) { obj->stop(); };
    auto _push    = [](bundle_t* obj) { obj->push(); };
    auto _pop     = [](bundle_t* obj) { obj->pop(); };
    auto _measure = [](bundle_t* obj) { obj->measure(); };
    auto _reset   = [](bundle_t* obj) { obj->reset(); };
    auto _hash    = [](bundle_t* obj) { return obj->hash(); };
    auto _key     = [](bundle_t* obj) { return obj->key(); };
    auto _laps    = [](bundle_t* obj) { return obj->laps(); };
    auto _store   = [](bundle_t* obj) { return obj->store(); };
    auto _rekey   = [](bundle_t* obj, std::string _key) { obj->rekey(_key); };
    auto _add     = [](bundle_t* lhs, bundle_t* rhs) -> bundle_t* {
        if(!lhs || !rhs)
            return nullptr;
        auto ret = new bundle_t(*lhs);
        *ret += *rhs;
        return ret;
    };
    auto _sub = [](bundle_t* lhs, bundle_t* rhs) -> bundle_t* {
        if(!lhs || !rhs)
            return nullptr;
        auto ret = new bundle_t(*lhs);
        *ret -= *rhs;
        return ret;
    };
    auto _iadd = [](bundle_t* lhs, bundle_t* rhs) {
        if(lhs && rhs)
            *lhs += *rhs;
        return lhs;
    };
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

    _pycomp.def(py::init(_init), "Creates component");
    _pycomp.def("start", _start, "Start measurement");
    _pycomp.def("stop", _stop, "Stop measurement");
    _pycomp.def("push", _push, "Push into the call-graph");
    _pycomp.def("pop", _pop, "Pop off the call-graph");
    _pycomp.def("measure", _measure, "Take a measurement");
    _pycomp.def("reset", _reset, "Reset the values");
    _pycomp.def("hash", _hash, "Get the current hash");
    _pycomp.def("key", _key, "Get the identifier");
    _pycomp.def("rekey", _rekey, "Change the identifier");
    _pycomp.def("laps", _laps, "Get the number of laps");
    _pycomp.def("store", _store, "Get the storage setting");
    pyinternal::get(_pycomp);

    // operators
    _pycomp.def("__add__", _add, "Get addition of two components");
    _pycomp.def("__sub__", _sub, "Get difference between two components");
    _pycomp.def("__iadd__", _iadd, "Add rhs to lhs");
    _pycomp.def("__isub__", _isub, "Subtract rhs from lhs");
    _pycomp.def("__repr__", _repr, "String representation");

    // auto _sample  = [](bundle_t* obj) { obj->sample(); };
    // _pycomp.def("sample", _sample, "Take a sample");

    _pycomp.def_static("label", &T::label, "Get the label for the type");
    _pycomp.def_static("description", &T::description,
                       "Get the description for the type");
    _pycomp.def_static("available", []() { return true; },
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

    auto _init    = [](const std::string&) -> bundle_t* { return nullptr; };
    auto _start   = [](bundle_t*) {};
    auto _stop    = [](bundle_t*) {};
    auto _push    = [](bundle_t*) {};
    auto _pop     = [](bundle_t*) {};
    auto _measure = [](bundle_t*) {};
    auto _reset   = [](bundle_t*) {};
    auto _hash    = [](bundle_t*) { return py::none{}; };
    auto _key     = [](bundle_t*) { return py::none{}; };
    auto _laps    = [](bundle_t*) { return py::none{}; };
    auto _store   = [](bundle_t*) { return py::none{}; };
    auto _rekey   = [](bundle_t*, std::string) {};
    auto _add     = [](bundle_t*, bundle_t*) { return py::none{}; };
    auto _sub     = [](bundle_t*, bundle_t*) { return py::none{}; };
    auto _iadd    = [](bundle_t*, bundle_t*) { return py::none{}; };
    auto _isub    = [](bundle_t*, bundle_t*) { return py::none{}; };
    auto _repr    = [](bundle_t*) { return std::string(""); };
    auto _get     = [](bundle_t*) { return py::none{}; };

    py::class_<bundle_t> _pycomp(_pymod, id.c_str(), _desc.c_str());

    _pycomp.def(py::init(_init), "Creates component");
    _pycomp.def("start", _start, "Start measurement");
    _pycomp.def("stop", _stop, "Stop measurement");
    _pycomp.def("push", _push, "Push into the call-graph");
    _pycomp.def("pop", _pop, "Pop off the call-graph");
    _pycomp.def("measure", _measure, "Take a measurement");
    _pycomp.def("reset", _reset, "Reset the values");
    _pycomp.def("hash", _hash, "Get the current hash");
    _pycomp.def("key", _key, "Get the identifier");
    _pycomp.def("rekey", _rekey, "Change the identifier");
    _pycomp.def("laps", _laps, "Get the number of laps");
    _pycomp.def("store", _store, "Get the storage setting");
    _pycomp.def("get", _get, "No value available");

    // operators
    _pycomp.def("__add__", _add, "Get addition of two components");
    _pycomp.def("__sub__", _sub, "Get difference between two components");
    _pycomp.def("__iadd__", _iadd, "Add rhs to lhs");
    _pycomp.def("__isub__", _isub, "Subtract rhs from lhs");
    _pycomp.def("__repr__", _repr, "String representation");

    _pycomp.def_static("available", []() { return false; },
                       "Whether the component is available");
    _pycomp.def_static("has_value", []() { return false; },
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
};  // namespace pyinternal
//
//======================================================================================//
//
namespace pycomponents
{
py::module
generate(py::module& _pymod)
{
    py::module _pycomp = _pymod.def_submodule(
        "components",
        "Stand-alone classes for the components. Unless push() and pop() are called on "
        "these objects, they will not store any data in the timemory call-graph (if "
        "applicable)");
    pyinternal::components(_pycomp, std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
    return _pycomp;
}
};  // namespace pycomponents
//
//======================================================================================//
