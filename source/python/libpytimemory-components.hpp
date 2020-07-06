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

#pragma once

#if !defined(TIMEMORY_PYBIND11_SOURCE)
#    define TIMEMORY_PYBIND11_SOURCE
#endif

//======================================================================================//
// disables a bunch of warnings
//
#include "timemory/components/types.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/utility/macros.hpp"

#include "pybind11/cast.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/operators.h"
#include "pybind11/options.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

//======================================================================================//

namespace py = pybind11;
using namespace py::literals;

//--------------------------------------------------------------------------------------//
//
//                                       GENERAL
//
//--------------------------------------------------------------------------------------//
//
namespace pytim
{
template <typename TupleT>
struct construct_dict
{
    using type = TupleT;
    construct_dict(const TupleT& _tup, py::dict& _dict)
    {
        auto _label = std::get<0>(_tup);
        if(_label.size() > 0)
            _dict[_label.c_str()] = std::get<1>(_tup);
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct construct_dict<std::tuple<std::string, void>>
{
    using type = std::tuple<std::string, void>;
    template <typename... ArgsT>
    construct_dict(ArgsT&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
struct dict
{
    template <typename... Types>
    static py::dict construct(const std::tuple<Types...>& _tup)
    {
        using apply_types = std::tuple<construct_dict<Types>...>;
        py::dict _dict;
        ::tim::apply<void>::access<apply_types>(_tup, std::ref(_dict));
        return _dict;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace pytim
//
//--------------------------------------------------------------------------------------//
//
//                                       COMPONENTS
//
//--------------------------------------------------------------------------------------//
//
namespace pycomponents
{
py::module
generate(py::module& _pymod);
}
//
//--------------------------------------------------------------------------------------//
//
//                                      ENUMERATION
//
//--------------------------------------------------------------------------------------//
//
namespace pyenumeration
{
py::enum_<TIMEMORY_NATIVE_COMPONENT>
generate(py::module& _pymod);
}
//
//--------------------------------------------------------------------------------------//
//
//                                       SETTINGS
//
//--------------------------------------------------------------------------------------//
//
namespace pysettings
{
struct settings
{};
//
py::class_<pysettings::settings>
generate(py::module& _pymod);
}  // namespace pysettings
//
//--------------------------------------------------------------------------------------//
//
//                                       SIGNALS
//
//--------------------------------------------------------------------------------------//
//
namespace pysignals
{
py::module
generate(py::module& _pymod);
}  // namespace pysignals
//
//--------------------------------------------------------------------------------------//
//
//                                        UNITS
//
//--------------------------------------------------------------------------------------//
//
namespace pyunits
{
py::module
generate(py::module& _pymod);
}  // namespace pyunits
//
//--------------------------------------------------------------------------------------//
//
//                                      HARDWARE_COUNTERS
//
//--------------------------------------------------------------------------------------//
//
namespace pyhardware_counters
{
py::module
generate(py::module& _pymod);
}  // namespace pyhardware_counters
//
//--------------------------------------------------------------------------------------//
//
//                                          APIs
//
//--------------------------------------------------------------------------------------//
//
namespace pyapi
{
py::module
generate(py::module& _pymod);
}  // namespace pyapi
//
//--------------------------------------------------------------------------------------//
//
//                                      COMPONENT_LIST
//
//--------------------------------------------------------------------------------------//
//
namespace pycomponent_list
{
void
generate(py::module& _pymod);
}  // namespace pycomponent_list
//
//--------------------------------------------------------------------------------------//
//
//                                      COMPONENT_BUNDLE
//
//--------------------------------------------------------------------------------------//
//
namespace pycomponent_bundle
{
void
generate(py::module& _pymod);
}  // namespace pycomponent_bundle
//
//--------------------------------------------------------------------------------------//
//
//                                      AUTO_TIMER
//
//--------------------------------------------------------------------------------------//
//
namespace pyauto_timer
{
void
generate(py::module& _pymod);
}  // namespace pyauto_timer
//
//--------------------------------------------------------------------------------------//
//
//                                      RSS USAGE
//
//--------------------------------------------------------------------------------------//
//
namespace pyrss_usage
{
using namespace tim::component;
using rss_usage_t =
    tim::component_bundle_t<TIMEMORY_API, page_rss, peak_rss, num_minor_page_faults,
                            num_major_page_faults, voluntary_context_switch,
                            priority_context_switch>;

py::class_<rss_usage_t>
generate(py::module& _pymod, py::module& _pyunits);
}  // namespace pyrss_usage
//
//--------------------------------------------------------------------------------------//
