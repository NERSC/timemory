//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file components.hpp
 * \headerfile components.hpp "timemory/components.hpp"
 * These are core tools provided by TiMemory. These tools can be used individually
 * or bundled together in a component_tuple (C++) or component_list (C, Python)
 *
 */

#pragma once

// forward declare any types
#include "timemory/components/types.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/variadic/types.hpp"

// general components
#include "timemory/components/general.hpp"
#include "timemory/components/rusage.hpp"
#include "timemory/components/timing.hpp"

// caliper components
#if defined(TIMEMORY_USE_CALIPER)
#    include "timemory/components/caliper.hpp"
#endif

// gotcha components
#if defined(TIMEMORY_USE_GOTCHA)
#    include "timemory/components/gotcha.hpp"
#endif

// cuda event
#if defined(TIMEMORY_USE_CUDA)
#    include "timemory/components/cuda_event.hpp"
#endif

// nvtx marker
#if defined(TIMEMORY_USE_NVTX)
#    include "timemory/components/nvtx_marker.hpp"
#endif

// GPU hardware counter components
#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/components/cupti_activity.hpp"
#    include "timemory/components/cupti_counters.hpp"
#    include "timemory/components/gpu_roofline.hpp"
#endif

// CPU/GPU hardware counter components
#if defined(TIMEMORY_USE_PAPI)
#    include "timemory/components/cpu_roofline.hpp"
#    include "timemory/components/papi_array.hpp"
#    include "timemory/components/papi_tuple.hpp"
#endif

#include "timemory/backends/cuda.hpp"

// device backend
#include "timemory/backends/device.hpp"

//======================================================================================//
//
//      helpers for generating components
//
//======================================================================================//

#include "timemory/enum.h"
#include "timemory/mpl/apply.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace tim
{
//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to initialize a auto_list or component_list from a list
///      of enumerations
//
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_list<real_clock, cpu_clock, cpu_util, cuda_event>;
//
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::initialize(*obj, { CPU_CLOCK, CPU_UTIL });
//
///  typename... _ExtraArgs
///      required because of extra "hidden" template parameters in STL containers
//
template <template <typename...> class _CompList, typename... _CompTypes,
          template <typename, typename...> class _Container, typename _Intp,
          typename... _ExtraArgs>
void
initialize(_CompList<_CompTypes...>&               obj,
           const _Container<_Intp, _ExtraArgs...>& components);

//--------------------------------------------------------------------------------------//
//
///  description:
///      use this function to generate an array of enumerations from a list of string
///      that can be subsequently used to initialize an auto_list or a component_list
///
///  usage:
///      using namespace tim::component;
///      using optional_t = tim::auto_list<real_clock, cpu_clock, cpu_util, cuda_event>;
///
///      auto obj = new optional_t(__FUNCTION__, __LINE__);
///      tim::initialize(*obj, tim::enumerate_components({ "cpu_clock", "cpu_util"}));
///
template <typename _StringT, typename... _ExtraArgs,
          template <typename, typename...> class _Container>
_Container<TIMEMORY_COMPONENT>
enumerate_components(const _Container<_StringT, _ExtraArgs...>& component_names);

}  // namespace tim

// initialize and enumerate_components
#include "timemory/bits/components.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//
//
//                  specializations for std::initializer_list
//
//--------------------------------------------------------------------------------------//

template <template <typename...> class _CompList, typename... _CompTypes,
          typename _EnumT = int>
inline void
initialize(_CompList<_CompTypes...>& obj, const std::initializer_list<_EnumT>& components)
{
    initialize(obj, std::vector<_EnumT>(components));
}

//--------------------------------------------------------------------------------------//

inline std::vector<TIMEMORY_COMPONENT>
enumerate_components(const std::initializer_list<std::string>& component_names)
{
    return enumerate_components(std::vector<std::string>(component_names));
}

//--------------------------------------------------------------------------------------//

inline std::vector<TIMEMORY_COMPONENT>
enumerate_components(const std::string& names, const std::string& env_id = "")
{
    if(env_id.length() > 0)
        return enumerate_components(tim::delimit(get_env<std::string>(env_id, names)));
    else
        return enumerate_components(tim::delimit(names));
}

//--------------------------------------------------------------------------------------//
//
//                  extra specializations for std::string
//
//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a container of string
//
template <template <typename...> class _CompList, typename... _CompTypes,
          typename... _ExtraArgs, template <typename, typename...> class _Container>
inline void
initialize(_CompList<_CompTypes...>&                     obj,
           const _Container<std::string, _ExtraArgs...>& components)
{
    initialize(obj, enumerate_components(components));
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing with a string
//
template <template <typename...> class _CompList, typename... _CompTypes>
inline void
initialize(_CompList<_CompTypes...>& obj, const std::string& components)
{
    initialize(obj, enumerate_components(tim::delimit(components)));
}

//--------------------------------------------------------------------------------------//
//
/// this is for initializing reading an environment variable, getting a string, breaking
/// into list of components, and initializing
//
namespace env
{
template <template <typename...> class _CompList, typename... _CompTypes,
          typename std::enable_if<(sizeof...(_CompTypes) > 0), int>::type = 0>
inline void
initialize(_CompList<_CompTypes...>& obj, const std::string& env_var,
           const std::string& default_env)
{
    auto env_result = tim::get_env(env_var, default_env);
    initialize(obj, enumerate_components(tim::delimit(env_result)));
}

template <template <typename...> class _CompList, typename... _CompTypes,
          typename std::enable_if<(sizeof...(_CompTypes) == 0), int>::type = 0>
inline void
initialize(_CompList<_CompTypes...>&, const std::string&, const std::string&)
{}

}  // namespace env

//--------------------------------------------------------------------------------------//

}  // namespace tim
