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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <numeric>
#include <string>

#include "timemory/macros.hpp"
#include "timemory/papi.hpp"
#include "timemory/storage.hpp"
#include "timemory/units.hpp"
#include "timemory/utility.hpp"

#include "timemory/components/base.hpp"
#include "timemory/components/type_traits.hpp"
#include "timemory/components/types.hpp"

// general components
#include "timemory/components/cuda_event.hpp"
#include "timemory/components/resource_usage.hpp"
#include "timemory/components/timing.hpp"

// hardware counter components
#include "timemory/components/cupti_array.hpp"
#include "timemory/components/cupti_tuple.hpp"
#include "timemory/components/papi_array.hpp"
#include "timemory/components/papi_tuple.hpp"

// advanced components
#include "timemory/components/cpu_roofline.hpp"
#include "timemory/components/gpu_roofline.hpp"

#if defined(TIMEMORY_USE_CUPTI)
#    include "timemory/cupti_event.hpp"
#endif

//======================================================================================//

namespace tim
{
namespace component
{
//======================================================================================//
// component initialization
//
/*
class init
{
public:
    using string_t  = std::string;
    bool     store  = false;
    int64_t  ncount = 0;
    int64_t  nhash  = 0;
    string_t key    = "";
    string_t tag    = "";
};
*/

//======================================================================================//
// construction tuple for a component
//
template <typename Type, typename... Args>
class constructor : public std::tuple<Args...>
{
public:
    using base_type                    = std::tuple<Args...>;
    static constexpr std::size_t nargs = std::tuple_size<decay_t<base_type>>::value;

    explicit constructor(Args&&... _args)
    : base_type(std::forward<Args>(_args)...)
    {
    }

    template <typename _Tuple, size_t... _Idx>
    Type operator()(_Tuple&& __t, index_sequence<_Idx...>)
    {
        return Type(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }

    Type operator()()
    {
        return (*this)(static_cast<base_type>(*this), make_index_sequence<nargs>{});
    }
};

//--------------------------------------------------------------------------------------//
//  component_tuple initialization
//
using init = constructor<void, std::string, std::string, int64_t, int64_t, bool>;

}  // namespace component

//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//
