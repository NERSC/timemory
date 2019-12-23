// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

/** \file component/properties.hpp
 * \headerfile component/properties.hpp "timemory/component/properties.hpp"
 * Provides properties for components
 *
 */

#pragma once

#include "timemory/enum.h"

namespace tim
{
namespace component
{
template <typename _Tp>
using uoset_t = std::unordered_set<_Tp>;
using idset_t = uoset_t<std::string>;

//--------------------------------------------------------------------------------------//
//
template <TIMEMORY_COMPONENT _ID>
struct enumerator;

//--------------------------------------------------------------------------------------//
//
template <typename _Tp>
struct state
{
    static bool& has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//
//
template <typename _Tp>
struct properties
{
    using type                                = _Tp;
    using value_type                          = TIMEMORY_COMPONENT;
    static constexpr TIMEMORY_COMPONENT value = TIMEMORY_COMPONENTS_END;
    static constexpr const char* enum_string() { return "TIMEMORY_COMPONENTS_END"; }
    static constexpr const char* id() { return ""; }
    static idset_t               ids() { return idset_t{}; }
};

//
//--------------------------------------------------------------------------------------//
}  // namespace component
}  // namespace tim
