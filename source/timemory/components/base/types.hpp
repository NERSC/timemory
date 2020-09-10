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

/**
 * \file timemory/components/base/types.hpp
 * \brief Declare the base component types
 */

#pragma once

#include "timemory/components/macros.hpp"
#include "timemory/components/opaque.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/stl.hpp"

#include <type_traits>

//======================================================================================//
//
namespace tim
{
namespace component
{
//
// generic static polymorphic base class
template <typename Tp, typename ValueType = int64_t>
struct base;
//
//--------------------------------------------------------------------------------------//
//
namespace operators
{
//
//--------------------------------------------------------------------------------------//
//
//         the operator- is used very, very often in stop() of components
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
Tp
operator-(Tp lhs, const Tp& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operators
//
//--------------------------------------------------------------------------------------//
//
struct empty_base;
//
struct dynamic_base;
//
template <typename Tp, typename Value>
struct base;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct base<Tp, void>;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
//
//----------------------------------------------------------------------------------//
//
namespace quirk
{
//
template <typename... Types>
struct config
: component::base<config<Types...>, void>
, type_list<Types...>
{
    using type = type_list<Types...>;
    void start() {}
    void stop() {}
};
//
}  // namespace quirk
//
//----------------------------------------------------------------------------------//
//
namespace trait
{
//
template <typename Tp>
struct dynamic_base : std::false_type
{
    using type = component::empty_base;
};
//
}  // namespace trait
//
//----------------------------------------------------------------------------------//
//
}  // namespace tim
//
//======================================================================================//
