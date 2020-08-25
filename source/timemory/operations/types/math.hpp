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
 * \file timemory/operations/types/math.hpp
 * \brief Definition for various functions for math in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::plus
/// \brief Define addition operations
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct plus
{
    using type       = Tp;
    using value_type = typename type::value_type;

    template <typename U>
    using base_t = typename U::base_type;

    TIMEMORY_DELETED_OBJECT(plus)

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    plus(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj += rhs;
        // ensures update to laps
        sfinae(obj, 0, 0, rhs);
    }

    template <typename Vt, typename Up = Tp, enable_if_t<!has_data<Up>::value, char> = 0>
    plus(type&, const Vt&)
    {}

private:
    template <typename Up>
    auto sfinae(Up& obj, int, int, const Up& rhs)
        -> decltype(static_cast<base_t<Up>&>(obj).plus(crtp::base{},
                                                       static_cast<base_t<Up>&>(rhs)),
                    void())
    {
        static_cast<base_t<Up>&>(obj).plus(crtp::base{}, static_cast<base_t<Up>&>(rhs));
    }

    template <typename U>
    auto sfinae(U& obj, int, long, const U& rhs) -> decltype(obj.plus(rhs), void())
    {
        obj.plus(rhs);
    }

    template <typename U>
    auto sfinae(U&, long, long, const U&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::minus
/// \brief Define subtraction operations
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct minus
{
    using type       = Tp;
    using value_type = typename type::value_type;

    template <typename U>
    using base_t = typename U::base_type;

    TIMEMORY_DELETED_OBJECT(minus)

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    minus(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        obj -= rhs;
        // ensures update to laps
        sfinae(obj, 0, 0, rhs);
    }

    template <typename Vt, typename Up = Tp, enable_if_t<!has_data<Up>::value, char> = 0>
    minus(type&, const Vt&)
    {}

private:
    template <typename Up>
    auto sfinae(Up& obj, int, int, const Up& rhs)
        -> decltype(static_cast<base_t<Up>&>(obj).minus(crtp::base{},
                                                        static_cast<base_t<Up>&>(rhs)),
                    void())
    {
        static_cast<base_t<Up>&>(obj).minus(crtp::base{}, static_cast<base_t<Up>&>(rhs));
    }

    template <typename U>
    auto sfinae(U& obj, int, long, const U& rhs) -> decltype(obj.minus(rhs), void())
    {
        obj.minus(rhs);
    }

    template <typename U>
    auto sfinae(U&, long, long, const U&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::multipy
/// \brief This operation class is used for multiplication of a component
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct multiply
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(multiply)

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    multiply(type& obj, const int64_t& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj *= rhs;
    }

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    multiply(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj *= rhs;
    }

    template <typename Vt, typename Up = Tp, enable_if_t<!has_data<Up>::value, char> = 0>
    multiply(type&, const Vt&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::divide
/// \brief This operation class is used for division of a component
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct divide
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(divide)

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    divide(type& obj, const int64_t& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj /= rhs;
    }

    template <typename Up = Tp, enable_if_t<has_data<Up>::value, char> = 0>
    divide(type& obj, const type& rhs)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        using namespace tim::stl;
        obj /= rhs;
    }

    template <typename Vt, typename Up = Tp, enable_if_t<!has_data<Up>::value, char> = 0>
    divide(type&, const Vt&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
