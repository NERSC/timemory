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

/** \file mpl/bits/apply.hpp
 * \headerfile mpl/bits/apply.hpp "timemory/mpl/bits/apply.hpp"
 * Provides some additional implementation for timemory/mpl/apply.hpp
 *
 */

#pragma once

#include "timemory/enum.h"
#include "timemory/mpl/apply.hpp"

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tim
{
//======================================================================================//

template <typename _Tp>
struct stl_tuple_printer
{
    using size_type = ::std::size_t;
    stl_tuple_printer(size_type _N, size_type _Ntot, const _Tp& obj, ::std::ostream& os)
    {
        os << ((_N == 0) ? "(" : "") << obj << ((_N + 1 == _Ntot) ? ")" : ",");
    }
};

/// the namespace is provided to hide stl overload from global namespace but provide
/// a method of using the namespace without a "using namespace tim;"
namespace stl_overload
{
//--------------------------------------------------------------------------------------//
//
//      operator <<
//
//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::ostream&
operator<<(::std::ostream& os, const ::std::tuple<_Types...>& p)
{
    using apply_t = ::std::tuple<stl_tuple_printer<_Types>...>;
    ::tim::apply<void>::access_with_indices<apply_t>(p, std::ref(os));
    return os;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename U>
::std::ostream&
operator<<(::std::ostream& os, const ::std::pair<T, U>& p)
{
    os << "(" << p.first << "," << p.second << ")";
    return os;
}

//--------------------------------------------------------------------------------------//
//
//      operator +=
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator+=(::std::array<_Tp, _N>& lhs, const ::std::array<_Tp, _N>& rhs)
{
    ::tim::apply<void>::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator+=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    ::tim::apply<void>::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator+=(::std::pair<_Lhs, _Rhs>& lhs, const ::std::pair<_Lhs, _Rhs>& rhs)
{
    lhs.first += rhs.first;
    lhs.second += rhs.second;
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator -=
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator-=(::std::array<_Tp, _N>& lhs, const ::std::array<_Tp, _N>& rhs)
{
    ::tim::apply<void>::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator-=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    ::tim::apply<void>::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator-=(::std::pair<_Lhs, _Rhs>& lhs, const ::std::pair<_Lhs, _Rhs>& rhs)
{
    lhs.first -= rhs.first;
    lhs.second -= rhs.second;
    return lhs;
}

//--------------------------------------------------------------------------------------//

}  // namespace stl_overload

using namespace stl_overload;

//======================================================================================//

}  // namespace tim
