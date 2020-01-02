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

/** \file mpl/stl_overload.hpp
 * \headerfile mpl/stl_overload.hpp "timemory/mpl/stl_overload.hpp"
 * Provides operators on common STL structures such as <<, +=, -=, *=, /=, +, -, *, /
 *
 */

#pragma once

#include <array>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include "timemory/utility/macros.hpp"

namespace tim
{
namespace stl_overload
{
//--------------------------------------------------------------------------------------//
//
//      operator <<
//
//--------------------------------------------------------------------------------------//

template <typename T, typename U>
::std::ostream&
operator<<(::std::ostream&, const ::std::pair<T, U>&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::ostream&
operator<<(::std::ostream&, const ::std::tuple<_Types...>&);

//--------------------------------------------------------------------------------------//
//
//      operator +=
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator+=(::std::array<_Tp, _N>&, const ::std::array<_Tp, _N>&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator+=(::std::pair<_Lhs, _Rhs>&, const ::std::pair<_Lhs, _Rhs>&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator+=(::std::tuple<_Types...>&, const ::std::tuple<_Types...>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
::std::vector<_Tp, _Extra...>&
operator+=(::std::vector<_Tp, _Extra...>&, const ::std::vector<_Tp, _Extra...>&);

//--------------------------------------------------------------------------------------//
//
//      operator -=
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator-=(::std::array<_Tp, _N>&, const ::std::array<_Tp, _N>&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator-=(::std::tuple<_Types...>&, const ::std::tuple<_Types...>&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator-=(::std::pair<_Lhs, _Rhs>&, const ::std::pair<_Lhs, _Rhs>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
::std::vector<_Tp, _Extra...>&
operator-=(::std::vector<_Tp, _Extra...>&, const ::std::vector<_Tp, _Extra...>&);

//--------------------------------------------------------------------------------------//
//
//      operator *=
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator*=(::std::array<_Tp, _N>&, const ::std::array<_Tp, _N>&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator*=(::std::pair<_Lhs, _Rhs>&, const ::std::pair<_Lhs, _Rhs>&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator*=(::std::tuple<_Types...>&, const ::std::tuple<_Types...>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
::std::vector<_Tp, _Extra...>&
operator*=(::std::vector<_Tp, _Extra...>&, const ::std::vector<_Tp, _Extra...>&);

//--------------------------------------------------------------------------------------//
//
//      operator /=
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator/=(::std::array<_Tp, _N>&, const ::std::array<_Tp, _N>&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator/=(::std::pair<_Lhs, _Rhs>&, const ::std::pair<_Lhs, _Rhs>&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator/=(::std::tuple<_Types...>&, const ::std::tuple<_Types...>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
::std::vector<_Tp, _Extra...>&
operator/=(::std::vector<_Tp, _Extra...>&, const ::std::vector<_Tp, _Extra...>&);

//--------------------------------------------------------------------------------------//
//
//      operator +
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
const ::std::array<_Tp, _N>
operator+(::std::array<_Tp, _N> lhs, const ::std::array<_Tp, _N>& rhs)
{
    return lhs += rhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
const ::std::tuple<_Types...>
operator+(::std::tuple<_Types...> lhs, const ::std::tuple<_Types...>& rhs)
{
    return lhs += rhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
const ::std::pair<_Lhs, _Rhs>
operator+(::std::pair<_Lhs, _Rhs> lhs, const ::std::pair<_Lhs, _Rhs>& rhs)
{
    return lhs += rhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
const ::std::vector<_Tp, _Extra...>
operator+(::std::vector<_Tp, _Extra...> lhs, const ::std::vector<_Tp, _Extra...>& rhs)
{
    return lhs += rhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator -
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
const ::std::array<_Tp, _N>
operator-(::std::array<_Tp, _N> lhs, const ::std::array<_Tp, _N>& rhs)
{
    return lhs -= rhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
const ::std::tuple<_Types...>
operator-(::std::tuple<_Types...> lhs, const ::std::tuple<_Types...>& rhs)
{
    return lhs -= rhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
const ::std::pair<_Lhs, _Rhs>
operator-(::std::pair<_Lhs, _Rhs> lhs, const ::std::pair<_Lhs, _Rhs>& rhs)
{
    return lhs -= rhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
const ::std::vector<_Tp, _Extra...>
operator-(::std::vector<_Tp, _Extra...> lhs, const ::std::vector<_Tp, _Extra...>& rhs)
{
    return lhs -= rhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator *
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
const ::std::array<_Tp, _N> operator*(::std::array<_Tp, _N>        lhs,
                                      const ::std::array<_Tp, _N>& rhs)
{
    return lhs *= rhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
const ::std::tuple<_Types...> operator*(::std::tuple<_Types...>        lhs,
                                        const ::std::tuple<_Types...>& rhs)
{
    return lhs *= rhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
const ::std::pair<_Lhs, _Rhs> operator*(::std::pair<_Lhs, _Rhs>        lhs,
                                        const ::std::pair<_Lhs, _Rhs>& rhs)
{
    return lhs *= rhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
const ::std::vector<_Tp, _Extra...> operator*(::std::vector<_Tp, _Extra...>        lhs,
                                              const ::std::vector<_Tp, _Extra...>& rhs)
{
    return lhs *= rhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator /
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
const ::std::array<_Tp, _N>
operator/(::std::array<_Tp, _N> lhs, const ::std::array<_Tp, _N>& rhs)
{
    return lhs /= rhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
const ::std::tuple<_Types...>
operator/(::std::tuple<_Types...> lhs, const ::std::tuple<_Types...>& rhs)
{
    return lhs /= rhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
const ::std::pair<_Lhs, _Rhs>
operator/(::std::pair<_Lhs, _Rhs> lhs, const ::std::pair<_Lhs, _Rhs>& rhs)
{
    return lhs /= rhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
const ::std::vector<_Tp, _Extra...>
operator/(::std::vector<_Tp, _Extra...> lhs, const ::std::vector<_Tp, _Extra...>& rhs)
{
    return lhs /= rhs;
}

}  // namespace stl_overload

}  // namespace tim

#include "timemory/mpl/bits/stl_overload.hpp"

#if defined(_WINDOWS)
namespace std
{
template <typename _Lhs, typename _Rhs>
const pair<_Lhs, _Rhs>
operator-(pair<_Lhs, _Rhs> lhs, const pair<_Lhs, _Rhs>& rhs)
{
    lhs.first -= rhs.first;
    lhs.second -= rhs.second;
    return lhs;
}

template <typename... _Types>
const tuple<_Types...>
operator-(tuple<_Types...> lhs, const tuple<_Types...>& rhs)
{
    ::tim::stl_overload::mpl::minus(lhs, rhs);
    return lhs;
}
}  // namespace std
#endif
