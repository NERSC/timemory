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

/** \file mpl/bits/types.hpp
 * \headerfile mpl/bits/types.hpp "timemory/mpl/bits/types.hpp"
 *
 * This is a declaration of the STL overload types
 *
 */

#pragma once

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "timemory/mpl/types.hpp"

namespace tim
{
namespace stl
{
namespace ostream
{
//--------------------------------------------------------------------------------------//
//
//      operator <<
//
//--------------------------------------------------------------------------------------//

template <typename T, typename U>
std::ostream&
operator<<(std::ostream&, const std::pair<T, U>&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::ostream&
operator<<(std::ostream&, const std::tuple<_Types...>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::ostream&
operator<<(std::ostream&, const std::vector<_Tp, _Extra...>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
std::ostream&
operator<<(std::ostream&, const std::array<_Tp, _N>&);

//--------------------------------------------------------------------------------------//

template <template <typename...> class _Tuple, typename... _Types, size_t... _Idx>
void
tuple_printer(const _Tuple<_Types...>& obj, std::ostream& os, index_sequence<_Idx...>)
{
    using namespace ::tim::stl::ostream;
    constexpr size_t _N = sizeof...(_Types);

    if(_N > 0)
        os << "(";
    char delim[_N];
    TIMEMORY_FOLD_EXPRESSION(delim[_Idx] = ',');
    delim[_N - 1] = '\0';
    TIMEMORY_FOLD_EXPRESSION(os << std::get<_Idx>(obj) << delim[_Idx]);
    if(_N > 0)
        os << ")";
}

//--------------------------------------------------------------------------------------//

}  // namespace ostream

//--------------------------------------------------------------------------------------//
//
//      operator +=
//      operator +
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N, typename _Other>
std::array<_Tp, _N>&
operator+=(std::array<_Tp, _N>&, _Other&&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename _Other>
std::pair<_Lhs, _Rhs>&
operator+=(std::pair<_Lhs, _Rhs>&, _Other&&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra, typename _Other>
std::vector<_Tp, _Extra...>&
operator+=(std::vector<_Tp, _Extra...>&, _Other&&);

//--------------------------------------------------------------------------------------//

template <typename... _Types, typename _Other>
std::tuple<_Types...>&
operator+=(std::tuple<_Types...>&, _Other&&);

//--------------------------------------------------------------------------------------//
//
//      operator -=
//      operator -
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
std::array<_Tp, _N>&
operator-=(std::array<_Tp, _N>&, const std::array<_Tp, _N>&);

template <typename _Lhs, size_t _N, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
std::array<_Lhs, _N>&
operator-=(std::array<_Lhs, _N>&, const _Rhs&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator-=(std::pair<_Lhs, _Rhs>&, const std::pair<_Lhs, _Rhs>&);

template <typename _Lhs, typename _Rhs, typename _Arith,
          enable_if_t<(std::is_arithmetic<decay_t<_Arith>>::value), int> = 0>
std::pair<_Lhs, _Rhs>&
operator-=(std::pair<_Lhs, _Rhs>&, const _Arith&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator-=(std::vector<_Tp, _Extra...>&, const std::vector<_Tp, _Extra...>&);

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
std::vector<_Lhs, _Extra...>&
operator-=(std::vector<_Lhs, _Extra...>&, const _Rhs&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator-=(std::tuple<_Types...>&, const std::tuple<_Types...>&);

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
std::tuple<_Lhs...>&
operator-=(std::tuple<_Lhs...>&, const _Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator *=
//      operator *
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
std::array<_Tp, _N>&
operator*=(std::array<_Tp, _N>&, const std::array<_Tp, _N>&);

template <typename _Lhs, size_t _N, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
std::array<_Lhs, _N>&
operator*=(std::array<_Lhs, _N>&, const _Rhs&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator*=(std::pair<_Lhs, _Rhs>&, const std::pair<_Lhs, _Rhs>&);

template <typename _Lhs, typename _Rhs, typename _Arith,
          enable_if_t<(std::is_arithmetic<decay_t<_Arith>>::value), int> = 0>
std::pair<_Lhs, _Rhs>&
operator*=(std::pair<_Lhs, _Rhs>&, const _Arith&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator*=(std::vector<_Tp, _Extra...>&, const std::vector<_Tp, _Extra...>&);

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
std::vector<_Lhs, _Extra...>&
operator*=(std::vector<_Lhs, _Extra...>&, const _Rhs&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator*=(std::tuple<_Types...>&, const std::tuple<_Types...>&);

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
std::tuple<_Lhs...>&
operator*=(std::tuple<_Lhs...>&, const _Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator /=
//      operator /
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
std::array<_Tp, _N>&
operator/=(std::array<_Tp, _N>&, const std::array<_Tp, _N>&);

template <typename _Lhs, size_t _N, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
std::array<_Lhs, _N>&
operator/=(std::array<_Lhs, _N>&, const _Rhs&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator/=(std::pair<_Lhs, _Rhs>&, const std::pair<_Lhs, _Rhs>&);

template <typename _Lhs, typename _Rhs, typename _Arith,
          enable_if_t<(std::is_arithmetic<decay_t<_Arith>>::value), int> = 0>
std::pair<_Lhs, _Rhs>&
operator/=(std::pair<_Lhs, _Rhs>&, const _Arith&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator/=(std::vector<_Tp, _Extra...>&, const std::vector<_Tp, _Extra...>&);

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
std::vector<_Lhs, _Extra...>&
operator/=(std::vector<_Lhs, _Extra...>&, const _Rhs&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator/=(std::tuple<_Types...>&, const std::tuple<_Types...>&);

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
std::tuple<_Lhs...>&
operator/=(std::tuple<_Lhs...>&, const _Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator * (fundamental)
//      operator / (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
_Lhs operator*(_Lhs, const _Rhs&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int> = 0>
_Lhs
operator/(_Lhs, const _Rhs&);

}  // namespace stl

}  // namespace tim
