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

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "timemory/mpl/bits/types.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/types.hpp"

//======================================================================================//

#if defined(_WINDOWS)

namespace std
{
template <typename _Lhs, typename _Rhs>
const pair<_Lhs, _Rhs>
operator-(pair<_Lhs, _Rhs>, const pair<_Lhs, _Rhs>&);

template <typename... _Types>
const tuple<_Types...>
operator-(tuple<_Types...>, const tuple<_Types...>&);

}  // namespace std

#endif

//======================================================================================//

namespace tim
{
/// \namespace tim::stl
/// \brief the namespace is provided to hide stl overload from global namespace but
/// provide a method of using the namespace without a "using namespace tim;"
namespace stl
{
namespace ostream
{
/// \namespace tim::stl::ostream
/// \brief the namespace provides overloads to output complex data types w/ streams
//--------------------------------------------------------------------------------------//
//
//      operator <<
//
//--------------------------------------------------------------------------------------//

template <typename T, typename U>
std::ostream&
operator<<(std::ostream& os, const std::pair<T, U>& p)
{
    os << "(" << p.first << "," << p.second << ")";
    return os;
}

//--------------------------------------------------------------------------------------//
//
template <typename... _Types>
std::ostream&
operator<<(std::ostream& os, const std::tuple<_Types...>& p)
{
    constexpr size_t _N = sizeof...(_Types);
    tuple_printer(p, os, make_index_sequence<_N>{});
    return os;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::ostream&
operator<<(std::ostream& os, const std::vector<_Tp, _Extra...>& p)
{
    os << "(";
    for(size_t i = 0; i < p.size(); ++i)
        os << p.at(i) << ((i + 1 < p.size()) ? "," : "");
    os << ")";
    return os;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
std::ostream&
operator<<(std::ostream& os, const std::array<_Tp, _N>& p)
{
    os << "(";
    for(size_t i = 0; i < p.size(); ++i)
        os << p.at(i) << ((i + 1 < p.size()) ? "," : "");
    os << ")";
    return os;
}

}  // namespace ostream

//--------------------------------------------------------------------------------------//
//
//      operator += (same type)
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N, typename _Other>
std::array<_Tp, _N>&
operator+=(std::array<_Tp, _N>& lhs, _Other&& rhs)
{
    math::plus(lhs, std::forward<_Other>(rhs));
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename _Other>
std::pair<_Lhs, _Rhs>&
operator+=(std::pair<_Lhs, _Rhs>& lhs, _Other&& rhs)
{
    math::plus(lhs, std::forward<_Other>(rhs));
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra, typename _Other>
std::vector<_Tp, _Extra...>&
operator+=(std::vector<_Tp, _Extra...>& lhs, _Other&& rhs)
{
    math::plus(lhs, std::forward<_Other>(rhs));
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types, typename _Other>
std::tuple<_Types...>&
operator+=(std::tuple<_Types...>& lhs, _Other&& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator -= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
std::array<_Tp, _N>&
operator-=(std::array<_Tp, _N>& lhs, const std::array<_Tp, _N>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator-=(std::pair<_Lhs, _Rhs>& lhs, const std::pair<_Lhs, _Rhs>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator-=(std::vector<_Tp, _Extra...>& lhs, const std::vector<_Tp, _Extra...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator-=(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator *= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
std::array<_Tp, _N>&
operator*=(std::array<_Tp, _N>& lhs, const std::array<_Tp, _N>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator*=(std::pair<_Lhs, _Rhs>& lhs, const std::pair<_Lhs, _Rhs>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator*=(std::vector<_Tp, _Extra...>& lhs, const std::vector<_Tp, _Extra...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator*=(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator /= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
std::array<_Tp, _N>&
operator/=(std::array<_Tp, _N>& lhs, const std::array<_Tp, _N>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator/=(std::pair<_Lhs, _Rhs>& lhs, const std::pair<_Lhs, _Rhs>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator/=(std::vector<_Tp, _Extra...>& lhs, const std::vector<_Tp, _Extra...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator/=(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator *= (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename _Lhs, size_t _N, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::array<_Lhs, _N>&
operator*=(std::array<_Lhs, _N>& lhs, const _Rhs& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename _Arith,
          enable_if_t<(std::is_arithmetic<decay_t<_Arith>>::value), int>>
std::pair<_Lhs, _Rhs>&
operator*=(std::pair<_Lhs, _Rhs>& lhs, const _Arith& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::vector<_Lhs, _Extra...>&
operator*=(std::vector<_Lhs, _Extra...>& lhs, const _Rhs& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::tuple<_Lhs...>&
operator*=(std::tuple<_Lhs...>& lhs, const _Rhs& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator /= (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename _Lhs, size_t _N, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::array<_Lhs, _N>&
operator/=(std::array<_Lhs, _N>& lhs, const _Rhs& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename _Arith,
          enable_if_t<(std::is_arithmetic<decay_t<_Arith>>::value), int>>
std::pair<_Lhs, _Rhs>&
operator/=(std::pair<_Lhs, _Rhs>& lhs, const _Arith& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::vector<_Lhs, _Extra...>&
operator/=(std::vector<_Lhs, _Extra...>& lhs, const _Rhs& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::tuple<_Lhs...>&
operator/=(std::tuple<_Lhs...>& lhs, const _Rhs& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator * (fundamental)
//      operator / (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
_Lhs operator*(_Lhs lhs, const _Rhs& rhs)
{
    return (lhs *= rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
_Lhs
operator/(_Lhs lhs, const _Rhs& rhs)
{
    return (lhs /= rhs);
}

//--------------------------------------------------------------------------------------//

}  // namespace stl

using namespace stl;

//======================================================================================//

}  // namespace tim

//======================================================================================//

namespace std
{
#if defined(_WINDOWS)

template <typename _Lhs, typename _Rhs>
const pair<_Lhs, _Rhs>
operator-(pair<_Lhs, _Rhs> lhs, const pair<_Lhs, _Rhs>& rhs)
{
    ::tim::math::minus(lhs, rhs);
    return lhs;
}

template <typename... _Types>
const tuple<_Types...>
operator-(tuple<_Types...> lhs, const tuple<_Types...>& rhs)
{
    ::tim::math::minus(lhs, rhs);
    return lhs;
}

#endif

template <typename _Tp>
tuple<>&
operator+=(tuple<>& _lhs, const _Tp&)
{
    return _lhs;
}

}  // namespace std
