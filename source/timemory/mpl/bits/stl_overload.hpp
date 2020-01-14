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

namespace tim
{
/// \namespace tim::stl_overload
/// \brief the namespace is provided to hide stl overload from global namespace but
/// provide a method of using the namespace without a "using namespace tim;"
namespace stl_overload
{
namespace ostream
{
/// \namespace tim::stl_overload::ostream
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

template <typename _Tp, size_t _N>
std::array<_Tp, _N>&
operator+=(std::array<_Tp, _N>& lhs, const std::array<_Tp, _N>& rhs)
{
    array_math::plus(lhs, rhs, make_index_sequence<_N>{});
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator+=(std::pair<_Lhs, _Rhs>& lhs, const std::pair<_Lhs, _Rhs>& rhs)
{
    lhs.first += rhs.first;
    lhs.second += rhs.second;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator+=(std::vector<_Tp, _Extra...>& lhs, const std::vector<_Tp, _Extra...>& rhs)
{
    const auto _L = lhs.size();
    const auto _R = rhs.size();
    if(_L < _R)
        lhs.resize(_R, _Tp{});
    for(size_t i = 0; i < _R; ++i)
        lhs[i] += rhs[i];
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator+=(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    constexpr size_t _N = sizeof...(_Types);
    tuple_math::plus(lhs, rhs, make_index_sequence<_N>{});
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
    array_math::minus(lhs, rhs, make_index_sequence<_N>{});
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator-=(std::pair<_Lhs, _Rhs>& lhs, const std::pair<_Lhs, _Rhs>& rhs)
{
    lhs.first -= rhs.first;
    lhs.second -= rhs.second;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator-=(std::vector<_Tp, _Extra...>& lhs, const std::vector<_Tp, _Extra...>& rhs)
{
    const auto _N = std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < _N; ++i)
        lhs[i] -= rhs[i];
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator-=(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    constexpr size_t _N = sizeof...(_Types);
    tuple_math::minus(lhs, rhs, make_index_sequence<_N>{});
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
    array_math::multiply(lhs, rhs, make_index_sequence<_N>{});
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator*=(std::pair<_Lhs, _Rhs>& lhs, const std::pair<_Lhs, _Rhs>& rhs)
{
    lhs.first *= rhs.first;
    lhs.second *= rhs.second;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator*=(std::vector<_Tp, _Extra...>& lhs, const std::vector<_Tp, _Extra...>& rhs)
{
    const auto _N = std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < _N; ++i)
        lhs[i] *= rhs[i];
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator*=(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    constexpr size_t _N = sizeof...(_Types);
    tuple_math::multiply(lhs, rhs, make_index_sequence<_N>{});
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
    array_math::divide(lhs, rhs, make_index_sequence<_N>{});
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
std::pair<_Lhs, _Rhs>&
operator/=(std::pair<_Lhs, _Rhs>& lhs, const std::pair<_Lhs, _Rhs>& rhs)
{
    lhs.first /= rhs.first;
    lhs.second /= rhs.second;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
std::vector<_Tp, _Extra...>&
operator/=(std::vector<_Tp, _Extra...>& lhs, const std::vector<_Tp, _Extra...>& rhs)
{
    const auto _N = std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < _N; ++i)
        lhs[i] /= rhs[i];
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>&
operator/=(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    constexpr size_t _N = sizeof...(_Types);
    tuple_math::divide(lhs, rhs, make_index_sequence<_N>{});
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
    array_math::multiply(lhs, rhs, make_index_sequence<_N>{});
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename _Arith,
          enable_if_t<(std::is_arithmetic<decay_t<_Arith>>::value), int>>
std::pair<_Lhs, _Rhs>&
operator*=(std::pair<_Lhs, _Rhs>& lhs, const _Arith& rhs)
{
    lhs.first *= rhs;
    lhs.second *= rhs;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::vector<_Lhs, _Extra...>&
operator*=(std::vector<_Lhs, _Extra...>& lhs, const _Rhs& rhs)
{
    for(auto& itr : lhs)
        itr *= static_cast<_Lhs>(rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::tuple<_Lhs...>&
operator*=(std::tuple<_Lhs...>& lhs, const _Rhs& rhs)
{
    constexpr size_t _N = sizeof...(_Lhs);
    tuple_math::multiply(lhs, rhs, make_index_sequence<_N>{});
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
    array_math::divide(lhs, rhs, make_index_sequence<_N>{});
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename _Arith,
          enable_if_t<(std::is_arithmetic<decay_t<_Arith>>::value), int>>
std::pair<_Lhs, _Rhs>&
operator/=(std::pair<_Lhs, _Rhs>& lhs, const _Arith& rhs)
{
    lhs.first /= rhs;
    lhs.second /= rhs;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::vector<_Lhs, _Extra...>&
operator/=(std::vector<_Lhs, _Extra...>& lhs, const _Rhs& rhs)
{
    for(auto& itr : lhs)
        itr /= static_cast<_Lhs>(rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_arithmetic<decay_t<_Rhs>>::value), int>>
std::tuple<_Lhs...>&
operator/=(std::tuple<_Lhs...>& lhs, const _Rhs& rhs)
{
    constexpr size_t _N = sizeof...(_Lhs);
    tuple_math::divide(lhs, rhs, make_index_sequence<_N>{});
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

}  // namespace stl_overload

using namespace stl_overload;

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
    lhs.first -= rhs.first;
    lhs.second -= rhs.second;
    return lhs;
}

template <typename... _Types>
const tuple<_Types...>
operator-(tuple<_Types...> lhs, const tuple<_Types...>& rhs)
{
    constexpr size_t _N = sizeof...(_Types);
    ::tim::stl_overload::tuple_math::minus(lhs, rhs, ::tim::make_index_sequence<_N>{});
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
