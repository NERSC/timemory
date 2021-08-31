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

#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/types.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

namespace tim
{
namespace math
{
template <typename Tp>
bool
is_finite(const Tp& val)
{
#if defined(TIMEMORY_WINDOWS)
    const Tp _infv = std::numeric_limits<Tp>::infinity();
    const Tp _inf  = (val < 0.0) ? -_infv : _infv;
    return (val == val && val != _inf);
#else
    return std::isfinite(val);
#endif
}

template <typename Tp>
TIMEMORY_INLINE Tp abs(Tp);

template <typename Tp>
TIMEMORY_INLINE Tp sqrt(Tp);

template <typename Tp>
TIMEMORY_INLINE Tp
pow(Tp, double);

template <typename Tp>
TIMEMORY_INLINE Tp sqr(Tp);

template <typename Tp>
TIMEMORY_INLINE Tp
min(const Tp&, const Tp&);

template <typename Tp>
TIMEMORY_INLINE Tp
max(const Tp&, const Tp&);

template <typename Tp, typename Up = Tp>
TIMEMORY_INLINE void
assign(Tp&, Up&&);

template <typename Tp, typename Up = Tp,
          enable_if_t<!concepts::is_null_type<Tp>::value> = 0>
TIMEMORY_INLINE Tp&
                plus(Tp&, const Up&);

template <typename Tp, typename Up = Tp,
          enable_if_t<!concepts::is_null_type<Tp>::value> = 0>
TIMEMORY_INLINE Tp&
                minus(Tp&, const Up&);

template <typename Tp, typename Up = Tp,
          enable_if_t<!concepts::is_null_type<Tp>::value> = 0>
TIMEMORY_INLINE Tp&
                multiply(Tp&, const Up&);

template <typename Tp, typename Up = Tp,
          enable_if_t<!concepts::is_null_type<Tp>::value> = 0>
TIMEMORY_INLINE Tp&
                divide(Tp&, const Up&);

template <typename Tp>
TIMEMORY_INLINE Tp
percent_diff(const Tp&, const Tp&);

}  // namespace math

inline namespace stl
{
//--------------------------------------------------------------------------------------//
//
//      operator +=
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N, typename Other>
std::array<Tp, N>&
operator+=(std::array<Tp, N>&, Other&&);

template <typename Lhs, typename Rhs, typename Other>
std::pair<Lhs, Rhs>&
operator+=(std::pair<Lhs, Rhs>&, Other&&);

template <typename Tp, typename... _Extra, typename Other>
std::vector<Tp, _Extra...>&
operator+=(std::vector<Tp, _Extra...>&, Other&&);

template <typename... Types, typename Other>
std::tuple<Types...>&
operator+=(std::tuple<Types...>&, Other&&);

//--------------------------------------------------------------------------------------//
//
//      operator -=
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator-=(std::array<Tp, N>&, const std::array<Tp, N>&);

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::array<Lhs, N>&
operator-=(std::array<Lhs, N>&, const Rhs&);

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator-=(std::pair<Lhs, Rhs>&, const std::pair<Lhs, Rhs>&);

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int> = 0>
std::pair<Lhs, Rhs>&
operator-=(std::pair<Lhs, Rhs>&, const ArithT&);

template <typename Tp, typename... _Extra>
std::vector<Tp, _Extra...>&
operator-=(std::vector<Tp, _Extra...>&, const std::vector<Tp, _Extra...>&);

template <typename Lhs, typename Rhs, typename... _Extra,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::vector<Lhs, _Extra...>&
operator-=(std::vector<Lhs, _Extra...>&, const Rhs&);

template <typename... Types>
std::tuple<Types...>&
operator-=(std::tuple<Types...>&, const std::tuple<Types...>&);

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::tuple<Lhs...>&
operator-=(std::tuple<Lhs...>&, const Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator *=
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator*=(std::array<Tp, N>&, const std::array<Tp, N>&);

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::array<Lhs, N>&
operator*=(std::array<Lhs, N>&, const Rhs&);

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator*=(std::pair<Lhs, Rhs>&, const std::pair<Lhs, Rhs>&);

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int> = 0>
std::pair<Lhs, Rhs>&
operator*=(std::pair<Lhs, Rhs>&, const ArithT&);

template <typename Tp, typename... _Extra>
std::vector<Tp, _Extra...>&
operator*=(std::vector<Tp, _Extra...>&, const std::vector<Tp, _Extra...>&);

template <typename Lhs, typename Rhs, typename... _Extra,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::vector<Lhs, _Extra...>&
operator*=(std::vector<Lhs, _Extra...>&, const Rhs&);

template <typename... Types>
std::tuple<Types...>&
operator*=(std::tuple<Types...>&, const std::tuple<Types...>&);

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::tuple<Lhs...>&
operator*=(std::tuple<Lhs...>&, const Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator /=
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator/=(std::array<Tp, N>&, const std::array<Tp, N>&);

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::array<Lhs, N>&
operator/=(std::array<Lhs, N>&, const Rhs&);

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator/=(std::pair<Lhs, Rhs>&, const std::pair<Lhs, Rhs>&);

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int> = 0>
std::pair<Lhs, Rhs>&
operator/=(std::pair<Lhs, Rhs>&, const ArithT&);

template <typename Tp, typename... _Extra>
std::vector<Tp, _Extra...>&
operator/=(std::vector<Tp, _Extra...>&, const std::vector<Tp, _Extra...>&);

template <typename Lhs, typename Rhs, typename... _Extra,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::vector<Lhs, _Extra...>&
operator/=(std::vector<Lhs, _Extra...>&, const Rhs&);

template <typename... Types>
std::tuple<Types...>&
operator/=(std::tuple<Types...>&, const std::tuple<Types...>&);

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::tuple<Lhs...>&
operator/=(std::tuple<Lhs...>&, const Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator * (fundamental)
//      operator / (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
Lhs operator*(Lhs, const Rhs&);

template <typename Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
Lhs
operator/(Lhs, const Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator +
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>
operator+(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs);

template <typename... Types>
std::tuple<Types...>
operator+(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs);

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>
operator+(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs);

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>
operator+(std::vector<Tp, Extra...> lhs, const std::vector<Tp, Extra...>& rhs);

//--------------------------------------------------------------------------------------//
//
//      operator -
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>
operator-(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs);

template <typename... Types>
std::tuple<Types...>
operator-(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs);

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>
operator-(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs);

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>
operator-(std::vector<Tp, Extra...> lhs, const std::vector<Tp, Extra...>& rhs);

//--------------------------------------------------------------------------------------//
//
//      operator *
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N> operator*(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs);

template <typename... Types>
std::tuple<Types...> operator*(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs);

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs> operator*(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs);

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...> operator*(std::vector<Tp, Extra...>        lhs,
                                    const std::vector<Tp, Extra...>& rhs);

//--------------------------------------------------------------------------------------//
//
//      operator /
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>
operator/(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs);

template <typename... Types>
std::tuple<Types...>
operator/(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs);

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>
operator/(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs);

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>
operator/(std::vector<Tp, Extra...> lhs, const std::vector<Tp, Extra...>& rhs);

}  // namespace stl
}  // namespace tim

//--------------------------------------------------------------------------------------//
//              dummy overloads for std::tuple<>, type_list<>, null_type
//
#define TIMEMORY_MATH_NULL_TYPE_OVERLOAD(TYPE)                                           \
    namespace tim                                                                        \
    {                                                                                    \
    namespace math                                                                       \
    {                                                                                    \
    TIMEMORY_INLINE TYPE abs(TYPE) { return TYPE{}; }                                    \
    TIMEMORY_INLINE TYPE sqrt(TYPE) { return TYPE{}; }                                   \
    TIMEMORY_INLINE TYPE pow(TYPE, double) { return TYPE{}; }                            \
    TIMEMORY_INLINE TYPE sqr(TYPE) { return TYPE{}; }                                    \
    TIMEMORY_INLINE TYPE min(const TYPE&, const TYPE&) { return TYPE{}; }                \
    TIMEMORY_INLINE TYPE max(const TYPE&, const TYPE&) { return TYPE{}; }                \
    TIMEMORY_INLINE void assign(TYPE&, TYPE&&) {}                                        \
    TIMEMORY_INLINE TYPE& plus(TYPE& lhs, const TYPE&) { return lhs; }                   \
    TIMEMORY_INLINE TYPE& minus(TYPE& lhs, const TYPE&) { return lhs; }                  \
    TIMEMORY_INLINE TYPE& multiply(TYPE& lhs, const TYPE&) { return lhs; }               \
    TIMEMORY_INLINE TYPE& divide(TYPE& lhs, const TYPE&) { return lhs; }                 \
    TIMEMORY_INLINE TYPE  percent_diff(const TYPE&, const TYPE&) { return TYPE{}; }      \
    template <typename Up>                                                               \
    TIMEMORY_INLINE TYPE& plus(TYPE& lhs, Up&&)                                          \
    {                                                                                    \
        return lhs;                                                                      \
    }                                                                                    \
    template <typename Up>                                                               \
    TIMEMORY_INLINE TYPE& minus(TYPE& lhs, Up&&)                                         \
    {                                                                                    \
        return lhs;                                                                      \
    }                                                                                    \
    template <typename Up>                                                               \
    TIMEMORY_INLINE TYPE& multiply(TYPE& lhs, Up&&)                                      \
    {                                                                                    \
        return lhs;                                                                      \
    }                                                                                    \
    template <typename Up>                                                               \
    TIMEMORY_INLINE TYPE& divide(TYPE& lhs, Up&&)                                        \
    {                                                                                    \
        return lhs;                                                                      \
    }                                                                                    \
    TIMEMORY_INLINE TYPE& divide(TYPE& lhs, uint64_t) { return lhs; }                    \
    TIMEMORY_INLINE TYPE& divide(TYPE& lhs, int64_t) { return lhs; }                     \
    }                                                                                    \
    }

TIMEMORY_MATH_NULL_TYPE_OVERLOAD(std::tuple<>)
TIMEMORY_MATH_NULL_TYPE_OVERLOAD(null_type)
TIMEMORY_MATH_NULL_TYPE_OVERLOAD(type_list<>)

#if defined(TIMEMORY_WINDOWS)
namespace std
{
template <typename Lhs, typename Rhs>
const pair<Lhs, Rhs>
operator-(pair<Lhs, Rhs>, const pair<Lhs, Rhs>&);

template <typename... Types>
const tuple<Types...>
operator-(tuple<Types...>, const tuple<Types...>&);

}  // namespace std
#endif
