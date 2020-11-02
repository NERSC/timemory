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

/** \file math.hpp
 * \headerfile math.hpp "timemory/mpl/math.hpp"
 * Provides the template meta-programming expansions for math operations
 *
 */

#pragma once

#include "timemory/mpl/types.hpp"
#include "timemory/utility/types.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

//======================================================================================//

namespace tim
{
namespace math
{
//--------------------------------------------------------------------------------------//

template <typename Tp>
bool
is_finite(const Tp& val)
{
#if defined(_WINDOWS)
    const Tp _infv = std::numeric_limits<Tp>::infinity();
    const Tp _inf  = (val < 0.0) ? -_infv : _infv;
    return (val == val && val != _inf);
#else
    return std::isfinite(val);
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline Tp abs(Tp);

template <typename Tp>
inline Tp sqrt(Tp);

template <typename Tp>
inline Tp
pow(Tp, double);

template <typename Tp>
inline Tp sqr(Tp);

template <typename Tp>
inline Tp
min(const Tp&, const Tp&);

template <typename Tp>
inline Tp
max(const Tp&, const Tp&);

template <typename Tp, typename Up = Tp>
inline void
assign(Tp&, Up&&);

template <typename Tp, typename Up = Tp>
inline Tp&
plus(Tp&, const Up&);

template <typename Tp, typename Up = Tp>
inline Tp&
minus(Tp&, const Up&);

template <typename Tp, typename Up = Tp>
inline void
multiply(Tp&, const Up&);

template <typename Tp, typename Up = Tp>
inline void
divide(Tp&, const Up&);

template <typename Tp>
inline Tp
percent_diff(const Tp&, const Tp&);

//--------------------------------------------------------------------------------------//
//              dummy overloads for std::tuple<>
//
inline std::tuple<> abs(std::tuple<>) { return std::tuple<>{}; }

inline std::tuple<> sqrt(std::tuple<>) { return std::tuple<>{}; }

inline std::tuple<>
pow(std::tuple<>, double)
{
    return std::tuple<>{};
}

inline std::tuple<> sqr(std::tuple<>) { return std::tuple<>{}; }

inline std::tuple<>
min(const std::tuple<>&, const std::tuple<>&)
{
    return std::tuple<>{};
}

inline std::tuple<>
max(const std::tuple<>&, const std::tuple<>&)
{
    return std::tuple<>{};
}

inline void
assign(std::tuple<>&, std::tuple<>&&)
{}

inline std::tuple<>&
plus(std::tuple<>& lhs, const std::tuple<>&)
{
    return lhs;
}

inline std::tuple<>&
minus(std::tuple<>& lhs, const std::tuple<>&)
{
    return lhs;
}

inline std::tuple<>&
multiply(std::tuple<>& lhs, const std::tuple<>&)
{
    return lhs;
}

inline std::tuple<>&
divide(std::tuple<>& lhs, const std::tuple<>&)
{
    return lhs;
}

inline std::tuple<>
percent_diff(const std::tuple<>&, const std::tuple<>&)
{
    return std::tuple<>{};
}

//--------------------------------------------------------------------------------------//
//
template <typename Up>
inline void
plus(std::tuple<>&, Up&&)
{}

template <typename Up>
inline void
minus(std::tuple<>&, Up&&)
{}

template <typename Up>
inline void
multiply(std::tuple<>&, Up&&)
{}

template <typename Up>
inline void
divide(std::tuple<>&, Up&&)
{}

inline void
divide(std::tuple<>&, const uint64_t&)
{}

inline void
divide(std::tuple<>&, const int64_t&)
{}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename std::enable_if<(std::is_integral<Tp>::value &&
                                                std::is_unsigned<Tp>::value),
                                               int>::type = 0>
auto
abs(Tp _val, std::tuple<>) -> decltype(Tp())
{
    return _val;
}

template <typename Tp,
          typename std::enable_if<(std::is_arithmetic<Tp>::value), int>::type = 0>
auto
abs(Tp _val, std::tuple<>) -> decltype(std::abs(_val), Tp())
{
    return std::abs(_val);
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
abs(Tp _val, std::tuple<>, ...) -> decltype(std::begin(_val), Tp())
{
    for(auto& itr : _val)
        itr = abs(itr, get_index_sequence<decay_t<decltype(itr)>>::value);
    return _val;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
abs(Tp _val, std::tuple<>) -> decltype(std::begin(_val), Tp())
{
    for(auto& itr : _val)
        itr.second =
            abs(itr.second, get_index_sequence<decay_t<decltype(itr.second)>>::value);
    return _val;
}

template <template <typename...> class Tuple, typename... Types, size_t... Idx>
auto
abs(Tuple<Types...> _val, index_sequence<Idx...>)
    -> decltype(std::get<0>(_val), Tuple<Types...>())
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(_val) = abs(std::get<Idx>(_val)));
    return _val;
}

template <typename Tp>
Tp
abs(Tp _val)
{
    return abs(_val, get_index_sequence<Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tp,
          typename std::enable_if<(std::is_arithmetic<Tp>::value), int>::type = 0>
auto
sqrt(Tp _val, std::tuple<>) -> decltype(std::sqrt(_val), Tp())
{
    return std::sqrt(_val);
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
sqrt(Tp _val, std::tuple<>, ...) -> decltype(std::begin(_val), Tp())
{
    for(auto& itr : _val)
        itr = sqrt(itr, get_index_sequence<decay_t<decltype(itr)>>::value);
    return _val;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
sqrt(Tp _val, std::tuple<>) -> decltype(std::begin(_val), Tp())
{
    for(auto& itr : _val)
        itr.second =
            sqrt(itr.second, get_index_sequence<decay_t<decltype(itr.second)>>::value);
    return _val;
}

template <template <typename...> class Tuple, typename... Types, size_t... Idx>
auto
sqrt(Tuple<Types...> _val, index_sequence<Idx...>)
    -> decltype(std::get<0>(_val), Tuple<Types...>())
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(_val) = sqrt(std::get<Idx>(_val)));
    return _val;
}

template <typename Tp>
Tp
sqrt(Tp _val)
{
    return sqrt(_val, get_index_sequence<Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tp,
          typename std::enable_if<(std::is_arithmetic<Tp>::value), int>::type = 0>
auto
pow(Tp _val, double _m, std::tuple<>) -> decltype(std::pow(_val, _m), Tp())
{
    return std::pow(_val, _m);
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
pow(Tp _val, double _m, std::tuple<>, ...) -> decltype(std::begin(_val), Tp())
{
    for(auto& itr : _val)
        itr = pow(itr, _m, get_index_sequence<decay_t<decltype(itr)>>::value);
    return _val;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
pow(Tp _val, double _m, std::tuple<>) -> decltype(std::begin(_val), Tp())
{
    for(auto& itr : _val)
        itr.second =
            pow(itr.second, _m, get_index_sequence<decay_t<decltype(itr.second)>>::value);
    return _val;
}

template <template <typename...> class Tuple, typename... Types, size_t... Idx>
auto
pow(Tuple<Types...> _val, double _m, index_sequence<Idx...>)
    -> decltype(std::get<0>(_val), Tuple<Types...>())
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(_val) = pow(std::get<Idx>(_val), _m));
    return _val;
}

template <typename Tp>
Tp
pow(Tp _val, double _m)
{
    return pow(_val, _m, get_index_sequence<Tp>::value);
}

template <typename Tp>
Tp
sqr(Tp _val)
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    return pow(_val, 2.0);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value, int> = 0>
Tp
min(Tp _lhs, Tp _rhs, std::tuple<>)
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    return (_rhs > _lhs) ? _lhs : _rhs;
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
min(const Tp& _lhs, const Tp& _rhs, std::tuple<>, ...) -> decltype(std::begin(_lhs), Tp())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = min(*litr, *ritr, get_index_sequence<decay_t<decltype(*itr)>>::value);
    }
    return _ret;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
min(const Tp& _lhs, const Tp& _rhs, std::tuple<>) -> decltype(std::begin(_lhs), Tp())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr   = std::begin(_lhs) + i;
        auto ritr   = std::begin(_rhs) + i;
        auto itr    = std::begin(_ret) + i;
        itr->second = min(litr->second, ritr->second,
                          get_index_sequence<decay_t<decltype(itr->second)>>::value);
    }
    return _ret;
}

template <typename Tp, size_t... Idx>
auto
min(const Tp& _lhs, const Tp& _rhs, index_sequence<Idx...>)
    -> decltype(std::get<0>(_lhs), Tp())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    Tp _ret{};
    TIMEMORY_FOLD_EXPRESSION(
        std::get<Idx>(_ret) =
            min(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
                get_index_sequence<decay_t<decltype(std::get<Idx>(_ret))>>::value));
    return _ret;
}

template <typename Tp>
Tp
min(const Tp& _lhs, const Tp& _rhs)
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    return min(_lhs, _rhs, get_index_sequence<Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value, int> = 0>
Tp
max(Tp _lhs, Tp _rhs, std::tuple<>)
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    return (_rhs < _lhs) ? _lhs : _rhs;
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
max(const Tp& _lhs, const Tp& _rhs, std::tuple<>, ...) -> decltype(std::begin(_lhs), Tp())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = max(*litr, *ritr, get_index_sequence<decay_t<decltype(*itr)>>::value);
    }
    return _ret;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
max(const Tp& _lhs, const Tp& _rhs, std::tuple<>) -> decltype(std::begin(_lhs), Tp())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr   = std::begin(_lhs) + i;
        auto ritr   = std::begin(_rhs) + i;
        auto itr    = std::begin(_ret) + i;
        itr->second = max(litr->second, ritr->second,
                          get_index_sequence<decay_t<decltype(itr->second)>>::value);
    }
    return _ret;
}

template <typename Tp, size_t... Idx>
auto
max(const Tp& _lhs, const Tp& _rhs, index_sequence<Idx...>)
    -> decltype(std::get<0>(_lhs), Tp())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    Tp _ret{};
    TIMEMORY_FOLD_EXPRESSION(
        std::get<Idx>(_ret) =
            max(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
                get_index_sequence<decay_t<decltype(std::get<Idx>(_ret))>>::value));
    return _ret;
}

template <typename Tp>
Tp
max(const Tp& _lhs, const Tp& _rhs)
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    return max(_lhs, _rhs, get_index_sequence<Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
inline void
assign(Tp& _lhs, Up&& _rhs)
{
    _lhs = std::forward<Up>(_rhs);
}

//--------------------------------------------------------------------------------------//

template <>
inline void
assign(std::tuple<>&, std::tuple<>&&)
{}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
auto
plus(Tp& _lhs, const Up& _rhs, std::tuple<>, ...) -> decltype(_lhs += _rhs, void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    _lhs += _rhs;
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type>
auto
plus(Tp& _lhs, const Up& _rhs, std::tuple<>, long) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        plus(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
plus(Tp& _lhs, const Up& _rhs, std::tuple<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
            continue;
        plus(litr->second, ritr->second,
             get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }

    for(auto ritr = std::begin(_rhs); ritr != std::end(_rhs); ++ritr)
    {
        auto litr = _lhs.find(ritr->first);
        if(litr == std::end(_lhs))
            continue;
        _lhs[ritr->first] = ritr->second;
    }
}

template <typename Tp, typename Up>
auto
plus(Tp&, const Up&, index_sequence<>, int)
{}

template <typename Tp, typename Up, size_t... Idx>
auto
plus(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    TIMEMORY_FOLD_EXPRESSION(
        plus(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
             get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up>
Tp&
plus(Tp& _lhs, const Up& _rhs)
{
    plus(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
    return _lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
auto
minus(Tp& _lhs, const Up& _rhs, std::tuple<>, ...) -> decltype(_lhs += _rhs, void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    _lhs -= _rhs;
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type>
auto
minus(Tp& _lhs, const Up& _rhs, std::tuple<>, long) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        minus(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
minus(Tp& _lhs, const Up& _rhs, std::tuple<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
            continue;
        minus(litr->second, ritr->second,
              get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up>
auto
minus(Tp&, const Up&, index_sequence<>, int)
{}

template <typename Tp, typename Up, size_t... Idx>
auto
minus(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    TIMEMORY_FOLD_EXPRESSION(
        minus(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
              get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up>
Tp&
minus(Tp& _lhs, const Up& _rhs)
{
    minus(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
    return _lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
auto
multiply(Tp& _lhs, Up _rhs, std::tuple<>, ...) -> decltype(_lhs *= _rhs, void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    _lhs *= _rhs;
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type>
auto
multiply(Tp& _lhs, const Up& _rhs, std::tuple<>, long)
    -> decltype((std::begin(_lhs), std::begin(_rhs)), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        multiply(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type,
          enable_if_t<std::is_arithmetic<Up>::value, int> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, std::tuple<>, long)
    -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_lhs);
    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        multiply(*litr, _rhs, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp                                      = typename Tp::mapped_type,
          enable_if_t<!std::is_arithmetic<Up>::value, int> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, std::tuple<>, int)
    -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
            continue;
        multiply(litr->second, ritr->second,
                 get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp                                     = typename Tp::mapped_type,
          enable_if_t<std::is_arithmetic<Up>::value, int> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, std::tuple<>, int)
    -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        multiply(litr->second, _rhs,
                 get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up>
auto
multiply(Tp&, const Up&, index_sequence<>, int)
{}

template <typename Tp, typename Up, size_t... Idx,
          enable_if_t<!std::is_arithmetic<Up>::value, int> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    TIMEMORY_FOLD_EXPRESSION(
        multiply(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
                 get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up, size_t... Idx,
          enable_if_t<std::is_arithmetic<Up>::value, int> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    TIMEMORY_FOLD_EXPRESSION(
        multiply(std::get<Idx>(_lhs), _rhs,
                 get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up>
void
multiply(Tp& _lhs, const Up& _rhs)
{
    multiply(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
auto
divide(Tp& _lhs, Up _rhs, std::tuple<>, ...) -> decltype(_lhs /= _rhs, void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    _lhs /= _rhs;
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type,
          enable_if_t<!std::is_arithmetic<Up>::value, int> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, std::tuple<>, long) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        divide(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type,
          enable_if_t<std::is_arithmetic<Up>::value, int> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, std::tuple<>, long) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_lhs);
    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        divide(*litr, _rhs, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp                                      = typename Tp::mapped_type,
          enable_if_t<!std::is_arithmetic<Up>::value, int> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, std::tuple<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
            continue;
        divide(litr->second, ritr->second,
               get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp                                     = typename Tp::mapped_type,
          enable_if_t<std::is_arithmetic<Up>::value, int> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, std::tuple<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        divide(litr->second, _rhs,
               get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up>
auto
divide(Tp&, const Up&, index_sequence<>, int)
{}

template <typename Tp, typename Up, size_t... Idx,
          enable_if_t<!std::is_arithmetic<Up>::value, int> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    TIMEMORY_FOLD_EXPRESSION(
        divide(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
               get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up, size_t... Idx,
          enable_if_t<std::is_arithmetic<Up>::value, int> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<Tp>, std::tuple<>>::value, "Error! tuple<>");
    TIMEMORY_FOLD_EXPRESSION(
        divide(std::get<Idx>(_lhs), _rhs,
               get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up>
void
divide(Tp& _lhs, const Up& _rhs)
{
    divide(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value, int> = 0>
Tp
percent_diff(Tp _lhs, Tp _rhs, std::tuple<>, ...)
{
    constexpr Tp _zero    = Tp(0.0);
    constexpr Tp _one     = Tp(1.0);
    constexpr Tp _hundred = Tp(100.0);
    Tp&&         _pdiff   = (_rhs > _zero) ? ((_one - (_lhs / _rhs)) * _hundred) : _zero;
    return (_pdiff < _zero) ? _zero : _pdiff;
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
percent_diff(const Tp& _lhs, const Tp& _rhs, std::tuple<>, long)
    -> decltype(std::begin(_lhs), Tp())
{
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    // initialize
    for(auto& itr : _ret)
        itr = Vp{};

    // compute
    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = percent_diff(*litr, *ritr,
                            get_index_sequence<decay_t<decltype(*itr)>>::value);
    }
    return _ret;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
percent_diff(const Tp& _lhs, const Tp& _rhs, std::tuple<>, int)
    -> decltype(std::begin(_lhs), Tp())
{
    assert(_lhs.size() == _rhs.size());
    Tp _ret{};
    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
        {
            _ret[litr->first] = Mp{};
        }
        else
        {
            _ret[litr->first] =
                percent_diff(litr->second, ritr->second,
                             get_index_sequence<decay_t<decltype(litr->second)>>::value);
        }
    }
    return _ret;
}

template <typename Tp, size_t... Idx>
auto
percent_diff(const Tp& _lhs, const Tp& _rhs, index_sequence<Idx...>, ...)
    -> decltype(std::get<0>(_lhs), Tp())
{
    Tp _ret{};
    TIMEMORY_FOLD_EXPRESSION(
        std::get<Idx>(_ret) = percent_diff(
            std::get<Idx>(_lhs), std::get<Idx>(_rhs),
            get_index_sequence<decay_t<decltype(std::get<Idx>(_ret))>>::value));
    return _ret;
}

template <typename Tp>
Tp
percent_diff(const Tp& _lhs, const Tp& _rhs)
{
    return percent_diff(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up = Tp>
struct compute
{
    using type       = Tp;
    using value_type = Up;

    static type abs(const type& _v) { return ::tim::math::abs(_v); }
    static type sqr(const type& _v) { return ::tim::math::sqr(_v); }
    static type sqrt(const type& _v) { return ::tim::math::sqrt(_v); }
    static type min(const type& _l, const type& _r) { return ::tim::math::min(_l, _r); }
    static type max(const type& _l, const type& _r) { return ::tim::math::max(_l, _r); }
    static type percent_diff(const type& _l, const type& _r)
    {
        return ::tim::math::percent_diff(_l, _r);
    }

    template <typename V = value_type>
    static void plus(type& _l, const V& _r)
    {
        ::tim::math::plus(_l, _r);
    }

    template <typename V = value_type>
    static void minus(type& _l, const V& _r)
    {
        ::tim::math::minus(_l, _r);
    }

    template <typename V = value_type>
    static void multiply(type& _l, const V& _r)
    {
        ::tim::math::multiply(_l, _r);
    }

    template <typename V = value_type>
    static void divide(type& _l, const V& _r)
    {
        ::tim::math::divide(_l, _r);
    }
};

//--------------------------------------------------------------------------------------//
/// \struct tim::math::compute<std::tuple<>>
/// \brief this specialization exists for statistics<tuple<>> which is the default
/// type when statistics have not been enabled
///
template <>
struct compute<std::tuple<>>
{
    using type = std::tuple<>;

    static type abs(const type&) { return type{}; }
    static type sqr(const type&) { return type{}; }
    static type sqrt(const type&) { return type{}; }
    static type max(const type&, const type&) { return type{}; }
    static type min(const type&, const type&) { return type{}; }
    static type percent_diff(const type&, const type&) { return type{}; }

    static void plus(type&, const type&) {}
    static void minus(type&, const type&) {}
    static void multiply(type&, const type&) {}
    static void divide(type&, const type&) {}
};

//--------------------------------------------------------------------------------------//

}  // namespace math
}  // namespace tim
