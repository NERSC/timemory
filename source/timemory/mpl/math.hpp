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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>
#include <vector>

#include "timemory/mpl/types.hpp"

//======================================================================================//

namespace tim
{
namespace math
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
bool
is_finite(const _Tp& val)
{
#if defined(_WINDOWS)
    const _Tp _infv = std::numeric_limits<_Tp>::infinity();
    const _Tp _inf  = (val < 0.0) ? -_infv : _infv;
    return (val == val && val != _inf);
#else
    return std::isfinite(val);
#endif
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp abs(_Tp);

template <typename _Tp>
_Tp sqrt(_Tp);

template <typename _Tp>
_Tp
pow(_Tp, double);

template <typename _Tp>
_Tp sqr(_Tp);

template <typename _Tp>
_Tp sqr(_Tp);

template <typename _Tp>
_Tp
min(const _Tp&, const _Tp&);

template <typename _Tp>
_Tp
max(const _Tp&, const _Tp&);

template <typename _Tp, typename _Up = _Tp>
void
plus(_Tp&, const _Up&);

template <typename _Tp, typename _Up = _Tp>
void
minus(_Tp&, const _Up&);

template <typename _Tp, typename _Up = _Tp>
void
multiply(_Tp&, const _Up&);

template <typename _Tp, typename _Up = _Tp>
void
divide(_Tp&, const _Up&);

template <typename _Tp>
_Tp
percent_diff(const _Tp&, const _Tp&);

//--------------------------------------------------------------------------------------//

template <typename _Tp,
          typename std::enable_if<(std::is_arithmetic<_Tp>::value), int>::type = 0>
auto
abs(_Tp _val, std::tuple<>) -> decltype(std::abs(_val), _Tp())
{
    return std::abs(_val);
}

template <typename _Tp, typename _Vp = typename _Tp::value_type>
auto
abs(_Tp _val, std::tuple<>, ...) -> decltype(std::begin(_val), _Tp())
{
    for(auto& itr : _val)
        itr = abs(itr, get_index_sequence<decay_t<decltype(itr)>>::value);
    return _val;
}

template <typename _Tp, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
abs(_Tp _val, std::tuple<>) -> decltype(std::begin(_val), _Tp())
{
    for(auto& itr : _val)
        itr.second =
            abs(itr.second, get_index_sequence<decay_t<decltype(itr.second)>>::value);
    return _val;
}

template <template <typename...> class _Tuple, typename... _Types, size_t... _Idx>
auto
abs(_Tuple<_Types...> _val, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_val), _Tuple<_Types...>())
{
    using init_list_t = std::initializer_list<int>;
    auto&& tmp =
        init_list_t({ (std::get<_Idx>(_val) = abs(std::get<_Idx>(_val)), 0)... });
    consume_parameters(tmp);
    return _val;
}

template <typename _Tp>
_Tp
abs(_Tp _val)
{
    return abs(_val, get_index_sequence<_Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp,
          typename std::enable_if<(std::is_arithmetic<_Tp>::value), int>::type = 0>
auto
sqrt(_Tp _val, std::tuple<>) -> decltype(std::sqrt(_val), _Tp())
{
    return std::sqrt(_val);
}

template <typename _Tp, typename _Vp = typename _Tp::value_type>
auto
sqrt(_Tp _val, std::tuple<>, ...) -> decltype(std::begin(_val), _Tp())
{
    for(auto& itr : _val)
        itr = sqrt(itr, get_index_sequence<decay_t<decltype(itr)>>::value);
    return _val;
}

template <typename _Tp, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
sqrt(_Tp _val, std::tuple<>) -> decltype(std::begin(_val), _Tp())
{
    for(auto& itr : _val)
        itr.second =
            sqrt(itr.second, get_index_sequence<decay_t<decltype(itr.second)>>::value);
    return _val;
}

template <template <typename...> class _Tuple, typename... _Types, size_t... _Idx>
auto
sqrt(_Tuple<_Types...> _val, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_val), _Tuple<_Types...>())
{
    using init_list_t = std::initializer_list<int>;
    auto&& tmp =
        init_list_t({ (std::get<_Idx>(_val) = sqrt(std::get<_Idx>(_val)), 0)... });
    consume_parameters(tmp);
    return _val;
}

template <typename _Tp>
_Tp
sqrt(_Tp _val)
{
    return sqrt(_val, get_index_sequence<_Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp,
          typename std::enable_if<(std::is_arithmetic<_Tp>::value), int>::type = 0>
auto
pow(_Tp _val, double _m, std::tuple<>) -> decltype(std::pow(_val, _m), _Tp())
{
    return std::pow(_val, _m);
}

template <typename _Tp, typename _Vp = typename _Tp::value_type>
auto
pow(_Tp _val, double _m, std::tuple<>, ...) -> decltype(std::begin(_val), _Tp())
{
    for(auto& itr : _val)
        itr = pow(itr, _m, get_index_sequence<decay_t<decltype(itr)>>::value);
    return _val;
}

template <typename _Tp, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
pow(_Tp _val, double _m, std::tuple<>) -> decltype(std::begin(_val), _Tp())
{
    for(auto& itr : _val)
        itr.second =
            pow(itr.second, _m, get_index_sequence<decay_t<decltype(itr.second)>>::value);
    return _val;
}

template <template <typename...> class _Tuple, typename... _Types, size_t... _Idx>
auto
pow(_Tuple<_Types...> _val, double _m, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_val), _Tuple<_Types...>())
{
    using init_list_t = std::initializer_list<int>;
    auto&& tmp =
        init_list_t({ (std::get<_Idx>(_val) = pow(std::get<_Idx>(_val), _m), 0)... });
    consume_parameters(tmp);
    return _val;
}

template <typename _Tp>
_Tp
pow(_Tp _val, double _m)
{
    return pow(_val, _m, get_index_sequence<_Tp>::value);
}

template <typename _Tp>
_Tp
sqr(_Tp _val)
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    return pow(_val, 2.0);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, enable_if_t<(std::is_arithmetic<_Tp>::value), int> = 0>
_Tp
min(_Tp _lhs, _Tp _rhs, std::tuple<>)
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    return (_rhs > _lhs) ? _lhs : _rhs;
}

template <typename _Tp, typename _Vp = typename _Tp::value_type>
auto
min(const _Tp& _lhs, const _Tp& _rhs, std::tuple<>, ...)
    -> decltype(std::begin(_lhs), _Tp())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    _Tp  _ret{};
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

template <typename _Tp, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
min(const _Tp& _lhs, const _Tp& _rhs, std::tuple<>) -> decltype(std::begin(_lhs), _Tp())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    _Tp  _ret{};
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

template <typename _Tp, size_t... _Idx>
auto
min(const _Tp& _lhs, const _Tp& _rhs, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_lhs), _Tp())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    _Tp _ret{};
    using init_list_t = std::initializer_list<int>;
    auto&& tmp        = init_list_t(
        { (std::get<_Idx>(_ret) =
               min(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs),
                   get_index_sequence<decay_t<decltype(std::get<_Idx>(_ret))>>::value),
           0)... });
    consume_parameters(tmp);
    return _ret;
}

template <typename _Tp>
_Tp
min(const _Tp& _lhs, const _Tp& _rhs)
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    return min(_lhs, _rhs, get_index_sequence<_Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, enable_if_t<(std::is_arithmetic<_Tp>::value), int> = 0>
_Tp
max(_Tp _lhs, _Tp _rhs, std::tuple<>)
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    return (_rhs < _lhs) ? _lhs : _rhs;
}

template <typename _Tp, typename _Vp = typename _Tp::value_type>
auto
max(const _Tp& _lhs, const _Tp& _rhs, std::tuple<>, ...)
    -> decltype(std::begin(_lhs), _Tp())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    _Tp  _ret{};
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

template <typename _Tp, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
max(const _Tp& _lhs, const _Tp& _rhs, std::tuple<>) -> decltype(std::begin(_lhs), _Tp())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    _Tp  _ret{};
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

template <typename _Tp, size_t... _Idx>
auto
max(const _Tp& _lhs, const _Tp& _rhs, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_lhs), _Tp())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    _Tp _ret{};
    using init_list_t = std::initializer_list<int>;
    auto&& tmp        = init_list_t(
        { (std::get<_Idx>(_ret) =
               max(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs),
                   get_index_sequence<decay_t<decltype(std::get<_Idx>(_ret))>>::value),
           0)... });
    consume_parameters(tmp);
    return _ret;
}

template <typename _Tp>
_Tp
max(const _Tp& _lhs, const _Tp& _rhs)
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    return max(_lhs, _rhs, get_index_sequence<_Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Up,
          enable_if_t<(std::is_arithmetic<_Tp>::value), int> = 0>
void
plus(_Tp& _lhs, const _Up& _rhs, std::tuple<>)
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    _lhs += _rhs;
}

template <typename _Tp, typename _Up, typename _Vp = typename _Tp::value_type>
auto
plus(_Tp& _lhs, const _Up& _rhs, std::tuple<>, ...) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        plus(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value);
    }
}

template <typename _Tp, typename _Up, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
plus(_Tp& _lhs, const _Up& _rhs, std::tuple<>) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = std::find(std::begin(_rhs), std::end(_rhs), litr->first);
        if(ritr == std::end(_rhs))
            continue;
        plus(litr->second, ritr->second,
             get_index_sequence<decay_t<decltype(litr->second)>>::value);
    }

    for(auto ritr = std::begin(_rhs); ritr != std::end(_rhs); ++ritr)
    {
        auto litr = std::find(std::begin(_lhs), std::end(_lhs), ritr->first);
        if(litr == std::end(_lhs))
            continue;
        _lhs[ritr->first] = ritr->second;
    }
}

template <typename _Tp, typename _Up, size_t... _Idx>
auto
plus(_Tp& _lhs, const _Up& _rhs, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    using init_list_t = std::initializer_list<int>;
    auto&& tmp        = init_list_t(
        { (plus(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs),
                get_index_sequence<decay_t<decltype(std::get<_Idx>(_lhs))>>::value),
           0)... });
    consume_parameters(tmp);
}

template <typename _Tp, typename _Up>
void
plus(_Tp& _lhs, const _Up& _rhs)
{
    plus(_lhs, _rhs, get_index_sequence<_Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Up,
          enable_if_t<(std::is_arithmetic<_Tp>::value), int> = 0>
void
minus(_Tp& _lhs, const _Up& _rhs, std::tuple<>)
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    _lhs -= _rhs;
}

template <typename _Tp, typename _Up, typename _Vp = typename _Tp::value_type>
auto
minus(_Tp& _lhs, const _Up& _rhs, std::tuple<>, ...) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        minus(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value);
    }
}

template <typename _Tp, typename _Up, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
minus(_Tp& _lhs, const _Up& _rhs, std::tuple<>) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = std::find(std::begin(_rhs), std::end(_rhs), litr->first);
        if(ritr == std::end(_rhs))
            continue;
        minus(litr->second, ritr->second,
              get_index_sequence<decay_t<decltype(litr->second)>>::value);
    }
}

template <typename _Tp, typename _Up, size_t... _Idx>
auto
minus(_Tp& _lhs, const _Up& _rhs, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    using init_list_t = std::initializer_list<int>;
    auto&& tmp        = init_list_t(
        { (minus(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs),
                 get_index_sequence<decay_t<decltype(std::get<_Idx>(_lhs))>>::value),
           0)... });
    consume_parameters(tmp);
}

template <typename _Tp, typename _Up>
void
minus(_Tp& _lhs, const _Up& _rhs)
{
    minus(_lhs, _rhs, get_index_sequence<_Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Up,
          enable_if_t<(std::is_arithmetic<_Tp>::value), int> = 0>
void
multiply(_Tp& _lhs, _Up _rhs, std::tuple<>)
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    _lhs *= _rhs;
}

template <typename _Tp, typename _Up, typename _Vp = typename _Tp::value_type>
auto
multiply(_Tp& _lhs, const _Up& _rhs, std::tuple<>, ...)
    -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        multiply(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value);
    }
}

template <typename _Tp, typename _Up, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
multiply(_Tp& _lhs, const _Up& _rhs, std::tuple<>) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = std::find(std::begin(_rhs), std::end(_rhs), litr->first);
        if(ritr == std::end(_rhs))
            continue;
        multiply(litr->second, ritr->second,
                 get_index_sequence<decay_t<decltype(litr->second)>>::value);
    }
}

template <typename _Tp, typename _Up, size_t... _Idx>
auto
multiply(_Tp& _lhs, const _Up& _rhs, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    using init_list_t = std::initializer_list<int>;
    auto&& tmp        = init_list_t(
        { (multiply(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs),
                    get_index_sequence<decay_t<decltype(std::get<_Idx>(_lhs))>>::value),
           0)... });
    consume_parameters(tmp);
}

template <typename _Tp, typename _Up>
void
multiply(_Tp& _lhs, const _Up& _rhs)
{
    multiply(_lhs, _rhs, get_index_sequence<_Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Up,
          enable_if_t<(std::is_arithmetic<_Tp>::value), int> = 0>
void
divide(_Tp& _lhs, _Up _rhs, std::tuple<>)
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    _lhs *= _rhs;
}

template <typename _Tp, typename _Up, typename _Vp = typename _Tp::value_type>
auto
divide(_Tp& _lhs, const _Up& _rhs, std::tuple<>, ...)
    -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        divide(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value);
    }
}

template <typename _Tp, typename _Up, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
divide(_Tp& _lhs, const _Up& _rhs, std::tuple<>) -> decltype(std::begin(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = std::find(std::begin(_rhs), std::end(_rhs), litr->first);
        if(ritr == std::end(_rhs))
            continue;
        divide(litr->second, ritr->second,
               get_index_sequence<decay_t<decltype(litr->second)>>::value);
    }
}

template <typename _Tp, typename _Up, size_t... _Idx>
auto
divide(_Tp& _lhs, const _Up& _rhs, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!std::is_same<decay_t<_Tp>, std::tuple<>>::value, "Error! tuple<>");
    using init_list_t = std::initializer_list<int>;
    auto&& tmp        = init_list_t(
        { (divide(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs),
                  get_index_sequence<decay_t<decltype(std::get<_Idx>(_lhs))>>::value),
           0)... });
    consume_parameters(tmp);
}

template <typename _Tp, typename _Up>
void
divide(_Tp& _lhs, const _Up& _rhs)
{
    divide(_lhs, _rhs, get_index_sequence<_Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, enable_if_t<(std::is_arithmetic<_Tp>::value), int> = 0>
_Tp
percent_diff(_Tp _lhs, _Tp _rhs, std::tuple<>)
{
    constexpr _Tp _zero    = _Tp(0.0);
    constexpr _Tp _one     = _Tp(1.0);
    constexpr _Tp _hundred = _Tp(100.0);
    _Tp&&         _pdiff   = (_rhs > _zero) ? ((_one - (_lhs / _rhs)) * _hundred) : _zero;
    return (_pdiff < _zero) ? _zero : _pdiff;
}

template <typename _Tp, typename _Vp = typename _Tp::value_type>
auto
percent_diff(const _Tp& _lhs, const _Tp& _rhs, std::tuple<>, ...)
    -> decltype(std::begin(_lhs), _Tp())
{
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);

    auto _n = std::min<Int_t>(_nl, _nr);
    _Tp  _ret{};
    mpl::resize(_ret, _n);

    // initialize
    for(auto& itr : _ret)
        itr = _Vp{};

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

template <typename _Tp, typename _Kp = typename _Tp::key_type,
          typename _Mp = typename _Tp::mapped_type>
auto
percent_diff(const _Tp& _lhs, const _Tp& _rhs, std::tuple<>)
    -> decltype(std::begin(_lhs), _Tp())
{
    assert(_lhs.size() == _rhs.size());
    _Tp _ret{};
    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = std::find(std::begin(_rhs), std::end(_rhs), litr->first);
        if(ritr == std::end(_rhs))
        {
            _ret[litr->second] = _Mp{};
        }
        else
        {
            _ret[litr->second] =
                percent_diff(litr->second, ritr->second,
                             get_index_sequence<decay_t<decltype(litr->second)>>::value);
        }
    }
    return _ret;
}

template <typename _Tp, size_t... _Idx>
auto
percent_diff(const _Tp& _lhs, const _Tp& _rhs, index_sequence<_Idx...>)
    -> decltype(std::get<0>(_lhs), _Tp())
{
    _Tp _ret{};
    using init_list_t = std::initializer_list<int>;
    auto&& tmp        = init_list_t(
        { (std::get<_Idx>(_ret) = percent_diff(
               std::get<_Idx>(_lhs), std::get<_Idx>(_rhs),
               get_index_sequence<decay_t<decltype(std::get<_Idx>(_ret))>>::value),
           0)... });
    consume_parameters(tmp);
    return _ret;
}

template <typename _Tp>
_Tp
percent_diff(const _Tp& _lhs, const _Tp& _rhs)
{
    return percent_diff(_lhs, _rhs, get_index_sequence<_Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Up = _Tp>
struct compute
{
    using type       = _Tp;
    using value_type = _Up;

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
/// \class tim::math::compute<std::tuple<>>
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
