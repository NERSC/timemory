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

/** \file math.hpp
 * \headerfile math.hpp "timemory/mpl/math.hpp"
 * Provides the template meta-programming expansions for math operations
 *
 */

#pragma once

#include <algorithm>
#include <array>
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
struct compute
{
    using type = _Tp;

    static type max(const type& _lhs, const type& _rhs)
    {
        return (_rhs < _lhs) ? _lhs : _rhs;
    }

    static type min(const type& _lhs, const type& _rhs)
    {
        return (_rhs > _lhs) ? _lhs : _rhs;
    }

    static void plus(type& _lhs, const type& _rhs) { _lhs += _rhs; }

    static void minus(type& _lhs, const type& _rhs) { _lhs -= _rhs; }

    static void multiply(type& _lhs, const type& _rhs) { _lhs *= _rhs; }

    static void divide(type& _lhs, const type& _rhs) { _lhs /= _rhs; }

    static void percent_diff(type& _ret, const type& _lhs, const type& _rhs)
    {
        static constexpr _Tp _zero    = _Tp(0.0);
        static constexpr _Tp _one     = _Tp(1.0);
        static constexpr _Tp _hundred = _Tp(100.0);
        _ret = (_rhs > 0) ? (_one - (_lhs / _rhs)) * _hundred : _zero;
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
struct compute<std::vector<_Tp, _Extra...>>
{
    using type         = std::vector<_Tp, _Extra...>;
    using value_type   = typename type::value_type;
    using compute_type = compute<value_type>;

    static uint64_t size(const type& _lhs, const type& _rhs)
    {
        return std::min<uint64_t>(_lhs.size(), _rhs.size());
    }

    static type max(const type& _lhs, const type& _rhs)
    {
        type _ret(_lhs.size());
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            _ret[i] = compute_type::max(_lhs[i], _rhs[i]);
        return _ret;
    }

    static type min(const type& _lhs, const type& _rhs)
    {
        type _ret(_lhs.size());
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            _ret[i] = compute_type::min(_lhs[i], _rhs[i]);
        return _ret;
    }

    static void plus(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            compute_type::plus(_lhs[i], _rhs[i]);
    }

    static void minus(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            compute_type::minus(_lhs[i], _rhs[i]);
    }

    static void multiply(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            compute_type::multiply(_lhs[i], _rhs[i]);
    }

    static void divide(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            compute_type::divide(_lhs[i], _rhs[i]);
    }

    static void percent_diff(type& _ret, const type& _lhs, const type& _rhs)
    {
        uint64_t _n = size();
        if(_n >= _ret.size())
            _ret.resize(_n);

        // in case there are nested types
        for(uint64_t i = 0; i < _n; ++i)
            compute_type::percent_diff(_ret[i], _lhs[i], _rhs[i]);
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
struct compute<std::deque<_Tp, _Extra...>>
{
    using type         = std::vector<_Tp, _Extra...>;
    using value_type   = typename type::value_type;
    using compute_type = compute<value_type>;

    static uint64_t size(const type& _lhs, const type& _rhs)
    {
        return std::min<uint64_t>(_lhs.size(), _rhs.size());
    }

    static type max(const type& _lhs, const type& _rhs)
    {
        type _ret(_lhs.size());
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            _ret[i] = compute_type::max(_lhs[i], _rhs[i]);
        return _ret;
    }

    static type min(const type& _lhs, const type& _rhs)
    {
        type _ret(_lhs.size());
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            _ret[i] = compute_type::min(_lhs[i], _rhs[i]);
        return _ret;
    }

    static void plus(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            compute_type::plus(_lhs[i], _rhs[i]);
    }

    static void minus(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            compute_type::minus(_lhs[i], _rhs[i]);
    }

    static void multiply(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            compute_type::multiply(_lhs[i], _rhs[i]);
    }

    static void divide(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            compute_type::divide(_lhs[i], _rhs[i]);
    }

    static void percent_diff(type& _ret, const type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < size(_lhs, _rhs); ++i)
            compute_type::percent_diff(_ret[i], _lhs[i], _rhs[i]);
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
struct compute<std::array<_Tp, _N>>
{
    using type         = std::array<_Tp, _N>;
    using value_type   = typename type::value_type;
    using compute_type = compute<value_type>;

    static type max(const type& _lhs, const type& _rhs)
    {
        type _ret;
        for(uint64_t i = 0; i < _N; ++i)
            _ret[i] = compute_type::max(_lhs[i], _rhs[i]);
        return _ret;
    }

    static type min(const type& _lhs, const type& _rhs)
    {
        type _ret;
        for(uint64_t i = 0; i < _N; ++i)
            _ret[i] = compute_type::min(_lhs[i], _rhs[i]);
        return _ret;
    }

    static void plus(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < _N; ++i)
            compute_type::plus(_lhs[i], _rhs[i]);
    }

    static void minus(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < _N; ++i)
            compute_type::minus(_lhs[i], _rhs[i]);
    }

    static void multiply(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < _N; ++i)
            compute_type::multiply(_lhs[i], _rhs[i]);
    }

    static void divide(type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < _N; ++i)
            compute_type::divide(_lhs[i], _rhs[i]);
    }

    static void percent_diff(type& _ret, const type& _lhs, const type& _rhs)
    {
        for(uint64_t i = 0; i < _N; ++i)
            compute_type::percent_diff(_ret[i], _lhs[i], _rhs[i]);
    }
};

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
struct compute<std::pair<_Lhs, _Rhs>>
{
    using type             = std::pair<_Lhs, _Rhs>;
    using lhs_compute_type = compute<_Lhs>;
    using rhs_compute_type = compute<_Rhs>;

    static type max(const type& _lhs, const type& _rhs)
    {
        return type{ lhs_compute_type::max(_lhs.first, _rhs.first),
                     rhs_compute_type::max(_lhs.second, _rhs.second) };
    }

    static type min(const type& _lhs, const type& _rhs)
    {
        return type{ lhs_compute_type::min(_lhs.first, _rhs.first),
                     rhs_compute_type::min(_lhs.second, _rhs.second) };
    }

    static void plus(type& _lhs, const type& _rhs)
    {
        lhs_compute_type::plus(_lhs.first, _rhs.first);
        rhs_compute_type::plus(_lhs.second, _rhs.second);
    }

    static void minus(type& _lhs, const type& _rhs)
    {
        lhs_compute_type::minus(_lhs.first, _rhs.first);
        rhs_compute_type::minus(_lhs.second, _rhs.second);
    }

    static void multiply(type& _lhs, const type& _rhs)
    {
        lhs_compute_type::multiply(_lhs.first, _rhs.first);
        rhs_compute_type::multiply(_lhs.second, _rhs.second);
    }

    static void divide(type& _lhs, const type& _rhs)
    {
        lhs_compute_type::divide(_lhs.first, _rhs.first);
        rhs_compute_type::divide(_lhs.second, _rhs.second);
    }

    static void percent_diff(type& _ret, const type& _lhs, const type& _rhs)
    {
        lhs_compute_type::percent_diff(_ret.first, _lhs.first, _rhs.first);
        rhs_compute_type::percent_diff(_ret.second, _lhs.second, _rhs.second);
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace math
}  // namespace tim
