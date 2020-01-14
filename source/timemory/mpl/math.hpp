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
struct compute
{
    using type = _Tp;

    static type abs(const type& _val) { return std::abs(_val); }

    static type sqr(const type& _val) { return _val * _val; }

    static type sqrt(const type& _val) { return std::sqrt(_val); }

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

    static type abs(type _val)
    {
        for(auto& itr : _val)
            itr = compute_type::abs(itr);
        return _val;
    }

    static type sqrt(type _val)
    {
        for(auto& itr : _val)
            itr = compute_type::sqrt(itr);
        return _val;
    }

    static type sqr(type _val)
    {
        for(auto& itr : _val)
            itr = compute_type::sqr(itr);
        return _val;
    }

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

    static type abs(type _val)
    {
        for(auto& itr : _val)
            itr = compute_type::abs(itr);
        return _val;
    }

    static type sqrt(type _val)
    {
        for(auto& itr : _val)
            itr = compute_type::sqrt(itr);
        return _val;
    }

    static type sqr(type _val)
    {
        for(auto& itr : _val)
            itr = compute_type::sqr(itr);
        return _val;
    }

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

    static type abs(type _val)
    {
        for(auto& itr : _val)
            itr = compute_type::abs(itr);
        return _val;
    }

    static type sqrt(type _val)
    {
        for(auto& itr : _val)
            itr = compute_type::sqrt(itr);
        return _val;
    }

    static type sqr(type _val)
    {
        for(auto& itr : _val)
            itr = compute_type::sqr(itr);
        return _val;
    }

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

    static type abs(const type& _val)
    {
        return type{ lhs_compute_type::abs(_val.first),
                     rhs_compute_type::abs(_val.second) };
    }

    static type sqrt(const type& _val)
    {
        return type{ lhs_compute_type::sqrt(_val.first),
                     rhs_compute_type::sqrt(_val.second) };
    }

    static type sqr(const type& _val)
    {
        return type{ lhs_compute_type::sqr(_val.first),
                     rhs_compute_type::sqr(_val.second) };
    }

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

template <typename... _Types>
struct compute<std::tuple<_Types...>>
{
    using type                   = std::tuple<_Types...>;
    static constexpr size_t size = sizeof...(_Types);

    struct impl
    {
        //------------------------------------------------------------------------------//
        //  per-element abs
        //
        template <typename _Tuple, size_t... _Idx>
        static _Tuple abs(_Tuple _ret, index_sequence<_Idx...>)
        {
            using init_list_type = std::initializer_list<int>;
            auto&& tmp           = init_list_type{ (
                std::get<_Idx>(_ret) =
                    compute<decay_t<decltype(std::get<_Idx>(_ret))>>::abs(
                        std::get<_Idx>(_ret)),
                0)... };
            consume_parameters(tmp);
            return _ret;
        }

        //------------------------------------------------------------------------------//
        //  per-element sqrt
        //
        template <typename _Tuple, size_t... _Idx>
        static _Tuple sqrt(_Tuple _ret, index_sequence<_Idx...>)
        {
            using init_list_type = std::initializer_list<int>;
            auto&& tmp           = init_list_type{ (
                std::get<_Idx>(_ret) =
                    compute<decay_t<decltype(std::get<_Idx>(_ret))>>::sqrt(
                        std::get<_Idx>(_ret)),
                0)... };
            consume_parameters(tmp);
            return _ret;
        }

        //------------------------------------------------------------------------------//
        //  per-element sqr
        //
        template <typename _Tuple, size_t... _Idx>
        static _Tuple sqr(_Tuple _ret, index_sequence<_Idx...>)
        {
            using init_list_type = std::initializer_list<int>;
            auto&& tmp           = init_list_type{ (
                std::get<_Idx>(_ret) =
                    compute<decay_t<decltype(std::get<_Idx>(_ret))>>::sqr(
                        std::get<_Idx>(_ret)),
                0)... };
            consume_parameters(tmp);
            return _ret;
        }

        //------------------------------------------------------------------------------//
        //  per-element max
        //
        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
        static void max(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs)
        {
            using compute_type = compute<decay_t<decltype(std::get<_Idx>(_lhs))>>;
            std::get<_Idx>(_ret) =
                compute_type::max(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        }

        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
        static void max(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs)
        {
            max<_Tuple, _Idx>(_ret, _lhs, _rhs);
            max<_Tuple, _Nt...>(_ret, _lhs, _rhs);
        }

        template <typename _Tuple, size_t... _Idx>
        static void max(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs,
                        index_sequence<_Idx...>)
        {
            max<_Tuple, _Idx...>(_ret, _lhs, _rhs);
        }

        //------------------------------------------------------------------------------//
        //  per-element min
        //
        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
        static void min(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs)
        {
            using compute_type = compute<decay_t<decltype(std::get<_Idx>(_lhs))>>;
            std::get<_Idx>(_ret) =
                compute_type::min(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        }

        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
        static void min(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs)
        {
            min<_Tuple, _Idx>(_ret, _lhs, _rhs);
            min<_Tuple, _Nt...>(_ret, _lhs, _rhs);
        }

        template <typename _Tuple, size_t... _Idx>
        static void min(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs,
                        index_sequence<_Idx...>)
        {
            min<_Tuple, _Idx...>(_ret, _lhs, _rhs);
        }

        //------------------------------------------------------------------------------//
        //  per-element addition
        //
        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
        static void plus(_Tuple& _lhs, const _Tuple& _rhs)
        {
            using compute_type = compute<decay_t<decltype(std::get<_Idx>(_lhs))>>;
            compute_type::plus(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        }

        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
        static void plus(_Tuple& _lhs, const _Tuple& _rhs)
        {
            plus<_Tuple, _Idx>(_lhs, _rhs);
            plus<_Tuple, _Nt...>(_lhs, _rhs);
        }

        template <typename _Tuple, size_t... _Idx>
        static void plus(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
        {
            plus<_Tuple, _Idx...>(_lhs, _rhs);
        }

        //------------------------------------------------------------------------------//
        //  per-element subtraction
        //
        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
        static void minus(_Tuple& _lhs, const _Tuple& _rhs)
        {
            using compute_type = compute<decay_t<decltype(std::get<_Idx>(_lhs))>>;
            compute_type::minus(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        }

        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
        static void minus(_Tuple& _lhs, const _Tuple& _rhs)
        {
            minus<_Tuple, _Idx>(_lhs, _rhs);
            minus<_Tuple, _Nt...>(_lhs, _rhs);
        }

        template <typename _Tuple, size_t... _Idx>
        static void minus(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
        {
            minus<_Tuple, _Idx...>(_lhs, _rhs);
        }

        //------------------------------------------------------------------------------//
        //  per-element multiplication
        //
        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
        static void multiply(_Tuple& _lhs, const _Tuple& _rhs)
        {
            using compute_type = compute<decay_t<decltype(std::get<_Idx>(_lhs))>>;
            compute_type::multiply(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        }

        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
        static void multiply(_Tuple& _lhs, const _Tuple& _rhs)
        {
            multiply<_Tuple, _Idx>(_lhs, _rhs);
            multiply<_Tuple, _Nt...>(_lhs, _rhs);
        }

        template <typename _Tuple, size_t... _Idx>
        static void multiply(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
        {
            multiply<_Tuple, _Idx...>(_lhs, _rhs);
        }

        //------------------------------------------------------------------------------//
        //  per-element division
        //
        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
        static void divide(_Tuple& _lhs, const _Tuple& _rhs)
        {
            using compute_type = compute<decay_t<decltype(std::get<_Idx>(_lhs))>>;
            compute_type::divide(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        }

        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
        static void divide(_Tuple& _lhs, const _Tuple& _rhs)
        {
            divide<_Tuple, _Idx>(_lhs, _rhs);
            divide<_Tuple, _Nt...>(_lhs, _rhs);
        }

        template <typename _Tuple, size_t... _Idx>
        static void divide(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
        {
            divide<_Tuple, _Idx...>(_lhs, _rhs);
        }

        //------------------------------------------------------------------------------//
        //  per-element percent diff
        //
        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
        static void percent_diff(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs)
        {
            using compute_type = compute<decay_t<decltype(std::get<_Idx>(_lhs))>>;
            compute_type::percent_diff(std::get<_Idx>(_ret), std::get<_Idx>(_lhs),
                                       std::get<_Idx>(_rhs));
        }

        template <typename _Tuple, size_t _Idx, size_t... _Nt,
                  enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
        static void percent_diff(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs)
        {
            percent_diff<_Tuple, _Idx>(_ret, _lhs, _rhs);
            percent_diff<_Tuple, _Nt...>(_ret, _lhs, _rhs);
        }

        template <typename _Tuple, size_t... _Idx>
        static void percent_diff(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs,
                                 index_sequence<_Idx...>)
        {
            percent_diff<_Tuple, _Idx...>(_ret, _lhs, _rhs);
        }
    };

    static type abs(const type& _ret)
    {
        return impl::template abs<type>(_ret, make_index_sequence<size>{});
    }

    static type sqrt(const type& _ret)
    {
        return impl::template sqrt<type>(_ret, make_index_sequence<size>{});
    }

    static type sqr(const type& _ret)
    {
        return impl::template sqr<type>(_ret, make_index_sequence<size>{});
    }

    static type max(const type& _lhs, const type& _rhs)
    {
        type _ret;
        impl::template max<type>(_ret, _lhs, _rhs, make_index_sequence<size>{});
        return _ret;
    }

    static type min(const type& _lhs, const type& _rhs)
    {
        type _ret;
        impl::template min<type>(_ret, _lhs, _rhs, make_index_sequence<size>{});
        return _ret;
    }

    static void plus(type& _lhs, const type& _rhs)
    {
        impl::template plus<type>(_lhs, _rhs, make_index_sequence<size>{});
    }

    static void minus(type& _lhs, const type& _rhs)
    {
        impl::template minus<type>(_lhs, _rhs, make_index_sequence<size>{});
    }

    static void multiply(type& _lhs, const type& _rhs)
    {
        impl::template multiply<type>(_lhs, _rhs, make_index_sequence<size>{});
    }

    static void divide(type& _lhs, const type& _rhs)
    {
        impl::template divide<type>(_lhs, _rhs, make_index_sequence<size>{});
    }

    static void percent_diff(type& _ret, const type& _lhs, const type& _rhs)
    {
        impl::template percent_diff<type>(_ret, _lhs, _rhs, make_index_sequence<size>{});
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
    static void plus(type&, const type&) {}
    static void minus(type&, const type&) {}
    static void multiply(type&, const type&) {}
    static void divide(type&, const type&) {}
    static void percent_diff(type&, const type&, const type&) {}
};

//--------------------------------------------------------------------------------------//

}  // namespace math
}  // namespace tim
