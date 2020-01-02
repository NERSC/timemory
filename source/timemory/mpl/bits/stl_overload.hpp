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

#include "timemory/mpl/math.hpp"

namespace tim
{
//======================================================================================//
//
//  Pre-C++11 tuple expansion
//
//======================================================================================//

// for pre-C++14 tuple expansion to arguments
namespace impl
{
//--------------------------------------------------------------------------------------//
// Stores a tuple of indices.  Used by tuple and pair, and by bind() to
// extract the elements in a tuple.
template <size_t... _Indexes>
struct _Index_tuple
{};

//--------------------------------------------------------------------------------------//
// Concatenates two _Index_tuples.
template <typename _Itup1, typename _Itup2>
struct _Itup_cat;

//--------------------------------------------------------------------------------------//

template <size_t... _Ind1, size_t... _Ind2>
struct _Itup_cat<_Index_tuple<_Ind1...>, _Index_tuple<_Ind2...>>
{
    using __type = _Index_tuple<_Ind1..., (_Ind2 + sizeof...(_Ind1))...>;
};

//--------------------------------------------------------------------------------------//
// Builds an _Index_tuple<0, 1, 2, ..., _Num-1>.
template <size_t _Num, size_t _Off = 0>
struct _Build_index_tuple
: _Itup_cat<typename _Build_index_tuple<_Num / 2, _Off>::__type,
            typename _Build_index_tuple<_Num - _Num / 2, _Off>::__type>
{};

//--------------------------------------------------------------------------------------//

template <size_t _Off>
struct _Build_index_tuple<1, _Off>
{
    using __type = _Index_tuple<0 + _Off>;
};

//--------------------------------------------------------------------------------------//

template <size_t _Off>
struct _Build_index_tuple<0, _Off>
{
    using __type = _Index_tuple<>;
};

//--------------------------------------------------------------------------------------//
/// Class template integer_sequence
template <typename _Tp, _Tp... _Idx>
struct integer_sequence
{
    using value_type = _Tp;
    static constexpr size_t size() noexcept { return sizeof...(_Idx); }
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, _Tp _Num,
          typename _ISeq = typename _Build_index_tuple<_Num>::__type>
struct _Make_integer_sequence;

//--------------------------------------------------------------------------------------//

template <typename _Tp, _Tp _Num, size_t... _Idx>
struct _Make_integer_sequence<_Tp, _Num, _Index_tuple<_Idx...>>
{
    static_assert(_Num >= 0, "Cannot make integer sequence of negative length");
    using __type = integer_sequence<_Tp, static_cast<_Tp>(_Idx)...>;
};

//--------------------------------------------------------------------------------------//

}  // namespace impl

//======================================================================================//

/// Alias template make_integer_sequence
template <typename _Tp, _Tp _Num>
using make_integer_sequence = typename impl::_Make_integer_sequence<_Tp, _Num>::__type;

/// Alias template index_sequence
template <size_t... _Idx>
using index_sequence = impl::integer_sequence<size_t, _Idx...>;

/// Alias template make_index_sequence
template <size_t _Num>
using make_index_sequence = make_integer_sequence<size_t, _Num>;

/// Alias template index_sequence_for
template <typename... _Types>
using index_sequence_for = make_index_sequence<sizeof...(_Types)>;

/// Alias template for enable_if
template <bool B, typename T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

/// Alias template for decay
template <typename T>
using decay_t = typename std::decay<T>::type;

/// the namespace is provided to hide stl overload from global namespace but provide
/// a method of using the namespace without a "using namespace tim;"
namespace stl_overload
{
//--------------------------------------------------------------------------------------//
//
//      tuple printer
//
//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct tuple_printer
{
    using size_type = ::std::size_t;
    tuple_printer(size_type _N, size_type _Ntot, const _Tp& obj, ::std::ostream& os)
    {
        os << ((_N == 0) ? "(" : "") << obj << ((_N + 1 == _Ntot) ? ")" : ",");
    }
};

//--------------------------------------------------------------------------------------//

struct mpl
{
    //----------------------------------------------------------------------------------//

    template <size_t _N, size_t _Nt, typename _Access, typename _Tuple, typename... _Args,
              typename std::enable_if<(_N == _Nt), char>::type = 0>
    static void impl_with_indices(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(_N, _Nt + 1, std::forward<Type>(std::get<_N>(__t)),
                   std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, size_t _Nt, typename _Access, typename _Tuple, typename... _Args,
              typename std::enable_if<(_N < _Nt), char>::type = 0>
    static void impl_with_indices(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(_N, _Nt + 1, std::forward<Type>(std::get<_N>(__t)),
                   std::forward<_Args>(__args)...);
        // recursive call
        impl_with_indices<_N + 1, _Nt, _Access, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void with_indices(_Tuple&& __t, _Args&&... __args)
    {
        impl_with_indices<0, _N - 1, _Access, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
    //  per-element addition
    //
    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
    static void impl_plus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        // using value_type = decay_t<decltype(std::get<_Idx>(_lhs))>;
        // math::compute<value_type>::plus(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        std::get<_Idx>(_lhs) += std::get<_Idx>(_rhs);
    }

    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
    static void impl_plus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        impl_plus<_Tuple, _Idx>(_lhs, _rhs);
        impl_plus<_Tuple, _Nt...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t... _Idx>
    static void impl_plus(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
    {
        impl_plus<_Tuple, _Idx...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
    static void plus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        impl_plus<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//
    //  per-element subtraction
    //
    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
    static void impl_minus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        // using value_type = decay_t<decltype(std::get<_Idx>(_lhs))>;
        // math::compute<value_type>::minus(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        std::get<_Idx>(_lhs) -= std::get<_Idx>(_rhs);
    }

    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
    static void impl_minus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        impl_minus<_Tuple, _Idx>(_lhs, _rhs);
        impl_minus<_Tuple, _Nt...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t... _Idx>
    static void impl_minus(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
    {
        impl_minus<_Tuple, _Idx...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
    static void minus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        impl_minus<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//
    //  per-element multiplication
    //
    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
    static void impl_multiply(_Tuple& _lhs, const _Tuple& _rhs)
    {
        using value_type = decay_t<decltype(std::get<_Idx>(_lhs))>;
        math::compute<value_type>::multiply(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        // std::get<_Idx>(_lhs) *= std::get<_Idx>(_rhs);
    }

    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
    static void impl_multiply(_Tuple& _lhs, const _Tuple& _rhs)
    {
        impl_multiply<_Tuple, _Idx>(_lhs, _rhs);
        impl_multiply<_Tuple, _Nt...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t... _Idx>
    static void impl_multiply(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
    {
        impl_multiply<_Tuple, _Idx...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
    static void multiply(_Tuple& _lhs, const _Tuple& _rhs)
    {
        impl_multiply<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//
    //  per-element division
    //
    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
    static void impl_divide(_Tuple& _lhs, const _Tuple& _rhs)
    {
        using value_type = decay_t<decltype(std::get<_Idx>(_lhs))>;
        math::compute<value_type>::divide(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
        // std::get<_Idx>(_lhs) /= std::get<_Idx>(_rhs);
    }

    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
    static void impl_divide(_Tuple& _lhs, const _Tuple& _rhs)
    {
        impl_divide<_Tuple, _Idx>(_lhs, _rhs);
        impl_divide<_Tuple, _Nt...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t... _Idx>
    static void impl_divide(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
    {
        impl_divide<_Tuple, _Idx...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
    static void divide(_Tuple& _lhs, const _Tuple& _rhs)
    {
        impl_divide<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//
};

//--------------------------------------------------------------------------------------//
//
//      operator <<
//
//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::ostream&
operator<<(::std::ostream& os, const ::std::tuple<_Types...>& p)
{
    using apply_t = ::std::tuple<tuple_printer<_Types>...>;
    ::tim::stl_overload::mpl::with_indices<apply_t>(p, std::ref(os));
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
    ::tim::stl_overload::mpl::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator+=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    ::tim::stl_overload::mpl::plus(lhs, rhs);
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

template <typename _Tp, typename... _Extra>
::std::vector<_Tp, _Extra...>&
operator+=(::std::vector<_Tp, _Extra...>& lhs, const ::std::vector<_Tp, _Extra...>& rhs)
{
    const auto _N = ::std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < _N; ++i)
        lhs[i] += rhs[i];
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
    ::tim::stl_overload::mpl::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator-=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    ::tim::stl_overload::mpl::minus(lhs, rhs);
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

template <typename _Tp, typename... _Extra>
::std::vector<_Tp, _Extra...>&
operator-=(::std::vector<_Tp, _Extra...>& lhs, const ::std::vector<_Tp, _Extra...>& rhs)
{
    const auto _N = ::std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < _N; ++i)
        lhs[i] -= rhs[i];
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator *=
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator*=(::std::array<_Tp, _N>& lhs, const ::std::array<_Tp, _N>& rhs)
{
    ::tim::stl_overload::mpl::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator*=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    ::tim::stl_overload::mpl::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator*=(::std::pair<_Lhs, _Rhs>& lhs, const ::std::pair<_Lhs, _Rhs>& rhs)
{
    lhs.first *= rhs.first;
    lhs.second *= rhs.second;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
::std::vector<_Tp, _Extra...>&
operator*=(::std::vector<_Tp, _Extra...>& lhs, const ::std::vector<_Tp, _Extra...>& rhs)
{
    const auto _N = ::std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < _N; ++i)
        lhs[i] *= rhs[i];
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator /=
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator/=(::std::array<_Tp, _N>& lhs, const ::std::array<_Tp, _N>& rhs)
{
    ::tim::stl_overload::mpl::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator/=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    ::tim::stl_overload::mpl::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator/=(::std::pair<_Lhs, _Rhs>& lhs, const ::std::pair<_Lhs, _Rhs>& rhs)
{
    lhs.first /= rhs.first;
    lhs.second /= rhs.second;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Extra>
::std::vector<_Tp, _Extra...>&
operator/=(::std::vector<_Tp, _Extra...>& lhs, const ::std::vector<_Tp, _Extra...>& rhs)
{
    const auto _N = ::std::min(lhs.size(), rhs.size());
    for(size_t i = 0; i < _N; ++i)
        lhs[i] /= rhs[i];
    return lhs;
}

//--------------------------------------------------------------------------------------//

}  // namespace stl_overload

using namespace stl_overload;

//======================================================================================//

}  // namespace tim
