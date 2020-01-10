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
#include "timemory/utility/stream.hpp"
#include "timemory/utility/types.hpp"

namespace tim
{
//--------------------------------------------------------------------------------------//

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

namespace impl_details
{
//----------------------------------------------------------------------------------//

template <size_t _N, size_t _Nt, typename _Access, typename _Tuple, typename... _Args,
          typename std::enable_if<(_N == _Nt), char>::type = 0>
static void
impl_with_indices(_Tuple&& __t, _Args&&... __args)
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
static void
impl_with_indices(_Tuple&& __t, _Args&&... __args)
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
static void
with_indices(_Tuple&& __t, _Args&&... __args)
{
    impl_with_indices<0, _N - 1, _Access, _Tuple, _Args...>(
        std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
}

//----------------------------------------------------------------------------------//
//  per-element addition
//
template <typename _Tuple, size_t _Idx, size_t... _Nt,
          enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
static void
impl_plus(_Tuple& _lhs, const _Tuple& _rhs)
{
    std::get<_Idx>(_lhs) += std::get<_Idx>(_rhs);
}

template <typename _Tuple, size_t _Idx, size_t... _Nt,
          enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
static void
impl_plus(_Tuple& _lhs, const _Tuple& _rhs)
{
    impl_plus<_Tuple, _Idx>(_lhs, _rhs);
    impl_plus<_Tuple, _Nt...>(_lhs, _rhs);
}

template <typename _Tuple, size_t... _Idx>
static void
impl_plus(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
{
    impl_plus<_Tuple, _Idx...>(_lhs, _rhs);
}

template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
static void
plus(_Tuple& _lhs, const _Tuple& _rhs)
{
    impl_plus<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
}

//----------------------------------------------------------------------------------//
//  per-element subtraction
//
template <typename _Tuple, size_t _Idx, size_t... _Nt,
          enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
static void
impl_minus(_Tuple& _lhs, const _Tuple& _rhs)
{
    std::get<_Idx>(_lhs) -= std::get<_Idx>(_rhs);
}

template <typename _Tuple, size_t _Idx, size_t... _Nt,
          enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
static void
impl_minus(_Tuple& _lhs, const _Tuple& _rhs)
{
    impl_minus<_Tuple, _Idx>(_lhs, _rhs);
    impl_minus<_Tuple, _Nt...>(_lhs, _rhs);
}

template <typename _Tuple, size_t... _Idx>
static void
impl_minus(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
{
    impl_minus<_Tuple, _Idx...>(_lhs, _rhs);
}

template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
static void
minus(_Tuple& _lhs, const _Tuple& _rhs)
{
    impl_minus<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
}

//----------------------------------------------------------------------------------//
//  per-element multiplication
//
template <typename _Tuple, size_t _Idx, size_t... _Nt,
          enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
static void
impl_multiply(_Tuple& _lhs, const _Tuple& _rhs)
{
    using value_type = decay_t<decltype(std::get<_Idx>(_lhs))>;
    math::compute<value_type>::multiply(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
    // std::get<_Idx>(_lhs) *= std::get<_Idx>(_rhs);
}

template <typename _Tuple, size_t _Idx, size_t... _Nt,
          enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
static void
impl_multiply(_Tuple& _lhs, const _Tuple& _rhs)
{
    impl_multiply<_Tuple, _Idx>(_lhs, _rhs);
    impl_multiply<_Tuple, _Nt...>(_lhs, _rhs);
}

template <typename _Tuple, size_t... _Idx>
static void
impl_multiply(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
{
    impl_multiply<_Tuple, _Idx...>(_lhs, _rhs);
}

template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
static void
multiply(_Tuple& _lhs, const _Tuple& _rhs)
{
    impl_multiply<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
}

//----------------------------------------------------------------------------------//
//  per-element division
//
/*
template <typename _Tuple, size_t _Idx, size_t... _Nt,
          enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
static void
impl_divide(_Tuple& _lhs, const _Tuple& _rhs)
{
    using value_type = decay_t<decltype(std::get<_Idx>(_lhs))>;
    math::compute<value_type>::divide(std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
}

template <typename _Tuple, size_t... _Idx>
static void
impl_divide(_Tuple& _lhs, const _Tuple& _rhs)
{
    using value_type = decay_t<decltype(std::get<_Idx>(_lhs))>;
    impl_divide<_Tuple, _Idx>(_lhs, _rhs);
    impl_divide<_Tuple, _Nt...>(_lhs, _rhs);
}
*/

template <typename _Tuple, size_t... _Idx>
static void
impl_divide(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
{
    using init_list_type = std::initializer_list<int>;
    auto&& ret =
        init_list_type{ (math::compute<decay_t<decltype(std::get<_Idx>(_lhs))>>::divide(
                             std::get<_Idx>(_lhs), std::get<_Idx>(_rhs)),
                         0)... };
    consume_parameters(ret);
    // impl_divide<_Tuple, _Idx...>(_lhs, _rhs);
}

template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
static void
divide(_Tuple& _lhs, const _Tuple& _rhs)
{
    impl_divide<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
}

//----------------------------------------------------------------------------------//
}  // namespace impl_details

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
    stl_overload::impl_details::with_indices<apply_t>(p, std::ref(os));
    return os;
}

//--------------------------------------------------------------------------------------//
//
//      operator <<
//
//--------------------------------------------------------------------------------------//

namespace vector_ostream
{
template <typename _Tp, typename... _Extra>
::std::ostream&
operator<<(::std::ostream& os, const ::std::vector<_Tp, _Extra...>& p)
{
    std::stringstream ss;
    ss.setf(os.flags());
    ss << "(";
    for(size_t i = 0; i < p.size(); ++i)
        ss << p.at(i) << ((i + 1 < p.size()) ? "," : "");
    ss << ")";
    os << ss.str();
    return os;
}
}  // namespace vector_ostream

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
//      operator <<
//
//--------------------------------------------------------------------------------------//

template <typename T, typename... _Extra>
::std::vector<std::string, _Extra...>&
operator<<(::std::vector<std::string, _Extra...>& os, const T& p)
{
    std::stringstream _ss;
    _ss << p;
    os.push_back(_ss.str());
    return os;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types, typename... _Extra>
::std::vector<std::string, _Extra...>&
operator<<(::std::vector<std::string, _Extra...>& os, const ::std::tuple<_Types...>& p)
{
    using apply_t = ::std::tuple<tuple_printer<_Types>...>;
    stl_overload::impl_details::with_indices<apply_t>(p, std::ref(os));
    return os;
}

//--------------------------------------------------------------------------------------//

template <typename T, typename U, typename... _Extra>
::std::vector<std::string, _Extra...>&
operator<<(::std::vector<std::string, _Extra...>& os, const ::std::pair<T, U>& p)
{
    std::stringstream _lhs, _rhs;
    _lhs << p.first;
    _rhs << p.second;
    os.push_back(_lhs.str());
    os.push_back(_rhs.str());
    os << "(" << p.first << "," << p.second << ")";
    return os;
}

//--------------------------------------------------------------------------------------//
//
//      operator += (same type)
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator+=(::std::array<_Tp, _N>& lhs, const ::std::array<_Tp, _N>& rhs)
{
    stl_overload::impl_details::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator+=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    stl_overload::impl_details::plus(lhs, rhs);
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
//      operator -= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator-=(::std::array<_Tp, _N>& lhs, const ::std::array<_Tp, _N>& rhs)
{
    stl_overload::impl_details::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator-=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    stl_overload::impl_details::minus(lhs, rhs);
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
//      operator *= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator*=(::std::array<_Tp, _N>& lhs, const ::std::array<_Tp, _N>& rhs)
{
    stl_overload::impl_details::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator*=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    stl_overload::impl_details::multiply(lhs, rhs);
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
//      operator /= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator/=(::std::array<_Tp, _N>& lhs, const ::std::array<_Tp, _N>& rhs)
{
    stl_overload::impl_details::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator/=(::std::tuple<_Types...>& lhs, const ::std::tuple<_Types...>& rhs)
{
    stl_overload::impl_details::divide(lhs, rhs);
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
//
//      operator *= (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename _Lhs, size_t _N, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::array<_Lhs, _N>&
operator*=(::std::array<_Lhs, _N>& lhs, _Rhs rhs)
{
    for(auto& itr : lhs)
        itr *= static_cast<_Lhs>(rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::tuple<_Lhs...>&
operator*=(::std::tuple<_Lhs...>& lhs, _Rhs rhs)
{
    using input_type     = ::std::tuple<_Lhs...>;
    using init_list_type = std::initializer_list<int>;

    auto&& ret =
        init_list_type{ (std::get<index_of<_Lhs, input_type>::value>(lhs) *= rhs, 0)... };
    consume_parameters(ret);

    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::pair<_Lhs, _Rhs>&
operator*=(::std::pair<_Lhs, _Rhs>& lhs, _Rhs rhs)
{
    lhs.first *= rhs;
    lhs.second *= rhs;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::vector<_Lhs, _Extra...>&
operator*=(::std::vector<_Lhs, _Extra...>& lhs, _Rhs rhs)
{
    for(auto& itr : lhs)
        itr *= static_cast<_Lhs>(rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator /= (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename _Lhs, size_t _N, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::array<_Lhs, _N>&
operator/=(::std::array<_Lhs, _N>& lhs, _Rhs rhs)
{
    for(auto& itr : lhs)
        itr /= static_cast<_Lhs>(rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::tuple<_Lhs...>&
operator/=(::std::tuple<_Lhs...>& lhs, _Rhs rhs)
{
    using input_type     = ::std::tuple<_Lhs...>;
    using init_list_type = std::initializer_list<int>;

    auto&& ret =
        init_list_type{ (std::get<index_of<_Lhs, input_type>::value>(lhs) /= rhs, 0)... };
    consume_parameters(ret);

    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::pair<_Lhs, _Rhs>&
operator/=(::std::pair<_Lhs, _Rhs>& lhs, _Rhs rhs)
{
    lhs.first /= rhs;
    lhs.second /= rhs;
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::vector<_Lhs, _Extra...>&
operator/=(::std::vector<_Lhs, _Extra...>& lhs, _Rhs rhs)
{
    for(auto& itr : lhs)
        itr /= static_cast<_Lhs>(rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator * (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename _Lhs, size_t _N, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::array<_Lhs, _N> operator*(::std::array<_Lhs, _N> lhs, _Rhs rhs)
{
    return (lhs *= rhs);
}

//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::tuple<_Lhs...> operator*(::std::tuple<_Lhs...> lhs, _Rhs rhs)
{
    return (lhs *= rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::pair<_Lhs, _Rhs> operator*(::std::pair<_Lhs, _Rhs> lhs, _Rhs rhs)
{
    return (lhs *= rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::vector<_Lhs, _Extra...> operator*(::std::vector<_Lhs, _Extra...> lhs, _Rhs rhs)
{
    return (lhs *= rhs);
}

//--------------------------------------------------------------------------------------//
//
//      operator / (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename _Lhs, size_t _N, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::array<_Lhs, _N>
operator/(::std::array<_Lhs, _N> lhs, _Rhs rhs)
{
    return (lhs /= rhs);
}

//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::tuple<_Lhs...>
operator/(::std::tuple<_Lhs...> lhs, _Rhs rhs)
{
    return (lhs /= rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::pair<_Lhs, _Rhs>
operator/(::std::pair<_Lhs, _Rhs> lhs, _Rhs rhs)
{
    return (lhs /= rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs, typename... _Extra,
          enable_if_t<(std::is_fundamental<_Rhs>::value), int> = 0>
::std::vector<_Lhs, _Extra...>
operator/(::std::vector<_Lhs, _Extra...> lhs, _Rhs rhs)
{
    return (lhs /= rhs);
}

//--------------------------------------------------------------------------------------//

}  // namespace stl_overload

using namespace stl_overload;

//======================================================================================//

}  // namespace tim
