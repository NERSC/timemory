//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file apply.hpp
 * \headerfile apply.hpp "timemory/mpl/apply.hpp"
 * Provides the template meta-programming expansions heavily utilized by TiMemory
 *
 */

#pragma once

#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__NVCC__)
#    define TIMEMORY_LAMBDA __host__ __device__
#    define TIMEMORY_HOST_LAMBDA __host__
#    define TIMEMORY_DEVICE_LAMBDA __device__
#else
#    define TIMEMORY_LAMBDA
#    define TIMEMORY_HOST_LAMBDA
#    define TIMEMORY_DEVICE_LAMBDA
#endif

//======================================================================================//

namespace tim
{
// clang-format off
namespace device { struct cpu; struct gpu; }  // namespace device
// clang-format on

//--------------------------------------------------------------------------------------//
//
//  STL overload
//
//--------------------------------------------------------------------------------------//

namespace stl_overload
{
//--------------------------------------------------------------------------------------//

template <typename T, typename U>
::std::ostream&
operator<<(::std::ostream&, const ::std::pair<T, U>&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::ostream&
operator<<(::std::ostream&, const ::std::tuple<_Types...>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator+=(::std::array<_Tp, _N>&, const ::std::array<_Tp, _N>&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator+=(::std::pair<_Lhs, _Rhs>&, const ::std::pair<_Lhs, _Rhs>&);

//--------------------------------------------------------------------------------------//

template <typename _Tp, size_t _N>
::std::array<_Tp, _N>&
operator-=(::std::array<_Tp, _N>&, const ::std::array<_Tp, _N>&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator-=(::std::tuple<_Types...>&, const ::std::tuple<_Types...>&);

//--------------------------------------------------------------------------------------//

template <typename _Lhs, typename _Rhs>
::std::pair<_Lhs, _Rhs>&
operator-=(::std::pair<_Lhs, _Rhs>&, const ::std::pair<_Lhs, _Rhs>&);

//--------------------------------------------------------------------------------------//

template <typename... _Types>
::std::tuple<_Types...>&
operator+=(::std::tuple<_Types...>&, const ::std::tuple<_Types...>&);

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

}  // namespace stl_overload

//--------------------------------------------------------------------------------------//
//
//  Function traits
//
//--------------------------------------------------------------------------------------//

template <typename T>
struct function_traits;

template <typename R, typename... Args>
struct function_traits<std::function<R(Args...)>>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R(Args...)>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = args_type;
};

// member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...)>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = std::tuple<C&, Args...>;
};

// const member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = true;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = std::tuple<C&, Args...>;
};

// member object pointer
template <typename C, typename R>
struct function_traits<R(C::*)>
{
    static constexpr bool is_memfun = true;
    static constexpr bool is_const  = false;
    static const size_t   nargs     = 0;
    using result_type               = R;
    using args_type                 = std::tuple<>;
    using call_type                 = std::tuple<C&>;
};

#if __cplusplus >= 201703L

template <typename R, typename... Args>
struct function_traits<std::function<R(Args...) noexcept>>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...) noexcept>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R(Args...) noexcept>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = args_type;
};

// member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) noexcept>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = std::tuple<C&, Args...>;
};

// const member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const noexcept>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = true;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = std::tuple<Args...>;
    using call_type                   = std::tuple<C&, Args...>;
};

#endif

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

//======================================================================================//

template <typename _Ret>
struct _apply_impl
{
    //----------------------------------------------------------------------------------//
    //  invoke a function with a tuple
    //
    template <typename _Fn, typename _Tuple, size_t... _Idx>
    static _Ret invoke(_Fn&& __f, _Tuple&& __t, index_sequence<_Idx...>)
    {
        return __f(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }

    //----------------------------------------------------------------------------------//
    // prefix with _sep
    //
    template <typename _Sep, typename _Arg,
              enable_if_t<std::is_same<_Ret, std::string>::value, char> = 0>
    static _Ret join_tail(std::stringstream& _ss, const _Sep& _sep, _Arg&& _arg)
    {
        _ss << _sep << std::forward<_Arg>(_arg);
        return _ss.str();
    }

    //----------------------------------------------------------------------------------//
    // prefix with _sep
    //
    template <typename _Sep, typename _Arg, typename... _Args,
              enable_if_t<std::is_same<_Ret, std::string>::value, char> = 0>
    static _Ret join_tail(std::stringstream& _ss, const _Sep& _sep, _Arg&& _arg,
                          _Args&&... __args)
    {
        _ss << _sep << std::forward<_Arg>(_arg);
        return join_tail<_Sep, _Args...>(_ss, _sep, std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
    // don't prefix
    //
    template <typename _Sep, typename _Arg,
              enable_if_t<std::is_same<_Ret, std::string>::value, char> = 0>
    static _Ret join(std::stringstream& _ss, const _Sep&, _Arg&& _arg)
    {
        _ss << std::forward<_Arg>(_arg);
        return _ss.str();
    }

    //----------------------------------------------------------------------------------//
    // don't prefix
    //
    template <typename _Sep, typename _Arg, typename... _Args,
              enable_if_t<std::is_same<_Ret, std::string>::value, char> = 0,
              enable_if_t<(sizeof...(_Args) > 0), int>                  = 0>
    static _Ret join(std::stringstream& _ss, const _Sep& _sep, _Arg&& _arg,
                     _Args&&... __args)
    {
        _ss << std::forward<_Arg>(_arg);
        return join_tail<_Sep, _Args...>(_ss, _sep, std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
};

//======================================================================================//

template <>
struct _apply_impl<void>
{
    using _Ret = void;

    //----------------------------------------------------------------------------------//
    //  invoke a function with a tuple
    //
    template <typename _Fn, typename _Tuple, size_t... _Idx>
    static _Ret invoke(_Fn&& __f, _Tuple&& __t, index_sequence<_Idx...>)
    {
        __f(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }

    //----------------------------------------------------------------------------------//
    //  add two tuples
    //
    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
    static void plus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        using namespace stl_overload;
        // assign argument
        std::get<_Idx>(_lhs) += std::get<_Idx>(_rhs);
    }

    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
    static void plus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        plus<_Tuple, _Idx>(_lhs, _rhs);
        plus<_Tuple, _Nt...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t... _Idx>
    static _Ret plus(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
    {
        plus<_Tuple, _Idx...>(_lhs, _rhs);
    }

    //----------------------------------------------------------------------------------//
    //  subtract two tuples
    //
    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
    static void minus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        // assign argument
        std::get<_Idx>(_lhs) -= std::get<_Idx>(_rhs);
    }

    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) > 0), char> = 0>
    static void minus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        minus<_Tuple, _Idx>(_lhs, _rhs);
        minus<_Tuple, _Nt...>(_lhs, _rhs);
    }

    template <typename _Tuple, size_t... _Idx>
    static _Ret minus(_Tuple& _lhs, const _Tuple& _rhs, index_sequence<_Idx...>)
    {
        minus<_Tuple, _Idx...>(_lhs, _rhs);
    }

    //----------------------------------------------------------------------------------//
    // temporary construction
    //
    template <typename _Type, typename... _Args>
    static void construct(_Args&&... _args)
    {
        _Type(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // temporary construction
    //
    template <typename _Type, typename... _Args, size_t... _Idx>
    static void construct_tuple(std::tuple<_Args...>&& _args, index_sequence<_Idx>...)
    {
        construct<_Type>(std::get<_Idx>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Value,
              enable_if_t<(_N == _Nt), char> = 0>
    static void set_value(_Tuple&& __t, _Value&& __v)
    {
        // assign argument
        std::get<_N>(__t) = __v;
    }

    template <size_t _N, size_t _Nt, typename _Tuple, typename _Value,
              enable_if_t<(_N < _Nt), char> = 0>
    static void set_value(_Tuple&& __t, _Value&& __v)
    {
        // call operator()
        std::get<_N>(__t) = __v;
        // recursive call
        set_value<_N + 1, _Nt, _Tuple, _Value>(std::forward<_Tuple>(__t),
                                               std::forward<_Value>(__v));
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, size_t _N, typename _Tuple,
              typename... _Args, enable_if_t<(_N == 0), char> = 0>
    static void unroll_access(_Tuple&& __t, _Args&&... __args)
    {
        using _Tp = decltype(std::get<_N>(__t));
        using _Rp = typename std::remove_reference<_Tp>::type;
        using _Ap = typename std::remove_const<_Rp>::type;
        _Access<_Ap>(std::forward<_Tp>(std::get<_N>(__t)),
                     std::forward<_Args>(__args)...);
    }

    template <template <typename> class _Access, size_t _N, typename _Tuple,
              typename... _Args, enable_if_t<(_N > 0), char> = 0>
    static void unroll_access(_Tuple&& __t, _Args&&... __args)
    {
        using _Tp = decltype(std::get<_N>(__t));
        using _Rp = typename std::remove_reference<_Tp>::type;
        using _Ap = typename std::remove_const<_Rp>::type;
        _Access<_Ap>(std::forward<_Tp>(std::get<_N>(__t)),
                     std::forward<_Args>(__args)...);
        unroll_access<_Access, _N - 1, _Tuple, _Args...>(std::forward<_Tuple>(__t),
                                                         std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args, size_t... _Idx>
    static void variadic_1d(_Tuple&& __t, _Args&&... _args, index_sequence<_Idx...>)
    {
        (void) std::initializer_list<int>{ (
            construct<typename std::tuple_element<_Idx, _Access>::type>(
                std::get<_Idx>(__t), std::forward<_Args>(_args)...),
            0)... };
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _TupleA, typename _TupleB, typename... _Args,
              size_t... _Idx>
    static void variadic_2d(_TupleA&& __a, _TupleB&& __b, _Args&&... _args,
                            index_sequence<_Idx...>)
    {
        (void) std::initializer_list<int>{ (
            construct<typename std::tuple_element<_Idx, _Access>::type>(
                std::get<_Idx>(__a), std::get<_Idx>(__b), std::forward<_Args>(_args)...),
            0)... };
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, size_t _N, typename _Tuple,
              typename... _Args, enable_if_t<(_N == 0), char> = 0>
    static void type_access(_Args&&... __args)
    {
        using _Tp = typename std::tuple_element<_N, _Tuple>::type;
        using _Rp = typename std::remove_reference<_Tp>::type;
        using _Ap = typename std::remove_const<_Rp>::type;
        _Access<_Ap>(std::forward<_Args>(__args)...);
    }

    template <template <typename> class _Access, size_t _N, typename _Tuple,
              typename... _Args, enable_if_t<(_N > 0), char> = 0>
    static void type_access(_Args&&... __args)
    {
        using _Tp = typename std::tuple_element<_N, _Tuple>::type;
        using _Rp = typename std::remove_reference<_Tp>::type;
        using _Ap = typename std::remove_const<_Rp>::type;
        _Access<_Ap>(std::forward<_Args>(__args)...);
        type_access<_Access, _N - 1, _Tuple, _Args...>(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
              typename... _Args, enable_if_t<(_N == _Nt), char> = 0>
    static void apply_access(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(std::forward<Type>(std::get<_N>(__t)), std::forward<_Args>(__args)...);
    }

    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
              typename... _Args, enable_if_t<(_N < _Nt), char> = 0>
    static void apply_access(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(std::forward<Type>(std::get<_N>(__t)), std::forward<_Args>(__args)...);
        // recursive call
        apply_access<_N + 1, _Nt, _Access, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
              typename... _Args, enable_if_t<(_N == _Nt), char> = 0>
    static void apply_access_with_indices(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(_N, _Nt + 1, std::forward<Type>(std::get<_N>(__t)),
                   std::forward<_Args>(__args)...);
    }

    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
              typename... _Args, enable_if_t<(_N < _Nt), char> = 0>
    static void apply_access_with_indices(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(_N, _Nt + 1, std::forward<Type>(std::get<_N>(__t)),
                   std::forward<_Args>(__args)...);
        // recursive call
        apply_access_with_indices<_N + 1, _Nt, _Access, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _TupleA,
              typename _TupleB, typename... _Args, enable_if_t<(_N == _Nt), char> = 0>
    static void apply_access2(_TupleA&& __ta, _TupleB&& __tb, _Args&&... __args)
    {
        // call constructor
        using TypeA      = decltype(std::get<_N>(__ta));
        using TypeB      = decltype(std::get<_N>(__tb));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(std::forward<TypeA>(std::get<_N>(__ta)),
                   std::forward<TypeB>(std::get<_N>(__tb)),
                   std::forward<_Args>(__args)...);
    }

    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _TupleA,
              typename _TupleB, typename... _Args, enable_if_t<(_N < _Nt), char> = 0>
    static void apply_access2(_TupleA&& __ta, _TupleB&& __tb, _Args&&... __args)
    {
        // call constructor
        using TypeA      = decltype(std::get<_N>(__ta));
        using TypeB      = decltype(std::get<_N>(__tb));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(std::forward<TypeA>(std::get<_N>(__ta)),
                   std::forward<TypeB>(std::get<_N>(__tb)),
                   std::forward<_Args>(__args)...);
        // recursive call
        apply_access2<_N + 1, _Nt, _Access, _TupleA, _TupleB, _Args...>(
            std::forward<_TupleA>(__ta), std::forward<_TupleB>(__tb),
            std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
    // unroll
    template <size_t _N, typename _Device, typename _Func, typename... _Args,
              typename std::enable_if<
                  (_N == 1 && std::is_same<_Device, device::gpu>::value), char>::type = 0>
    TIMEMORY_LAMBDA static void unroll(_Func&& __func, _Args&&... __args)
    {
        std::forward<_Func>(__func)(std::forward<_Args>(__args)...);
    }

    template <size_t _N, typename _Device, typename _Func, typename... _Args,
              typename std::enable_if<
                  (_N > 1 && std::is_same<_Device, device::gpu>::value), char>::type = 0>
    TIMEMORY_LAMBDA static void unroll(_Func&& __func, _Args&&... __args)
    {
        std::forward<_Func>(__func)(std::forward<_Args>(__args)...);
        unroll<_N - 1, _Device, _Func, _Args...>(std::forward<_Func>(__func),
                                                 std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
    // unroll
    template <size_t _N, typename _Device, typename _Func, typename... _Args,
              typename std::enable_if<
                  (_N == 1 && std::is_same<_Device, device::cpu>::value), int>::type = 0>
    static void unroll(_Func&& __func, _Args&&... __args)
    {
        std::forward<_Func>(__func)(std::forward<_Args>(__args)...);
    }

    template <size_t _N, typename _Device, typename _Func, typename... _Args,
              typename std::enable_if<
                  (_N > 1 && std::is_same<_Device, device::cpu>::value), int>::type = 0>
    static void unroll(_Func&& __func, _Args&&... __args)
    {
        std::forward<_Func>(__func)(std::forward<_Args>(__args)...);
        unroll<_N - 1, _Device, _Func, _Args...>(std::forward<_Func>(__func),
                                                 std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
};

//======================================================================================//
//
//      Declaration
//
template <typename _Ret>
struct apply;

//======================================================================================//

template <>
struct apply<std::string>
{
    using _Ret           = std::string;
    using string_t       = std::string;
    using string_tuple_t = std::tuple<std::string>;

    //----------------------------------------------------------------------------------//
    //      Helper
    //----------------------------------------------------------------------------------//

    template <typename _Tp, bool _Val = true, typename _Up = int,
              typename _Dt = typename std::remove_const<decay_t<_Tp>>::type>
    using if_string_t = enable_if_t<(std::is_same<_Dt, char*>::value) == _Val, _Up>;

    //----------------------------------------------------------------------------------//

    template <typename _Sep, typename... _Args, typename _Return = _Ret,
              size_t _N = sizeof...(_Args), enable_if_t<(_N > 0), char> = 0>
    static _Return join(_Sep&& separator, _Args&&... __args)
    {
        std::stringstream ss;
        ss << std::boolalpha;
        return _apply_impl<_Ret>::template join<_Sep, _Args...>(
            std::ref(ss), std::forward<_Sep>(separator), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Sep, typename _Arg, if_string_t<_Arg, true> = 0>
    static _Ret join(_Sep&&, _Arg&& _arg)
    {
        return std::move(_arg);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Sep, typename _Arg, if_string_t<_Arg, false> = 0>
    static _Ret join(_Sep&&, _Arg&& _arg)
    {
        std::stringstream ss;
        ss << _arg;
        return ss.str();
    }

    //----------------------------------------------------------------------------------//

    static _Ret join(const string_t&) { return _Ret{ "" }; }

    //----------------------------------------------------------------------------------//
};

//======================================================================================//

template <typename _Ret>
struct apply
{
    using string_t = std::string;

    //----------------------------------------------------------------------------------//
    //  invoke a function
    //
    template <typename _Fn, typename... _Args, size_t _N = sizeof...(_Args)>
    static _Ret invoke(_Fn&& __f, _Args&&... __args)
    {
        return __f(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
    //  invoke a function with a tuple
    //
    template <typename _Fn, template <typename...> class _Tuple, typename... _Args,
              size_t _N = sizeof...(_Args)>
    static _Ret invoke(_Fn&& __f, _Tuple<_Args...>&& __t)
    {
        using _Tuple_t = _Tuple<_Args...>;
        return _apply_impl<_Ret>::template invoke<_Fn, _Tuple_t>(
            std::forward<_Fn>(__f), std::forward<_Tuple_t>(__t),
            make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//

    template <typename _Sep, typename _Tuple, size_t... _Idx>
    static string_t join(_Sep&& separator, _Tuple&& __tup, index_sequence<_Idx...>)
    {
        return apply<string_t>::join(separator, std::get<_Idx>(__tup)...);
    }
};

//======================================================================================//

template <>
struct apply<std::tuple<std::string>>
{
    using string_t = std::string;
    using _Ret     = string_t;
    using apply_v  = apply<string_t>;

    //----------------------------------------------------------------------------------//
    //  implementation for label + entry join
    //
    struct _impl
    {
        template <typename _LabelSep, typename _EntrySep, typename _LabelTup,
                  typename _EntryTup, size_t... _Idx>
        static _Ret join(_LabelSep&& _label_sep, _EntrySep&& _entry_sep,
                         _LabelTup&& _label_tup, _EntryTup&& _entry_tup,
                         index_sequence<_Idx...>)
        {
            return apply_v::join(std::forward<_LabelSep>(_label_sep),
                                 apply_v::join(std::forward<_EntrySep>(_entry_sep),
                                               std::get<_Idx>(_label_tup),
                                               std::get<_Idx>(_entry_tup))...);
        }
    };

    //----------------------------------------------------------------------------------//
    //  join a tuple of labels with entries
    //
    template <typename _LabelSep, typename _EntrySep, typename _LabelTup,
              typename _EntryTup, size_t _N = std::tuple_size<decay_t<_LabelTup>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static _Ret join(_LabelSep&& _label_sep, _EntrySep&& _entry_sep,
                     _LabelTup&& _label_tup, _EntryTup&& _entry_tup)
    {
        // clang-format off
        return _impl::join(std::forward<_LabelSep>(_label_sep),
                           std::forward<_EntrySep>(_entry_sep),
                           std::forward<_LabelTup>(_label_tup),
                           std::forward<_EntryTup>(_entry_tup),
                           make_index_sequence<_N>{});
        // clang-format on
    }

    //----------------------------------------------------------------------------------//
    //  join a tuple of labels with entries
    //
    template <typename _LabelSep, typename _EntrySep, typename _LabelTup,
              typename _EntryTup, size_t _N = std::tuple_size<decay_t<_LabelTup>>::value,
              enable_if_t<(_N == 0), int> = 0>
    static _Ret join(_LabelSep&&, _EntrySep&&, _LabelTup&&, _EntryTup&&)
    {
        return "";
    }

    //----------------------------------------------------------------------------------//
};

//======================================================================================//

template <>
struct apply<void>
{
    using _Ret = void;

    //----------------------------------------------------------------------------------//
    //  invoke a function
    //
    template <typename _Fn, typename... _Args, size_t _N = sizeof...(_Args)>
    static _Ret invoke(_Fn&& __f, _Args&&... __args)
    {
        __f(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
    //  invoke a function with a tuple
    //
    template <typename _Fn, template <typename...> class _Tuple, typename... _Args,
              size_t _N = sizeof...(_Args)>
    static _Ret invoke(_Fn&& __f, _Tuple<_Args...>&& __t)
    {
        using _Tuple_t = _Tuple<_Args...>;
        _apply_impl<_Ret>::template invoke<_Fn, _Tuple_t>(std::forward<_Fn>(__f),
                                                          std::forward<_Tuple_t>(__t),
                                                          make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//
    //  add two tuples
    //
    template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
    static void plus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        _apply_impl<_Ret>::template plus<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//
    //  subtract two tuples
    //
    template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
    static void minus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        _apply_impl<_Ret>::template minus<_Tuple>(_lhs, _rhs, make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Device, typename _Func, typename... _Args,
              enable_if_t<std::is_same<_Device, device::gpu>::value, char> = 0>
    TIMEMORY_LAMBDA static void unroll(_Func&& __func, _Args&&... __args)
    {
        _apply_impl<void>::template unroll<_N, _Device, _Func, _Args...>(
            std::forward<_Func>(__func), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Device, typename _Func, typename... _Args,
              enable_if_t<std::is_same<_Device, device::cpu>::value, char> = 0>
    static void unroll(_Func&& __func, _Args&&... __args)
    {
        _apply_impl<void>::template unroll<_N, _Device, _Func, _Args...>(
            std::forward<_Func>(__func), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    //      _N > 0
    //
    //----------------------------------------------------------------------------------//

    template <typename _Tuple, typename _Value,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void set_value(_Tuple&& __t, _Value&& __v)
    {
        _apply_impl<void>::template set_value<0, _N - 1, _Tuple, _Value>(
            std::forward<_Tuple>(__t), std::forward<_Value>(__v));
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void access(_Tuple&& __t, _Args&&... __args)
    {
        _apply_impl<void>::template apply_access<0, _N - 1, _Access, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
        // _apply_impl<void>::template variadic_1d<_Access, _Tuple, _Args...>(
        //    std::forward<_Tuple>(__t), std::forward<_Args>(__args)...,
        //    make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void access_with_indices(_Tuple&& __t, _Args&&... __args)
    {
        _apply_impl<void>::template apply_access_with_indices<0, _N - 1, _Access, _Tuple,
                                                              _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _TupleA, typename _TupleB, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_TupleA>>::value,
              std::size_t _Nb            = std::tuple_size<decay_t<_TupleB>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void access2(_TupleA&& __ta, _TupleB&& __tb, _Args&&... __args)
    {
        static_assert(_N == _Nb, "tuple_size 1 must match tuple_size 2");
        _apply_impl<void>::template apply_access2<0, _N - 1, _Access, _TupleA, _TupleB,
                                                  _Args...>(
            std::forward<_TupleA>(__ta), std::forward<_TupleB>(__tb),
            std::forward<_Args>(__args)...);
        // _apply_impl<void>::template variadic_2d<_Access, _TupleA, _TupleB, _Args...>(
        //    std::forward<_TupleA>(__ta), std::forward<_TupleB>(__tb),
        //    std::forward<_Args>(__args)..., make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, typename _Tuple, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void unroll_access(_Tuple&& __t, _Args&&... __args)
    {
        _apply_impl<void>::template unroll_access<_Access, _N - 1, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, typename _Tuple, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void type_access(_Args&&... __args)
    {
        _apply_impl<void>::template type_access<_Access, _N - 1, _Tuple, _Args...>(
            std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
    //
    //      _N == 0
    //
    //----------------------------------------------------------------------------------//

    template <typename _Tuple, typename _Value,
              std::size_t _N              = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N == 0), int> = 0>
    static void set_value(_Tuple&&, _Value&&)
    {}

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N              = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N == 0), int> = 0>
    static void access(_Tuple&&, _Args&&...)
    {}

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N              = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N == 0), int> = 0>
    static void access_with_indices(_Tuple&&, _Args&&...)
    {}

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _TupleA, typename _TupleB, typename... _Args,
              std::size_t _N              = std::tuple_size<decay_t<_TupleA>>::value,
              std::size_t _Nb             = std::tuple_size<decay_t<_TupleB>>::value,
              enable_if_t<(_N == 0), int> = 0>
    static void access2(_TupleA&&, _TupleB&&, _Args&&...)
    {}

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, typename _Tuple, typename... _Args,
              std::size_t _N              = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N == 0), int> = 0>
    static void unroll_access(_Tuple&&, _Args&&...)
    {}

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, typename _Tuple, typename... _Args,
              std::size_t _N              = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N == 0), int> = 0>
    static void type_access(_Args&&...)
    {}

    //----------------------------------------------------------------------------------//
};

//======================================================================================//

}  // namespace tim

//======================================================================================//

#include "timemory/mpl/bits/apply.hpp"
