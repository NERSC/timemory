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
 * \headerfile apply.hpp "timemory/apply.hpp"
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

//======================================================================================//

namespace tim
{
//======================================================================================//

// for pre-C++14 tuple expansion to arguments
namespace impl
{
//--------------------------------------------------------------------------------------//

// Stores a tuple of indices.  Used by tuple and pair, and by bind() to
// extract the elements in a tuple.
template <size_t... _Indexes>
struct _Index_tuple
{
};

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
{
};

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

}  // impl

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
template <bool B, typename T>
using enable_if_t = typename std::enable_if<B, T>::type;

/// Alias template for decay
template <class T>
using decay_t = typename std::decay<T>::type;

//======================================================================================//
// check if type is in expansion
//
namespace impl
{
template <typename...>
struct is_one_of
{
    static constexpr bool value = false;
};

template <typename F, typename S, typename... T>
struct is_one_of<F, S, std::tuple<T...>>
{
    static constexpr bool value =
        std::is_same<F, S>::value || is_one_of<F, std::tuple<T...>>::value;
};

template <typename F, typename S, typename... T>
struct is_one_of<F, std::tuple<S, T...>>
{
    static constexpr bool value = is_one_of<F, S, std::tuple<T...>>::value;
};

}  // namespace impl

template <typename _Tp, typename _Types>
using is_one_of_v = typename impl::is_one_of<_Tp, _Types>;

//======================================================================================//
// remove first type from expansion
//
template <typename List>
class pop_front_t;

template <typename Head, typename... Tail>
class pop_front_t<std::tuple<Head, Tail...>>
{
public:
    using Type = std::tuple<Tail...>;
};

template <typename List>
using pop_front = typename pop_front_t<List>::Type;

//======================================================================================//
// add type to expansion
//
template <typename List, typename NewElement>
class push_back_t;

template <typename... Elements, typename NewElement>
class push_back_t<std::tuple<Elements...>, NewElement>
{
public:
    using type = std::tuple<Elements..., NewElement>;
};

template <typename List, typename NewElement>
using push_back = typename push_back_t<List, NewElement>::type;

//======================================================================================//
// get the index of a type in expansion
//
template <typename _Tp, typename Type>
struct index_of;

template <typename _Tp, typename... Types>
struct index_of<_Tp, std::tuple<_Tp, Types...>>
{
    static constexpr std::size_t value = 0;
};

template <typename _Tp, typename Head, typename... Tail>
struct index_of<_Tp, std::tuple<Head, Tail...>>
{
    static constexpr std::size_t value = 1 + index_of<_Tp, std::tuple<Tail...>>::value;
};

//======================================================================================//

template <typename _Ret>
struct _apply_impl
{
    //----------------------------------------------------------------------------------//

    template <typename _Fn, typename _Tuple, size_t... _Idx>
    static _Ret all(_Fn&& __f, _Tuple&& __t, index_sequence<_Idx...>)
    {
        return __f(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }

    //----------------------------------------------------------------------------------//

    // prefix with _sep
    template <typename _Sep, typename _Arg,
              enable_if_t<std::is_same<_Ret, std::string>::value, char> = 0>
    static _Ret join_tail(std::stringstream& _ss, const _Sep& _sep, _Arg&& _arg)
    {
        _ss << _sep << std::forward<_Arg>(_arg);
        return _ss.str();
    }

    // prefix with _sep
    template <typename _Sep, typename _Arg, typename... _Args,
              enable_if_t<std::is_same<_Ret, std::string>::value, char> = 0>
    static _Ret join_tail(std::stringstream& _ss, const _Sep& _sep, _Arg&& _arg,
                          _Args&&... __args)
    {
        _ss << _sep << std::forward<_Arg>(_arg);
        return join_tail<_Sep, _Args...>(_ss, _sep, std::forward<_Args>(__args)...);
    }

    // don't prefix
    template <typename _Sep, typename _Arg,
              enable_if_t<std::is_same<_Ret, std::string>::value, char> = 0>
    static _Ret join(std::stringstream& _ss, const _Sep&, _Arg&& _arg)
    {
        _ss << std::forward<_Arg>(_arg);
        return _ss.str();
    }

    // don't prefix
    template <typename _Sep, typename _Arg, typename... _Args,
              enable_if_t<std::is_same<_Ret, std::string>::value, char> = 0>
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
    //----------------------------------------------------------------------------------//
    /*
    template <typename _Fn, typename _Tuple, size_t... _Idx>
    static void all(_Fn&& __f, _Tuple&& __t, index_sequence<_Idx...>)
    {
        __f(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }
    */
    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Value,
              enable_if_t<(_N == _Nt), char> = 0>
    static void set_value(_Tuple&& __t, _Value&& __v)
    {
        // assign argument
        std::get<_N>(__t) = std::forward<_Value>(__v);
    }

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Value,
              enable_if_t<(_N < _Nt), char> = 0>
    static void set_value(_Tuple&& __t, _Value&& __v)
    {
        // call operator()
        std::get<_N>(__t) = std::forward<_Value>(__v);
        // recursive call
        set_value<_N + 1, _Nt, _Tuple, _Value>(std::forward<_Tuple>(__t),
                                               std::forward<_Value>(__v));
    }

    //----------------------------------------------------------------------------------//
    /*
    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename... _Args,
              enable_if_t<(_N == _Nt), char> = 0>
    static void loop(_Tuple&& __t, _Args&&... __args)
    {
        // call operator()
        std::get<_N>(__t)(std::forward<_Args>(__args)...);
    }

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename... _Args,
              enable_if_t<(_N < _Nt), char> = 0>
    static void loop(_Tuple&& __t, _Args&&... __args)
    {
        // call operator()
        std::get<_N>(__t)(std::forward<_Args>(__args)...);
        // recursive call
        loop<_N + 1, _Nt, _Tuple, _Args...>(std::forward<_Tuple>(__t),
                                            std::forward<_Args>(__args)...);
    }
    */
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
    /*
    template <typename _Tp, typename _Funct, typename... _Args,
              enable_if_t<std::is_pointer<_Tp>::value, char> = 0>
    static void apply_function(_Tp&& __t, _Funct&& __f, _Args&&... __args)
    {
        (__t)->*(__f)(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Tp, typename _Funct, typename... _Args,
              enable_if_t<!std::is_pointer<_Tp>::value, char> = 0>
    static void apply_function(_Tp&& __t, _Funct&& __f, _Args&&... __args)
    {
        (__t).*(__f)(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N == _Nt), char> = 0>
    static void functions(_Tuple&& __t, _Funct&& __f, _Args&&... __args)
    {
        // call member function at index _N
        apply_function(std::get<_N>(__t), std::get<_N>(__f),
                       std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N < _Nt), char> = 0>
    static void functions(_Tuple&& __t, _Funct&& __f, _Args&&... __args)
    {
        // call member function at index _N
        apply_function(std::get<_N>(__t), std::get<_N>(__f),
                       std::forward<_Args>(__args)...);
        // recursive call
        functions<_N + 1, _Nt, _Tuple, _Funct, _Args...>(std::forward<_Tuple>(__t),
                                                         std::forward<_Funct>(__f),
                                                         std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Funct, typename... _Args,
              enable_if_t<(_N == _Nt), char> = 0>
    static void unroll(_Funct&& __f, _Args&&... __args)
    {
        (__f)(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Funct, typename... _Args,
              enable_if_t<(_N < _Nt), char> = 0>
    static void unroll(_Funct&& __f, _Args&&... __args)
    {
        (__f)(std::forward<_Args>(__args)...);
        unroll<_N + 1, _Nt, _Funct, _Args...>(std::forward<_Funct>(__f),
                                              std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N == _Nt), char> = 0>
    static void unroll_indices(_Tuple&& __t, _Funct&& __f, _Args&&... args)
    {
        using Type = decltype(std::get<_N>(__t));
        (__f)(_N, std::forward<Type>(std::get<_N>(__t)), std::forward<_Args>(args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N < _Nt), char> = 0>
    static void unroll_indices(_Tuple&& __t, _Funct&& __f, _Args&&... args)
    {
        using Type = decltype(std::get<_N>(__t));
        (__f)(_N, std::forward<Type>(std::get<_N>(__t)), std::forward<_Args>(args)...);
        unroll_indices<_N + 1, _Nt, _Tuple, _Funct, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Funct>(__f),
            std::forward<_Args>(args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N == _Nt), char> = 0>
    static void unroll_members(_Tuple&& __t, _Funct&& __f, _Args&&... __args)
    {
        (std::get<_N>(__t)).*(std::get<_N>(__f))(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N < _Nt), char> = 0>
    static void unroll_members(_Tuple&& __t, _Funct&& __f, _Args&&... __args)
    {
        (std::get<_N>(__t)).*(std::get<_N>(__f))(std::forward<_Args>(__args)...);
        unroll_members<_N + 1, _Nt, _Tuple, _Funct, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Funct>(__f),
            std::forward<_Args>(__args)...);
    }
    */
    //----------------------------------------------------------------------------------//
};

//======================================================================================//

template <typename _Ret>
struct apply
{
    //----------------------------------------------------------------------------------//
    /*
    template <typename _Fn, typename _Tuple,
              std::size_t _N    = std::tuple_size<decay_t<_Tuple>>::value,
              typename _Indices = make_index_sequence<_N>>
    static _Ret all(_Fn&& __f, _Tuple&& __t)
    {
        return _apply_impl<_Ret>::template all<_Fn, _Tuple>(
            std::forward<_Fn>(__f), std::forward<_Tuple>(__t), _Indices{});
    }
    */
    //----------------------------------------------------------------------------------//

    template <typename... _Args,
              enable_if_t<std::is_same<_Ret, std::string>::value, char> = 0>
    static _Ret join(const std::string& separator, _Args&&... __args)
    {
        std::stringstream ss;
        ss << std::boolalpha;
        return _apply_impl<_Ret>::template join<std::string, _Args...>(
            std::ref(ss), separator, std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
};

//======================================================================================//

template <>
struct apply<void>
{
    //----------------------------------------------------------------------------------//
    /*
    template <typename _Fn, typename _Tuple,
              std::size_t _N    = std::tuple_size<decay_t<_Tuple>>::value,
              typename _Indices = make_index_sequence<_N>>
    static void all(_Fn&& __f, _Tuple&& __t)
    {
        _apply_impl<void>::template all<_Fn, _Tuple>(
            std::forward<_Fn>(__f), std::forward<_Tuple>(__t), _Indices{});
    }
    */
    //----------------------------------------------------------------------------------//

    template <typename _Tuple, typename _Value,
              std::size_t _N = std::tuple_size<decay_t<_Tuple>>::value>
    static void set_value(_Tuple&& __t, _Value&& __v)
    {
        _apply_impl<void>::template set_value<0, _N - 1, _Tuple, _Value>(
            std::forward<_Tuple>(__t), std::forward<_Value>(__v));
    }

    //----------------------------------------------------------------------------------//
    /*
    template <typename _Tuple, typename... _Args,
              std::size_t _N = std::tuple_size<decay_t<_Tuple>>::value>
    static void loop(_Tuple&& __t, _Args&&... __args)
    {
        _apply_impl<void>::template loop<0, _N - 1, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }
    */
    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N = std::tuple_size<decay_t<_Tuple>>::value>
    static void access(_Tuple&& __t, _Args&&... __args)
    {
        _apply_impl<void>::template apply_access<0, _N - 1, _Access, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N = std::tuple_size<decay_t<_Tuple>>::value>
    static void access_with_indices(_Tuple&& __t, _Args&&... __args)
    {
        _apply_impl<void>::template apply_access_with_indices<0, _N - 1, _Access, _Tuple,
                                                              _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _TupleA, typename _TupleB, typename... _Args,
              std::size_t _N  = std::tuple_size<decay_t<_TupleA>>::value,
              std::size_t _Nb = std::tuple_size<decay_t<_TupleB>>::value>
    static void access2(_TupleA&& __ta, _TupleB&& __tb, _Args&&... __args)
    {
        static_assert(_N == _Nb, "tuple_size 1 must match tuple_size 2");
        _apply_impl<void>::template apply_access2<0, _N - 1, _Access, _TupleA, _TupleB,
                                                  _Args...>(
            std::forward<_TupleA>(__ta), std::forward<_TupleB>(__tb),
            std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, typename _Tuple, typename... _Args,
              std::size_t _N = std::tuple_size<decay_t<_Tuple>>::value>
    static void unroll_access(_Tuple&& __t, _Args&&... __args)
    {
        _apply_impl<void>::template unroll_access<_Access, _N - 1, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, typename _Tuple, typename... _Args,
              std::size_t _N = std::tuple_size<decay_t<_Tuple>>::value>
    static void type_access(_Args&&... __args)
    {
        _apply_impl<void>::template type_access<_Access, _N - 1, _Tuple, _Args...>(
            std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
    /*
    template <typename _Tuple, typename _Funct, typename... _Args,
              std::size_t _Nt = std::tuple_size<decay_t<_Tuple>>::value,
              std::size_t _Nf = std::tuple_size<decay_t<_Funct>>::value>
    static void functions(_Tuple&& __t, _Funct&& __f, _Args&&... __args)
    {
        static_assert(_Nt == _Nf,
                      "tuple_size of objects must match tuple_size of functions");
        _apply_impl<void>::template functions<0, _Nt - 1, _Tuple, _Funct, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Funct>(__f),
            std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, typename _Funct, typename... _Args>
    static void unroll(_Funct&& __f, _Args&&... __args)
    {
        _apply_impl<void>::template unroll<0, _N - 1, _Funct, _Args...>(
            std::forward<_Funct>(__f), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Tuple, typename _Funct,
              std::size_t _N = std::tuple_size<decay_t<_Funct>>::value, typename... _Args>
    static void unroll_indices(_Tuple&& __t, _Funct&& __f, _Args&&... args)
    {
        _apply_impl<void>::template unroll_indices<0, _N - 1, _Tuple, _Funct, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Funct>(__f),
            std::forward<_Args>(args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, typename _Tuple, typename _Funct, typename... _Args>
    static void unroll_members(_Tuple&& __t, _Funct&& __f, _Args&&... __args)
    {
        _apply_impl<void>::template unroll_members<0, _N - 1, _Tuple, _Funct, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Funct>(__f),
            std::forward<_Args>(__args)...);
    }
    */
    //----------------------------------------------------------------------------------//
};

//======================================================================================//

}  // namespace tim

//======================================================================================//
