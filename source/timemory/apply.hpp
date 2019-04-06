//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California,
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

/** \file timemory.hpp
 * \headerfile timemory.hpp "timemory/timemory.hpp"
 * All-inclusive timemory header + extern declarations
 *
 */

#pragma once

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
template <size_t _Num>
struct _Build_index_tuple
: _Itup_cat<typename _Build_index_tuple<_Num / 2>::__type,
            typename _Build_index_tuple<_Num - _Num / 2>::__type>
{
};

//--------------------------------------------------------------------------------------//

template <>
struct _Build_index_tuple<1>
{
    typedef _Index_tuple<0> __type;
};

//--------------------------------------------------------------------------------------//

template <>
struct _Build_index_tuple<0>
{
    typedef _Index_tuple<> __type;
};

//--------------------------------------------------------------------------------------//

/// Class template integer_sequence
template <typename _Tp, _Tp... _Idx>
struct integer_sequence
{
    typedef _Tp             value_type;
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

    typedef integer_sequence<_Tp, static_cast<_Tp>(_Idx)...> __type;
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

template <typename _Tp, typename _Tuple>
struct index_of;

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... Types>
struct index_of<_Tp, std::tuple<_Tp, Types...>>
{
    static constexpr std::size_t value = 0;
};

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Head, typename... Tail>
struct index_of<_Tp, std::tuple<Head, Tail...>>
{
    static constexpr std::size_t value = 1 + index_of<_Tp, std::tuple<Tail...>>::value;
};

//======================================================================================//

template <typename _Ret>
struct _apply_impl
{
    template <typename _Fn, typename _Tuple, size_t... _Idx>
    static _Ret all(_Fn&& __f, _Tuple&& __t, index_sequence<_Idx...>)
    {
        return __f(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }
};

//======================================================================================//

template <>
struct _apply_impl<void>
{
    //----------------------------------------------------------------------------------//

    template <typename _Fn, typename _Tuple, size_t... _Idx>
    static void all(_Fn&& __f, _Tuple&& __t, index_sequence<_Idx...>)
    {
        __f(std::get<_Idx>(std::forward<_Tuple>(__t))...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename... _Args,
              enable_if_t<(_N == _Nt), int> = 0>
    static void loop(_Tuple&& __t, _Args&&... __args)
    {
        // call operator()
        std::get<_N>(__t)(std::forward<_Args>(__args)...);
    }

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename... _Args,
              enable_if_t<(_N < _Nt), int> = 0>
    static void loop(_Tuple&& __t, _Args&&... __args)
    {
        // call operator()
        std::get<_N>(__t)(std::forward<_Args>(__args)...);
        // recursive call
        loop<_N + 1, _Nt, _Tuple, _Args...>(std::forward<_Tuple>(__t),
                                            std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
              typename... _Args, enable_if_t<(_N == _Nt), int> = 0>
    static void apply_access(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(std::forward<Type>(std::get<_N>(__t)), std::forward<_Args>(__args)...);
    }

    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
              typename... _Args, enable_if_t<(_N < _Nt), int> = 0>
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
              typename... _Args, enable_if_t<(_N == _Nt), int> = 0>
    static void apply_access_with_indices(_Tuple&& __t, _Args&&... __args)
    {
        // call constructor
        using Type       = decltype(std::get<_N>(__t));
        using AccessType = typename std::tuple_element<_N, _Access>::type;
        AccessType(_N, _Nt + 1, std::forward<Type>(std::get<_N>(__t)),
                   std::forward<_Args>(__args)...);
    }

    template <std::size_t _N, std::size_t _Nt, typename _Access, typename _Tuple,
              typename... _Args, enable_if_t<(_N < _Nt), int> = 0>
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
              typename _TupleB, typename... _Args, enable_if_t<(_N == _Nt), int> = 0>
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
              typename _TupleB, typename... _Args, enable_if_t<(_N < _Nt), int> = 0>
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

    template <typename _Tp, typename _Funct, typename... _Args,
              enable_if_t<std::is_pointer<_Tp>::value, int> = 0>
    static void apply_function(_Tp&& __t, _Funct&& __f, _Args&&... __args)
    {
        (__t)->*(__f)(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Tp, typename _Funct, typename... _Args,
              enable_if_t<!std::is_pointer<_Tp>::value, int> = 0>
    static void apply_function(_Tp&& __t, _Funct&& __f, _Args&&... __args)
    {
        (__t).*(__f)(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N == _Nt), int> = 0>
    static void functions(_Tuple&& __t, _Funct&& __f, _Args&&... __args)
    {
        // call member function at index _N
        apply_function(std::get<_N>(__t), std::get<_N>(__f),
                       std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N < _Nt), int> = 0>
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
              enable_if_t<(_N == _Nt), int> = 0>
    static void unroll(_Funct&& __f, _Args&&... __args)
    {
        (__f)(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Funct, typename... _Args,
              enable_if_t<(_N < _Nt), int> = 0>
    static void unroll(_Funct&& __f, _Args&&... __args)
    {
        (__f)(std::forward<_Args>(__args)...);
        unroll<_N + 1, _Nt, _Funct, _Args...>(std::forward<_Funct>(__f),
                                              std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N == _Nt), int> = 0>
    static void unroll_indices(_Tuple&& __t, _Funct&& __f, _Args&&... args)
    {
        using Type = decltype(std::get<_N>(__t));
        (__f)(_N, std::forward<Type>(std::get<_N>(__t)), std::forward<_Args>(args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N < _Nt), int> = 0>
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
              typename... _Args, enable_if_t<(_N == _Nt), int> = 0>
    static void unroll_members(_Tuple&& __t, _Funct&& __f, _Args&&... __args)
    {
        (std::get<_N>(__t)).*(std::get<_N>(__f))(std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <std::size_t _N, std::size_t _Nt, typename _Tuple, typename _Funct,
              typename... _Args, enable_if_t<(_N < _Nt), int> = 0>
    static void unroll_members(_Tuple&& __t, _Funct&& __f, _Args&&... __args)
    {
        (std::get<_N>(__t)).*(std::get<_N>(__f))(std::forward<_Args>(__args)...);
        unroll_members<_N + 1, _Nt, _Tuple, _Funct, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Funct>(__f),
            std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//
};

//======================================================================================//

template <typename _Ret>
struct apply
{
    template <typename _Fn, typename _Tuple,
              std::size_t _N    = std::tuple_size<decay_t<_Tuple>>::value,
              typename _Indices = make_index_sequence<_N>>
    static _Ret all(_Fn&& __f, _Tuple&& __t)
    {
        return _apply_impl<_Ret>::template all<_Fn, _Tuple>(
            std::forward<_Fn>(__f), std::forward<_Tuple>(__t), _Indices{});
    }
};

//======================================================================================//

template <>
struct apply<void>
{
    //----------------------------------------------------------------------------------//

    template <typename _Fn, typename _Tuple,
              std::size_t _N    = std::tuple_size<decay_t<_Tuple>>::value,
              typename _Indices = make_index_sequence<_N>>
    static void all(_Fn&& __f, _Tuple&& __t)
    {
        _apply_impl<void>::template all<_Fn, _Tuple>(
            std::forward<_Fn>(__f), std::forward<_Tuple>(__t), _Indices{});
    }

    //----------------------------------------------------------------------------------//

    template <typename _Tuple, typename... _Args,
              std::size_t _N = std::tuple_size<decay_t<_Tuple>>::value>
    static void loop(_Tuple&& __t, _Args&&... __args)
    {
        _apply_impl<void>::template loop<0, _N - 1, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

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

    //----------------------------------------------------------------------------------//
};

//======================================================================================//

}  // namespace tim

//======================================================================================//
