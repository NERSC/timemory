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

/** \file mpl/apply.hpp
 * \headerfile mpl/apply.hpp "timemory/mpl/apply.hpp"
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

#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/stl_overload.hpp"

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

namespace internal
{
//======================================================================================//

template <typename _Ret>
struct apply
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
struct apply<void>
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
    //  per-element percent difference calculation
    //
    template <typename _Tuple, size_t _Idx, size_t... _Nt,
              enable_if_t<(sizeof...(_Nt) == 0), char> = 0>
    static void percent_diff(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs)
    {
        using value_type = decay_t<decltype(std::get<_Idx>(_ret))>;
        math::compute<value_type>::percent_diff(
            std::get<_Idx>(_ret), std::get<_Idx>(_lhs), std::get<_Idx>(_rhs));
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

}  // namespace internal

//======================================================================================//
//
//      Declaration
//
//======================================================================================//

template <typename _Ret>
struct apply;

//======================================================================================//
//
//          apply --> std::string
//
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
        return internal::apply<_Ret>::template join<_Sep, _Args...>(
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
//
//          apply --> generic
//
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
        return internal::apply<_Ret>::template invoke<_Fn, _Tuple_t>(
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
//
//          apply --> std::tuple<std::string>
//
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
    struct impl
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
        return impl::join(std::forward<_LabelSep>(_label_sep),
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
//
//          apply --> void
//
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
        internal::apply<_Ret>::template invoke<_Fn, _Tuple_t>(std::forward<_Fn>(__f),
                                                              std::forward<_Tuple_t>(__t),
                                                              make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//
    //  per-element addition
    //
    template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
    static void plus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        return stl_overload::mpl::plus(_lhs, _rhs);
    }

    //----------------------------------------------------------------------------------//
    //  per-element subtraction
    //
    template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
    static void minus(_Tuple& _lhs, const _Tuple& _rhs)
    {
        return stl_overload::mpl::minus(_lhs, _rhs);
    }

    //----------------------------------------------------------------------------------//
    //  per-element percent difference
    //
    template <typename _Tuple, size_t _N = std::tuple_size<_Tuple>::value>
    static void percent_diff(_Tuple& _ret, const _Tuple& _lhs, const _Tuple& _rhs)
    {
        internal::apply<_Ret>::template percent_diff<_Tuple>(_ret, _lhs, _rhs,
                                                             make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Device, typename _Func, typename... _Args,
              enable_if_t<std::is_same<_Device, device::gpu>::value, char> = 0>
    TIMEMORY_LAMBDA static void unroll(_Func&& __func, _Args&&... __args)
    {
        internal::apply<void>::template unroll<_N, _Device, _Func, _Args...>(
            std::forward<_Func>(__func), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Device, typename _Func, typename... _Args,
              enable_if_t<std::is_same<_Device, device::cpu>::value, char> = 0>
    static void unroll(_Func&& __func, _Args&&... __args)
    {
        internal::apply<void>::template unroll<_N, _Device, _Func, _Args...>(
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
        internal::apply<void>::template set_value<0, _N - 1, _Tuple, _Value>(
            std::forward<_Tuple>(__t), std::forward<_Value>(__v));
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void access(_Tuple&& __t, _Args&&... __args)
    {
        internal::apply<void>::template apply_access<0, _N - 1, _Access, _Tuple,
                                                     _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
        // internal::apply<void>::template variadic_1d<_Access, _Tuple, _Args...>(
        //    std::forward<_Tuple>(__t), std::forward<_Args>(__args)...,
        //    make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//

    template <typename _Access, typename _Tuple, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void access_with_indices(_Tuple&& __t, _Args&&... __args)
    {
        internal::apply<void>::template apply_access_with_indices<0, _N - 1, _Access,
                                                                  _Tuple, _Args...>(
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
        internal::apply<void>::template apply_access2<0, _N - 1, _Access, _TupleA,
                                                      _TupleB, _Args...>(
            std::forward<_TupleA>(__ta), std::forward<_TupleB>(__tb),
            std::forward<_Args>(__args)...);
        // internal::apply<void>::template variadic_2d<_Access, _TupleA, _TupleB,
        // _Args...>(
        //    std::forward<_TupleA>(__ta), std::forward<_TupleB>(__tb),
        //    std::forward<_Args>(__args)..., make_index_sequence<_N>{});
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, typename _Tuple, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void unroll_access(_Tuple&& __t, _Args&&... __args)
    {
        internal::apply<void>::template unroll_access<_Access, _N - 1, _Tuple, _Args...>(
            std::forward<_Tuple>(__t), std::forward<_Args>(__args)...);
    }

    //----------------------------------------------------------------------------------//

    template <template <typename> class _Access, typename _Tuple, typename... _Args,
              std::size_t _N             = std::tuple_size<decay_t<_Tuple>>::value,
              enable_if_t<(_N > 0), int> = 0>
    static void type_access(_Args&&... __args)
    {
        internal::apply<void>::template type_access<_Access, _N - 1, _Tuple, _Args...>(
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
