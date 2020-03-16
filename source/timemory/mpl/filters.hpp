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

#include "timemory/components/types.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/variadic/types.hpp"

#include <tuple>

namespace tim
{
namespace impl
{
//======================================================================================//
//
//      get data tuple
//
//======================================================================================//

template <typename T>
struct get_data_tuple_type
{
    static_assert(!std::is_fundamental<T>::value,
                  "get_data_tuple_type called for fundamental type");
    using get_type = decltype(std::declval<T>().get());

    static constexpr bool is_void_v = std::is_void<get_type>::value;

    using type       = conditional_t<(is_void_v), type_list<>, type_list<T>>;
    using value_type = conditional_t<(is_void_v), type_list<>, type_list<get_type>>;
    using label_type = conditional_t<(is_void_v), type_list<>,
                                     type_list<std::tuple<std::string, get_type>>>;
};

template <>
struct get_data_tuple_type<std::tuple<>>
{
    using type       = type_list<>;
    using value_type = type_list<>;
    using label_type = type_list<>;
};

template <>
struct get_data_tuple_type<type_list<>>
{
    using type       = type_list<>;
    using value_type = type_list<>;
    using label_type = type_list<>;
};

//--------------------------------------------------------------------------------------//

template <typename... Types>
struct get_data_tuple
{
    using type = convert_t<type_concat_t<typename get_data_tuple_type<Types>::type...>,
                           std::tuple<>>;
    using value_type =
        convert_t<type_concat_t<typename get_data_tuple_type<Types>::value_type...>,
                  std::tuple<>>;
    using label_type =
        convert_t<type_concat_t<typename get_data_tuple_type<Types>::label_type...>,
                  std::tuple<>>;
};

template <typename... Types>
struct get_data_tuple<type_list<Types...>> : public get_data_tuple<Types...>
{};

template <typename... Types>
struct get_data_tuple<std::tuple<Types...>> : public get_data_tuple<Types...>
{};

template <>
struct get_data_tuple<std::tuple<>>
{
    using type       = std::tuple<>;
    using value_type = std::tuple<>;
    using label_type = std::tuple<>;
};

template <>
struct get_data_tuple<type_list<>>
{
    using type       = std::tuple<>;
    using value_type = std::tuple<>;
    using label_type = std::tuple<>;
};

//======================================================================================//
// check if type is in expansion
//
template <typename...>
struct is_one_of
{
    static constexpr bool value = false;
};

template <typename F, typename S, template <typename...> class Tuple, typename... T>
struct is_one_of<F, S, Tuple<T...>>
{
    static constexpr bool value =
        std::is_same<F, S>::value || is_one_of<F, Tuple<T...>>::value;
};

template <typename F, typename S, template <typename...> class Tuple, typename... T>
struct is_one_of<F, Tuple<S, T...>>
{
    static constexpr bool value = is_one_of<F, S, Tuple<T...>>::value;
};

//======================================================================================//
// check if trait is satisfied by at least one type in variadic sequence
//
template <template <typename> class Test, typename Sequence>
struct contains_one_of;

template <template <typename> class Test, template <typename...> class Tuple>
struct contains_one_of<Test, Tuple<>>
{
    static constexpr bool value = false;
    using type                  = Tuple<>;
};

template <template <typename> class Test, typename F, template <typename...> class Tuple,
          typename... T>
struct contains_one_of<Test, Tuple<F, T...>>
{
    static constexpr bool value =
        Test<F>::value || contains_one_of<Test, Tuple<T...>>::value;
    using type = conditional_t<(Test<F>::value), F,
                               typename contains_one_of<Test, Tuple<T...>>::type>;
};

//======================================================================================//
// check if any types are integral types
//
template <typename...>
struct is_one_of_integral
{
    static constexpr bool value = false;
};

template <typename T, template <typename...> class Tuple, typename... Tail>
struct is_one_of_integral<Tuple<T, Tail...>>
{
    static constexpr bool value =
        std::is_integral<T>::value || is_one_of_integral<Tuple<Tail...>>::value;
};

//======================================================================================//

template <typename In, typename Out>
struct remove_duplicates;

template <typename Out>
struct remove_duplicates<type_list<>, Out>
{
    using type = Out;
};

template <typename In, typename... InTail, typename... Out>
struct remove_duplicates<type_list<In, InTail...>, type_list<Out...>>
{
    using type = conditional_t<
        !(is_one_of<In, type_list<Out...>>::value),
        typename remove_duplicates<type_list<InTail...>, type_list<Out..., In>>::type,
        typename remove_duplicates<type_list<InTail...>, type_list<Out...>>::type>;
};

//--------------------------------------------------------------------------------------//

template <typename In, typename Out>
struct unique;

template <template <typename...> class InTuple, typename... In,
          template <typename...> class OutTuple, typename... Out>
struct unique<InTuple<In...>, OutTuple<Out...>>
{
    using tuple_type = convert_t<InTuple<In...>, OutTuple<>>;
    using dupl_type  = typename remove_duplicates<tuple_type, OutTuple<>>::type;
    using type       = convert_t<dupl_type, InTuple<>>;
};

//======================================================================================//

template <template <typename> class _Prio, typename _Beg, typename Tp, typename _End>
struct sortT;

//--------------------------------------------------------------------------------------//

template <template <typename> class _Prio, typename Tuple, typename _Beg = type_list<>,
          typename _End = type_list<>>
using sort = typename sortT<_Prio, Tuple, _Beg, _End>::type;

//--------------------------------------------------------------------------------------//
//  Initiate recursion (zeroth sort operation)
//
template <template <typename> class _Prio, typename _In, typename... _InT>
struct sortT<_Prio, type_list<_In, _InT...>, type_list<>, type_list<>>
{
    using type =
        typename sortT<_Prio, type_list<_InT...>, type_list<>, type_list<_In>>::type;
};

//--------------------------------------------------------------------------------------//
//  Initiate recursion (zeroth sort operation)
//
template <template <typename> class _Prio, typename _In, typename... _InT>
struct sortT<_Prio, type_list<type_list<_In, _InT...>>, type_list<>, type_list<>>
{
    using type =
        typename sortT<_Prio, type_list<_InT...>, type_list<>, type_list<_In>>::type;
};

//--------------------------------------------------------------------------------------//
//  Terminate recursion (last sort operation)
//
template <template <typename> class _Prio, typename... _BegT, typename... _EndT>
struct sortT<_Prio, type_list<>, type_list<_BegT...>, type_list<_EndT...>>
{
    using type = type_list<_BegT..., _EndT...>;
};

//--------------------------------------------------------------------------------------//
//  If no current end, transfer begin to end ()
//
template <template <typename> class _Prio, typename _In, typename... _InT,
          typename... _BegT>
struct sortT<_Prio, type_list<_In, _InT...>, type_list<_BegT...>, type_list<>>
{
    using type = typename sortT<_Prio, type_list<_In, _InT...>, type_list<>,
                                type_list<_BegT...>>::type;
};

//--------------------------------------------------------------------------------------//
//  Specialization for first sort operation
//
template <template <typename> class _Prio, typename _In, typename Tp, typename... _InT>
struct sortT<_Prio, type_list<_In, _InT...>, type_list<>, type_list<Tp>>
{
    static constexpr bool value = (_Prio<_In>::value < _Prio<Tp>::value);

    using type = conditional_t<
        (value),
        typename sortT<_Prio, type_list<_InT...>, type_list<>, type_list<_In, Tp>>::type,
        typename sortT<_Prio, type_list<_InT...>, type_list<>, type_list<Tp, _In>>::type>;
};

//--------------------------------------------------------------------------------------//
//  Specialization for second sort operation
//
template <template <typename> class _Prio, typename _In, typename _Ta, typename _Tb,
          typename... _BegT, typename... _InT>
struct sortT<_Prio, type_list<_In, _InT...>, type_list<_BegT...>, type_list<_Ta, _Tb>>
{
    static constexpr bool iavalue = (_Prio<_In>::value < _Prio<_Ta>::value);
    static constexpr bool ibvalue = (_Prio<_In>::value < _Prio<_Tb>::value);
    static constexpr bool abvalue = (_Prio<_Ta>::value <= _Prio<_Tb>::value);

    using type = conditional_t<
        (iavalue),
        typename sortT<
            _Prio, type_list<_InT...>, sort<_Prio, type_list<_BegT..., _In>>,
            conditional_t<(abvalue), type_list<_Ta, _Tb>, type_list<_Tb, _Ta>>>::type,
        typename sortT<
            _Prio, type_list<_InT...>, sort<_Prio, type_list<_BegT..., _Ta>>,
            conditional_t<(ibvalue), type_list<_In, _Tb>, type_list<_Tb, _In>>>::type>;
};

//--------------------------------------------------------------------------------------//
//  Specialization for all other sort operations after first and second
//
template <template <typename> class _Prio, typename _In, typename Tp, typename... _InT,
          typename... _BegT, typename... _EndT>
struct sortT<_Prio, type_list<_In, _InT...>, type_list<_BegT...>, type_list<Tp, _EndT...>>
{
    static constexpr bool value = (_Prio<_In>::value < _Prio<Tp>::value);

    using type =
        conditional_t<(value),
                      typename sortT<_Prio, type_list<_InT...>, type_list<>,
                                     type_list<_BegT..., _In, Tp, _EndT...>>::type,
                      typename sortT<_Prio, type_list<_In, _InT...>,
                                     type_list<_BegT..., Tp>, type_list<_EndT...>>::type>;
};

//======================================================================================//

}  // namespace impl

//======================================================================================//

///
/// check if type is in expansion
///
template <typename Tp, typename Types>
using is_one_of = typename impl::is_one_of<Tp, Types>;

///
/// check if type is in expansion
///
template <template <typename> class Predicate, typename Types>
using contains_one_of = typename impl::contains_one_of<Predicate, Types>;

template <template <typename> class Predicate, typename Types>
using contains_one_of_t = typename contains_one_of<Predicate, Types>::type;

//======================================================================================//

///
/// check if type is in expansion
///
template <typename Types>
using is_one_of_integral = typename impl::is_one_of_integral<Types>;

//======================================================================================//

template <typename T>
using remove_duplicates = typename impl::unique<T, type_list<>>::type;

template <typename T>
using unique = typename impl::unique<T, type_list<>>::type;

//======================================================================================//
//
//      counters
//
//======================================================================================//

/// filter out any types that are not available
template <typename... Types>
using filter_gotchas = impl::filter_false<trait::is_gotcha, std::tuple<Types...>>;

template <typename T>
using filter_gotchas_t = impl::filter_false<trait::is_gotcha, T>;

template <typename T>
using filter_empty_t = impl::filter_true<concepts::is_empty, T>;

//======================================================================================//
//
//      {auto,component}_{hybrid,list,tuple} get() and get_labeled() types
//
//======================================================================================//

/// get the tuple of types
template <typename TypeList>
using get_data_type_t = typename impl::template get_data_tuple<TypeList>::type;

/// get the tuple of values
template <typename TypeList>
using get_data_value_t = typename impl::template get_data_tuple<TypeList>::value_type;

/// get the tuple of pair of descriptor and value
template <typename TypeList>
using get_data_label_t = typename impl::template get_data_tuple<TypeList>::label_type;

//======================================================================================//
//
//      sort
//
//======================================================================================//

namespace mpl
{
template <template <typename> class _Prio, typename Tuple, typename _Beg = type_list<>,
          typename _End = type_list<>>
using sort = convert_t<typename impl::sortT<_Prio, convert_t<Tuple, type_list<>>,
                                            convert_t<_Beg, type_list<>>,
                                            convert_t<_End, type_list<>>>::type,
                       std::tuple<>>;

}  // namespace mpl

template <typename Tp, typename _Op>
struct negative_priority;

template <typename Tp, typename _Op>
struct positive_priority;

template <typename Tp, template <typename> class _Op>
struct negative_priority<Tp, _Op<Tp>>
{
    static constexpr bool value = (_Op<Tp>::value < 0);
};

template <typename Tp, template <typename> class _Op>
struct positive_priority<Tp, _Op<Tp>>
{
    static constexpr bool value = (_Op<Tp>::value > 0);
};

template <typename Tp>
struct negative_start_priority : negative_priority<Tp, trait::start_priority<Tp>>
{};

template <typename Tp>
struct positive_start_priority : positive_priority<Tp, trait::start_priority<Tp>>
{};

template <typename Tp>
struct negative_stop_priority : negative_priority<Tp, trait::stop_priority<Tp>>
{};

template <typename Tp>
struct positive_stop_priority : positive_priority<Tp, trait::stop_priority<Tp>>
{};

//--------------------------------------------------------------------------------------//

}  // namespace tim
