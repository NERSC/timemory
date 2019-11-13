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

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/mpl/type_traits.hpp"

#include <tuple>

namespace tim
{
namespace impl
{
//======================================================================================//
//
//      tuple concatenation
//
//======================================================================================//

template <typename... Types>
struct tuple_concat
{
    using type = std::tuple<Types...>;
};

//--------------------------------------------------------------------------------------//

template <>
struct tuple_concat<>
{
    using type = std::tuple<>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts>
struct tuple_concat<std::tuple<Ts...>>
{
    using type = std::tuple<Ts...>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts0, typename... Ts1, typename... Rest>
struct tuple_concat<std::tuple<Ts0...>, std::tuple<Ts1...>, Rest...>
: tuple_concat<std::tuple<Ts0..., Ts1...>, Rest...>
{};

//--------------------------------------------------------------------------------------//

}  // namespace impl

//--------------------------------------------------------------------------------------//

template <typename... Ts>
using tuple_concat_t = typename impl::tuple_concat<Ts...>::type;

//--------------------------------------------------------------------------------------//

namespace impl
{
//======================================================================================//
//
//      filter if predicate evaluates to false (result)
//
//======================================================================================//

template <bool>
struct filter_if_false_result
{
    template <typename T>
    using type = std::tuple<T>;

    template <template <typename...> class Operator, typename T, typename... Tail>
    using operation_type = std::tuple<Operator<T, Tail...>>;
};

//--------------------------------------------------------------------------------------//

template <>
struct filter_if_false_result<false>
{
    template <typename T>
    using type = std::tuple<>;

    template <template <typename...> class Operator, typename T, typename... Tail>
    using operation_type = std::tuple<>;
};

//======================================================================================//
//
//      filter if predicate evaluates to true (result)
//
//======================================================================================//

template <bool>
struct filter_if_true_result
{
    template <typename T>
    using type = std::tuple<T>;

    template <template <typename...> class Operator, typename T, typename... Tail>
    using operation_type = std::tuple<Operator<T, Tail...>>;
};

//--------------------------------------------------------------------------------------//

template <>
struct filter_if_true_result<true>
{
    template <typename T>
    using type = std::tuple<>;

    template <template <typename...> class Operator, typename T, typename... Tail>
    using operation_type = std::tuple<>;
};

//======================================================================================//
//
//      filter if predicate evaluates to false (operator)
//
//======================================================================================//

template <template <typename> class Predicate, typename Sequence>
struct filter_if_false;

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_false<Predicate, std::tuple<Ts...>>
{
    using type = tuple_concat_t<
        typename filter_if_false_result<Predicate<Ts>::value>::template type<Ts>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename Sequence>
using filter_false = typename filter_if_false<Predicate, Sequence>::type;

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename... Ts>
struct operation_filter_if_false
{
    using type = tuple_concat_t<typename filter_if_false_result<
        Predicate<Ts>::value>::template operation_type<Operator, Ts>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename... Ts>
struct operation_filter_if_false<Predicate, Operator, std::tuple<Ts...>>
{
    using type = tuple_concat_t<typename filter_if_false_result<
        Predicate<Ts>::value>::template operation_type<Operator, Ts>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename... Ts>
struct operation_filter_if_false<Predicate, Operator, std::tuple<std::tuple<Ts...>>>
{
    using type = tuple_concat_t<typename filter_if_false_result<
        Predicate<Ts>::value>::template operation_type<Operator, Ts>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename Sequence>
using operation_filter_false =
    typename operation_filter_if_false<Predicate, Operator, Sequence>::type;

//======================================================================================//
//
//      filter if predicate evaluates to true (operator)
//
//======================================================================================//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_true
{
    using type = tuple_concat_t<
        typename filter_if_true_result<Predicate<Ts>::value>::template type<Ts>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_true<Predicate, std::tuple<Ts...>>
{
    using type = tuple_concat_t<
        typename filter_if_true_result<Predicate<Ts>::value>::template type<Ts>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_true<Predicate, std::tuple<std::tuple<Ts...>>>
{
    using type = tuple_concat_t<
        typename filter_if_true_result<Predicate<Ts>::value>::template type<Ts>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename Sequence>
using filter_true = typename filter_if_true<Predicate, Sequence>::type;

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename Sequence>
struct operation_filter_if_true;

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename... Ts>
struct operation_filter_if_true<Predicate, Operator, std::tuple<Ts...>>
{
    using type = tuple_concat_t<typename filter_if_true_result<
        Predicate<Ts>::value>::template operation_type<Operator, Ts>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename Sequence>
using operation_filter_true =
    typename operation_filter_if_true<Predicate, Operator, Sequence>::type;

//======================================================================================//
//
//      get data tuple
//
//======================================================================================//

template <typename... _ImplTypes>
struct get_data_tuple
{
    using value_type = std::tuple<_ImplTypes...>;
    using label_type = std::tuple<std::tuple<std::string, _ImplTypes>...>;
};

template <typename... _ImplTypes, template <typename...> class _Tuple>
struct get_data_tuple<_Tuple<_ImplTypes...>>
{
    using value_type = _Tuple<decltype(std::declval<_ImplTypes>().get())...>;
    using label_type =
        _Tuple<_Tuple<std::string, decltype(std::declval<_ImplTypes>().get())>...>;
};

//======================================================================================//
// check if type is in expansion
//
template <typename...>
struct is_one_of
{
    static constexpr bool value = false;
};

template <typename F, typename S, template <typename...> class _Tuple, typename... T>
struct is_one_of<F, S, _Tuple<T...>>
{
    static constexpr bool value =
        std::is_same<F, S>::value || is_one_of<F, _Tuple<T...>>::value;
};

template <typename F, typename S, template <typename...> class _Tuple, typename... T>
struct is_one_of<F, _Tuple<S, T...>>
{
    static constexpr bool value = is_one_of<F, S, _Tuple<T...>>::value;
};

//======================================================================================//

template <typename In, typename Out>
struct remove_duplicates;

template <typename Out>
struct remove_duplicates<std::tuple<>, Out>
{
    using type = Out;
};

template <typename In, typename... InTail, typename... Out>
struct remove_duplicates<std::tuple<In, InTail...>, std::tuple<Out...>>
{
    using type = typename std::conditional<
        !(is_one_of<In, std::tuple<Out...>>::value),
        typename remove_duplicates<std::tuple<InTail...>, std::tuple<Out..., In>>::type,
        typename remove_duplicates<std::tuple<InTail...>,
                                   std::tuple<Out...>>::type>::type;
};

//======================================================================================//

}  // namespace impl

//======================================================================================//

///
/// get the index of a type in expansion
///
template <typename _Tp, typename Type>
struct index_of;

template <typename _Tp, template <typename...> class _Tuple, typename... Types>
struct index_of<_Tp, _Tuple<_Tp, Types...>>
{
    static constexpr std::size_t value = 0;
};

template <typename _Tp, typename Head, template <typename...> class _Tuple,
          typename... Tail>
struct index_of<_Tp, _Tuple<Head, Tail...>>
{
    static constexpr std::size_t value = 1 + index_of<_Tp, _Tuple<Tail...>>::value;
};

//======================================================================================//

///
/// check if type is in expansion
///
template <typename _Tp, typename _Types>
using is_one_of = typename impl::is_one_of<_Tp, _Types>;

//======================================================================================//

template <typename T>
using remove_duplicates = typename impl::remove_duplicates<std::tuple<>, T>::type;

//======================================================================================//
//
//      determines if storage should be implemented
//
//======================================================================================//

/// filter out any types that are not available
template <typename... Types>
using implemented = impl::filter_false<trait::is_available, std::tuple<Types...>>;

template <typename _Tuple>
using available_tuple = impl::filter_false<trait::is_available, _Tuple>;

/// filter out any operations on types that are not available
template <template <typename...> class Operator, typename... Types>
using modifiers =
    impl::operation_filter_false<trait::is_available, Operator, std::tuple<Types...>>;

//======================================================================================//
//
//      trait::num_gotchas
//
//======================================================================================//

/// filter out any types that are not available
template <typename... Types>
using filter_gotchas = impl::filter_false<trait::is_gotcha, std::tuple<Types...>>;

//======================================================================================//
//
//      {auto,component}_{hybrid,list,tuple} get() and get_labeled() types
//
//======================================================================================//

/// get the tuple of values
template <typename _Tuple>
using get_data_value_t = typename impl::template get_data_tuple<_Tuple>::value_type;

/// get the tuple of pair of descriptor and value
template <typename _Tuple>
using get_data_label_t = typename impl::template get_data_tuple<_Tuple>::label_type;

}  // namespace tim
