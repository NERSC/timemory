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
{
};

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

template <typename... _ImplTypes>
struct get_data_tuple<std::tuple<_ImplTypes...>>
{
    using value_type = std::tuple<decltype(std::declval<_ImplTypes>().get())...>;
    using label_type = std::tuple<
        std::tuple<std::string, decltype(std::declval<_ImplTypes>().get())>...>;
};

//======================================================================================//

}  // namespace impl

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
