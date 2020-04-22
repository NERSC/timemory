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

#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/variadic/types.hpp"

#include <tuple>

namespace tim
{
namespace impl
{
//======================================================================================//
//
//      filter if predicate evaluates to false (result)
//
//======================================================================================//

template <bool, typename T>
struct filter_if_false_result;

//--------------------------------------------------------------------------------------//

template <typename T>
struct filter_if_false_result<true, T>
{
    using type = std::tuple<T>;
    template <template <typename...> class Op>
    using operation_type = Op<T>;
};

//--------------------------------------------------------------------------------------//

template <typename T>
struct filter_if_false_result<false, T>
{
    using type = std::tuple<>;
    template <template <typename...> class Op>
    using operation_type = std::tuple<>;
};

//======================================================================================//
//
//      filter if predicate evaluates to true (result)
//
//======================================================================================//

template <bool, typename T>
struct filter_if_true_result;

template <typename T>
struct filter_if_true_result<false, T>
{
    using type = std::tuple<T>;
    template <template <typename...> class Op>
    using operation_type = Op<T>;
};

//--------------------------------------------------------------------------------------//

template <typename T>
struct filter_if_true_result<true, T>
{
    using type = std::tuple<>;
    template <template <typename...> class Op>
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
        typename filter_if_false_result<Predicate<Ts>::value, Ts>::type...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Tuple,
          typename... Ts>
struct filter_if_false<Predicate, Tuple<Ts...>>
{
    using type =
        convert_t<tuple_concat_t<
                      typename filter_if_false_result<Predicate<Ts>::value, Ts>::type...>,
                  Tuple<>>;
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
        Predicate<Ts>::value, Ts>::template operation_type<Operator>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename... Ts>
struct operation_filter_if_false<Predicate, Operator, std::tuple<Ts...>>
{
    using type = tuple_concat_t<typename filter_if_false_result<
        Predicate<Ts>::value, Ts>::template operation_type<Operator>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          template <typename...> class Tuple, typename... Ts>
struct operation_filter_if_false<Predicate, Operator, Tuple<Ts...>>
{
    using type = tuple_concat_t<typename filter_if_false_result<
        Predicate<Ts>::value, Ts>::template operation_type<Operator>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename... Ts>
struct operation_filter_if_false<Predicate, Operator, type_list<Ts...>>
{
    using type = tuple_concat_t<typename filter_if_false_result<
        Predicate<Ts>::value, Ts>::template operation_type<Operator>...>;
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
    using type =
        tuple_concat_t<typename filter_if_true_result<Predicate<Ts>::value, Ts>::type...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_true<Predicate, std::tuple<Ts...>>
{
    using type =
        tuple_concat_t<typename filter_if_true_result<Predicate<Ts>::value, Ts>::type...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Ts>
struct filter_if_true<Predicate, type_list<Ts...>>
{
    using type =
        tuple_concat_t<typename filter_if_true_result<Predicate<Ts>::value, Ts>::type...>;
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
        Predicate<Ts>::value, Ts>::template operation_type<Operator>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, template <typename...> class Operator,
          typename Sequence>
using operation_filter_true =
    typename operation_filter_if_true<Predicate, Operator, Sequence>::type;

//======================================================================================//
//
}  // namespace impl

//======================================================================================//

///
///  generic alias for extracting all types with a specified trait enabled
///
template <template <typename> class Predicate, typename... Sequence>
struct get_true_types
{
    using type = impl::filter_false<Predicate, std::tuple<Sequence...>>;
};

template <template <typename> class Predicate, typename... Sequence>
struct get_true_types<Predicate, std::tuple<Sequence...>>
{
    using type = impl::filter_false<Predicate, std::tuple<Sequence...>>;
};

template <template <typename> class Predicate, typename... Sequence>
struct get_true_types<Predicate, type_list<Sequence...>>
{
    using type = impl::filter_false<Predicate, std::tuple<Sequence...>>;
};

///
///  generic alias for extracting all types with a specified trait disabled
///
template <template <typename> class Predicate, typename... Sequence>
struct get_false_types
{
    using type = impl::filter_true<Predicate, std::tuple<Sequence...>>;
};

template <template <typename> class Predicate, typename... Sequence>
struct get_false_types<Predicate, std::tuple<Sequence...>>
{
    using type = impl::filter_true<Predicate, std::tuple<Sequence...>>;
};

template <template <typename> class Predicate, typename... Sequence>
struct get_false_types<Predicate, type_list<Sequence...>>
{
    using type = impl::filter_true<Predicate, std::tuple<Sequence...>>;
};

//======================================================================================//
//
//      determines if storage should be implemented
//
//======================================================================================//

/// filter out any types that are not available
template <typename... Types>
using implemented = impl::filter_false<trait::is_available, std::tuple<Types...>>;

template <typename Tuple>
using available_tuple = impl::filter_false<trait::is_available, Tuple>;

template <typename T>
using available_t = impl::filter_false<trait::is_available, T>;

//--------------------------------------------------------------------------------------//

template <typename... T>
using stl_tuple_t = convert_t<available_tuple<concat<T...>>, std::tuple<>>;

template <typename... T>
using type_list_t = convert_t<available_tuple<concat<T...>>, std::tuple<>>;

template <typename... T>
using component_tuple_t = convert_t<available_tuple<concat<T...>>, component_tuple<>>;

template <typename... T>
using component_list_t = convert_t<available_tuple<concat<T...>>, component_list<>>;

template <typename... T>
using auto_tuple_t = convert_t<available_tuple<concat<T...>>, auto_tuple<>>;

template <typename... T>
using auto_list_t = convert_t<available_tuple<concat<T...>>, auto_list<>>;

//--------------------------------------------------------------------------------------//

}  // namespace tim
