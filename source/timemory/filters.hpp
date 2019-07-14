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

#include "timemory/components/type_traits.hpp"
#include "timemory/components/types.hpp"

#include <tuple>

namespace tim
{
namespace details
{
//======================================================================================//
//
//      tuple
//
//======================================================================================//

template <typename...>
struct tuple_concat
{
};

template <>
struct tuple_concat<>
{
    using type = std::tuple<>;
};

template <typename... Ts>
struct tuple_concat<std::tuple<Ts...>>
{
    using type = std::tuple<Ts...>;
};

template <typename... Ts0, typename... Ts1, typename... Rest>
struct tuple_concat<std::tuple<Ts0...>, std::tuple<Ts1...>, Rest...>
: tuple_concat<std::tuple<Ts0..., Ts1...>, Rest...>
{
};

template <typename... Ts>
using tuple_concat_t = typename tuple_concat<Ts...>::type;

//--------------------------------------------------------------------------------------//

template <bool>
struct tuple_filter_if_result
{
    template <typename T>
    using type = std::tuple<T>;
};

template <>
struct tuple_filter_if_result<false>
{
    template <typename T>
    using type = std::tuple<>;
};

template <template <typename> class Predicate, typename Sequence>
struct tuple_filter_if;

template <template <typename> class Predicate, typename... Ts>
struct tuple_filter_if<Predicate, std::tuple<Ts...>>
{
    using type = tuple_concat_t<
        typename tuple_filter_if_result<Predicate<Ts>::value>::template type<Ts>...>;
};

template <template <typename> class Predicate, typename Sequence>
using tuple_type_filter = typename tuple_filter_if<Predicate, Sequence>::type;

//======================================================================================//
//
//      component_list
//
//======================================================================================//

template <typename...>
struct component_list_concat
{
};

template <>
struct component_list_concat<>
{
    using type = component_list<>;
};

template <typename... Ts>
struct component_list_concat<component_list<Ts...>>
{
    using type = component_list<Ts...>;
};

template <typename... Ts0, typename... Ts1, typename... Rest>
struct component_list_concat<component_list<Ts0...>, component_list<Ts1...>, Rest...>
: component_list_concat<component_list<Ts0..., Ts1...>, Rest...>
{
};

template <typename... Ts>
using component_list_concat_t = typename component_list_concat<Ts...>::type;

//--------------------------------------------------------------------------------------//

template <bool>
struct component_list_filter_if_result
{
    template <typename T>
    using type = component_list<T>;
};

template <>
struct component_list_filter_if_result<false>
{
    template <typename T>
    using type = component_list<>;
};

template <template <typename> class Predicate, typename Sequence>
struct component_list_filter_if;

template <template <typename> class Predicate, typename... Ts>
struct component_list_filter_if<Predicate, component_list<Ts...>>
{
    using type = component_list_concat_t<typename component_list_filter_if_result<
        Predicate<Ts>::value>::template type<Ts>...>;
};

template <template <typename> class Predicate, typename Sequence>
using list_type_filter =
    typename details::component_list_filter_if<Predicate, Sequence>::type;

//======================================================================================//
//
//      component_tuple
//
//======================================================================================//

template <typename... _Tp>
struct component_tuple_concat
{
    using type = component_tuple<_Tp...>;
};

template <typename... _Tp>
struct component_tuple_concat<component_tuple<_Tp...>>
{
    using type = component_tuple<_Tp...>;
};

template <typename... _Tp0, typename... _Tp1, typename... Rest>
struct component_tuple_concat<component_tuple<_Tp0...>, component_tuple<_Tp1...>, Rest...>
: component_tuple_concat<component_tuple<_Tp0..., _Tp1...>, Rest...>
{
};

template <typename... _Tp>
using component_tuple_concat_t = typename component_tuple_concat<_Tp...>::type;

//--------------------------------------------------------------------------------------//

template <bool>
struct component_tuple_filter_if_result
{
    template <typename _Tp>
    using type = component_tuple<_Tp>;
};

template <>
struct component_tuple_filter_if_result<false>
{
    template <typename _Tp>
    using type = component_tuple<>;
};

template <template <typename> class Predicate, typename Sequence>
struct component_tuple_filter_if;

template <template <typename> class Predicate, typename... _Tp>
struct component_tuple_filter_if<Predicate, component_tuple<_Tp...>>
{
    using type = component_tuple_concat_t<typename component_tuple_filter_if_result<
        Predicate<_Tp>::value>::template type<_Tp>...>;
};

template <template <typename> class Predicate, typename Sequence>
using component_tuple_type_filter = component_tuple_filter_if<Predicate, Sequence>;

//--------------------------------------------------------------------------------------//
//
// sort types
//
//--------------------------------------------------------------------------------------//
/*
template <bool>
struct tuple_sort_result
{
    template <typename T>
    using type = std::tuple<T>;

    template <typename T, typename U>
    using type = std::tuple<T, U>;
};

template <>
struct tuple_sort_result<false>
{
    template <typename T>
    using type = std::tuple<T>;

    template <typename T, typename U>
    using type = std::tuple<U, T>;
};

template <template <typename> class Predicate, typename Sequence>
struct tuple_sort_if;

template <template <typename> class Predicate, typename T, typename... Ts>
struct tuple_sort_if<Predicate, std::tuple<T, Ts...>>
{
    using type = tuple_concat_t<
        typename tuple_sort_result<Predicate<Ts>::value>::template type<Ts>...>;
};

template <template <typename> class Predicate, typename Sequence>
using tuple_type_sort = typename tuple_sort_if<Predicate, Sequence>::type;
*/

}  // namespace details

//======================================================================================//
//
//      tuple
//
//======================================================================================//

template <typename... Types>
using implemented_tuple =
    details::tuple_type_filter<component::impl_available, std::tuple<Types...>>;

//======================================================================================//
//
//      component_list
//
//======================================================================================//

template <typename... Types>
using implemented_component_list =
    details::list_type_filter<component::impl_available, component_list<Types...>>;

//======================================================================================//
//
//      component_tuple
//
//======================================================================================//

template <typename... Types>
using implemented_component_tuple =
    typename details::component_tuple_type_filter<component::impl_available,
                                                  component_tuple<Types...>>::type;
}
