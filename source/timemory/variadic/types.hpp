//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file timemory/variadic/types.hpp
 * \headerfile timemory/variadic/types.hpp "timemory/variadic/types.hpp"
 *
 * This is a declaration of all the variadic wrappers.
 * Care should be taken to make sure that this includes a minimal
 * number of additional headers. Also provides concat of types
 *
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"

//======================================================================================//
//
namespace tim
{
//--------------------------------------------------------------------------------------//
//
//  Forward declaration of variadic wrapper types
//
//--------------------------------------------------------------------------------------//

template <typename... Types>
class component_tuple;

template <typename... Types>
class component_list;

template <typename Tuple, typename _List>
class component_hybrid;

template <typename... Types>
class auto_tuple;

template <typename... Types>
class auto_list;

template <typename Tuple, typename _List>
class auto_hybrid;

}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                                  IS VARIADIC / IS WRAPPER
//
//--------------------------------------------------------------------------------------//

// these are variadic types used to bundle components together
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, auto_tuple, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, auto_list, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, auto_hybrid, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, component_tuple, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, component_list, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, component_hybrid, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, std::tuple, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_variadic, type_list, true_type, typename)

// there are timemory-specific variadic wrappers
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_wrapper, auto_tuple, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_wrapper, auto_list, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_wrapper, auto_hybrid, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_wrapper, component_tuple, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_wrapper, component_list, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_wrapper, component_hybrid, true_type, typename)

// tuple wrappers (stack-allocated components)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_stack_wrapper, auto_tuple, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_stack_wrapper, component_tuple, true_type, typename)

// list wrappers (heap-allocated components)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_heap_wrapper, auto_list, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_heap_wrapper, component_list, true_type, typename)

// hybrid wrappers (stack- and heap- allocated components)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_hybrid_wrapper, auto_hybrid, true_type, typename)
TIMEMORY_DEFINE_VARIADIC_CONCEPT(is_hybrid_wrapper, component_hybrid, true_type, typename)

//======================================================================================//

namespace tim
{
namespace impl
{
template <typename... Types>
struct concat
{
    using type = std::tuple<Types...>;
};

template <typename... Types>
struct concat<std::tuple<Types...>>
{
    using type = std::tuple<Types...>;
};

template <typename... Types>
struct concat<component_tuple<Types...>>
{
    using type = typename concat<Types...>::type;
};

template <typename... Types>
struct concat<component_list<Types...>>
{
    using type = typename concat<Types...>::type;
};

template <typename... Types>
struct concat<component_list<Types*...>> : concat<component_list<Types...>>
{};

template <typename... Types>
struct concat<auto_tuple<Types...>>
{
    using type = typename concat<Types...>::type;
};

template <typename... Types>
struct concat<auto_list<Types...>>
{
    using type = typename concat<Types...>::type;
};

template <typename... Types>
struct concat<auto_list<Types*...>> : concat<auto_list<Types...>>
{};

template <typename... Lhs, typename... Rhs>
struct concat<std::tuple<Lhs...>, std::tuple<Rhs...>>
{
    using type = typename concat<Lhs..., Rhs...>::type;
};

//--------------------------------------------------------------------------------------//
//      component_hybrid
//--------------------------------------------------------------------------------------//

template <typename... TupTypes, typename... LstTypes>
struct concat<component_hybrid<std::tuple<TupTypes...>, std::tuple<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<component_hybrid<component_tuple<TupTypes...>, std::tuple<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<component_hybrid<std::tuple<TupTypes...>, component_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

//--------------------------------------------------------------------------------------//
//      auto_hybrid
//--------------------------------------------------------------------------------------//

template <typename... TupTypes, typename... LstTypes>
struct concat<auto_hybrid<std::tuple<TupTypes...>, std::tuple<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<auto_hybrid<component_tuple<TupTypes...>, std::tuple<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

template <typename... TupTypes, typename... LstTypes>
struct concat<auto_hybrid<std::tuple<TupTypes...>, component_list<LstTypes...>>>
{
    using tuple_type = component_tuple<TupTypes...>;
    using list_type  = component_list<LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

//--------------------------------------------------------------------------------------//
//
//      Combine
//
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<std::tuple<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      component_tuple
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<component_tuple<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      component_list
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<component_list<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

template <typename... Lhs, typename... Rhs>
struct concat<component_list<Lhs...>*, Rhs*...>
: public concat<component_list<Lhs...>, Rhs...>
{};

template <typename... Lhs, typename... Rhs>
struct concat<component_list<Lhs...>*, Rhs...>
: public concat<component_list<Lhs...>, Rhs...>
{};

//--------------------------------------------------------------------------------------//
//      auto_tuple
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<auto_tuple<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      auto_list
//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename... Rhs>
struct concat<auto_list<Lhs...>, Rhs...>
{
    using type = typename concat<typename concat<Lhs...>::type,
                                 typename concat<Rhs...>::type>::type;
};

template <typename... Lhs, typename... Rhs>
struct concat<auto_list<Lhs...>*, Rhs*...> : public concat<auto_list<Lhs...>, Rhs...>
{};

template <typename... Lhs, typename... Rhs>
struct concat<auto_list<Lhs...>*, Rhs...> : public concat<auto_list<Lhs...>, Rhs...>
{};

//--------------------------------------------------------------------------------------//
//      component_hybrid
//--------------------------------------------------------------------------------------//

template <typename Tup, typename Lst, typename... Rhs, typename... Tail>
struct concat<component_hybrid<Tup, Lst>, component_tuple<Rhs...>, Tail...>
{
    using type =
        typename concat<component_hybrid<typename concat<Tup, Rhs...>::type, Lst>,
                        Tail...>::type;
    using tuple_type = typename type::tuple_type;
    using list_type  = typename type::list_type;
};

template <typename Tup, typename Lst, typename... Rhs, typename... Tail>
struct concat<component_hybrid<Tup, Lst>, component_list<Rhs...>, Tail...>
{
    using type =
        typename concat<component_hybrid<Tup, typename concat<Lst, Rhs...>::type>,
                        Tail...>::type;
    using tuple_type = typename type::tuple_type;
    using list_type  = typename type::list_type;
};

//--------------------------------------------------------------------------------------//
//      auto_hybrid
//--------------------------------------------------------------------------------------//

template <typename Tup, typename Lst, typename... Rhs, typename... Tail>
struct concat<auto_hybrid<Tup, Lst>, component_tuple<Rhs...>, Tail...>
{
    using type = typename concat<auto_hybrid<typename concat<Tup, Rhs...>::type, Lst>,
                                 Tail...>::type;
    using tuple_type = typename type::tuple_type;
    using list_type  = typename type::list_type;
};

template <typename Tup, typename Lst, typename... Rhs, typename... Tail>
struct concat<auto_hybrid<Tup, Lst>, component_list<Rhs...>, Tail...>
{
    using type = typename concat<auto_hybrid<Tup, typename concat<Lst, Rhs...>::type>,
                                 Tail...>::type;
    using tuple_type = typename type::tuple_type;
    using list_type  = typename type::list_type;
};

}  // namespace impl

template <typename... Types>
using concat = typename impl::concat<Types...>::type;

}  // namespace tim

//======================================================================================//
