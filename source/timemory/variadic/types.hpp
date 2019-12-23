//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
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
 * This is a pre-declaration of all the variadic wrappers.
 * Care should be taken to make sure that this includes a minimal
 * number of additional headers. Also provides concat of types
 *
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

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

template <typename _Tuple, typename _List>
class component_hybrid;

template <typename... Types>
class auto_tuple;

template <typename... Types>
class auto_list;

template <typename _Tuple, typename _List>
class auto_hybrid;

//--------------------------------------------------------------------------------------//

namespace impl
{
template <typename... _Types>
struct concat
{
    using type = std::tuple<_Types...>;
};

template <typename... _Types>
struct concat<std::tuple<_Types...>>
{
    using type = std::tuple<_Types...>;
};

template <typename... _Types>
struct concat<component_tuple<_Types...>>
{
    using type = typename concat<_Types...>::type;
};

template <typename... _Types>
struct concat<component_list<_Types...>>
{
    using type = typename concat<_Types...>::type;
};

template <typename... _Types>
struct concat<auto_tuple<_Types...>>
{
    using type = typename concat<_Types...>::type;
};

template <typename... _Types>
struct concat<auto_list<_Types...>>
{
    using type = typename concat<_Types...>::type;
};

template <typename... _Lhs, typename... _Rhs>
struct concat<std::tuple<_Lhs...>, std::tuple<_Rhs...>>
{
    using type = typename concat<_Lhs..., _Rhs...>::type;
};

//--------------------------------------------------------------------------------------//
//      component_hybrid
//--------------------------------------------------------------------------------------//

template <typename... _TupTypes, typename... _LstTypes>
struct concat<component_hybrid<std::tuple<_TupTypes...>, std::tuple<_LstTypes...>>>
{
    using tuple_type = component_tuple<_TupTypes...>;
    using list_type  = component_list<_LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

template <typename... _TupTypes, typename... _LstTypes>
struct concat<component_hybrid<component_tuple<_TupTypes...>, std::tuple<_LstTypes...>>>
{
    using tuple_type = component_tuple<_TupTypes...>;
    using list_type  = component_list<_LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

template <typename... _TupTypes, typename... _LstTypes>
struct concat<component_hybrid<std::tuple<_TupTypes...>, component_list<_LstTypes...>>>
{
    using tuple_type = component_tuple<_TupTypes...>;
    using list_type  = component_list<_LstTypes...>;
    using type       = component_hybrid<tuple_type, list_type>;
};

//--------------------------------------------------------------------------------------//
//      auto_hybrid
//--------------------------------------------------------------------------------------//

template <typename... _TupTypes, typename... _LstTypes>
struct concat<auto_hybrid<std::tuple<_TupTypes...>, std::tuple<_LstTypes...>>>
{
    using tuple_type = component_tuple<_TupTypes...>;
    using list_type  = component_list<_LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

template <typename... _TupTypes, typename... _LstTypes>
struct concat<auto_hybrid<component_tuple<_TupTypes...>, std::tuple<_LstTypes...>>>
{
    using tuple_type = component_tuple<_TupTypes...>;
    using list_type  = component_list<_LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

template <typename... _TupTypes, typename... _LstTypes>
struct concat<auto_hybrid<std::tuple<_TupTypes...>, component_list<_LstTypes...>>>
{
    using tuple_type = component_tuple<_TupTypes...>;
    using list_type  = component_list<_LstTypes...>;
    using type       = auto_hybrid<tuple_type, list_type>;
};

//--------------------------------------------------------------------------------------//
//
//      Combine
//
//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename... _Rhs>
struct concat<std::tuple<_Lhs...>, _Rhs...>
{
    using type = typename concat<typename concat<_Lhs...>::type,
                                 typename concat<_Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      component_tuple
//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename... _Rhs>
struct concat<component_tuple<_Lhs...>, _Rhs...>
{
    using type = typename concat<typename concat<_Lhs...>::type,
                                 typename concat<_Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      component_list
//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename... _Rhs>
struct concat<component_list<_Lhs...>, _Rhs...>
{
    using type = typename concat<typename concat<_Lhs...>::type,
                                 typename concat<_Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      auto_tuple
//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename... _Rhs>
struct concat<auto_tuple<_Lhs...>, _Rhs...>
{
    using type = typename concat<typename concat<_Lhs...>::type,
                                 typename concat<_Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      auto_list
//--------------------------------------------------------------------------------------//

template <typename... _Lhs, typename... _Rhs>
struct concat<auto_list<_Lhs...>, _Rhs...>
{
    using type = typename concat<typename concat<_Lhs...>::type,
                                 typename concat<_Rhs...>::type>::type;
};

//--------------------------------------------------------------------------------------//
//      component_hybrid
//--------------------------------------------------------------------------------------//

template <typename _Tup, typename _Lst, typename... _Rhs, typename... _Tail>
struct concat<component_hybrid<_Tup, _Lst>, component_tuple<_Rhs...>, _Tail...>
{
    using type =
        typename concat<component_hybrid<typename concat<_Tup, _Rhs...>::type, _Lst>,
                        _Tail...>::type;
    using tuple_type = typename type::tuple_type;
    using list_type  = typename type::list_type;
};

template <typename _Tup, typename _Lst, typename... _Rhs, typename... _Tail>
struct concat<component_hybrid<_Tup, _Lst>, component_list<_Rhs...>, _Tail...>
{
    using type =
        typename concat<component_hybrid<_Tup, typename concat<_Lst, _Rhs...>::type>,
                        _Tail...>::type;
    using tuple_type = typename type::tuple_type;
    using list_type  = typename type::list_type;
};

//--------------------------------------------------------------------------------------//
//      auto_hybrid
//--------------------------------------------------------------------------------------//

template <typename _Tup, typename _Lst, typename... _Rhs, typename... _Tail>
struct concat<auto_hybrid<_Tup, _Lst>, component_tuple<_Rhs...>, _Tail...>
{
    using type = typename concat<auto_hybrid<typename concat<_Tup, _Rhs...>::type, _Lst>,
                                 _Tail...>::type;
    using tuple_type = typename type::tuple_type;
    using list_type  = typename type::list_type;
};

template <typename _Tup, typename _Lst, typename... _Rhs, typename... _Tail>
struct concat<auto_hybrid<_Tup, _Lst>, component_list<_Rhs...>, _Tail...>
{
    using type = typename concat<auto_hybrid<_Tup, typename concat<_Lst, _Rhs...>::type>,
                                 _Tail...>::type;
    using tuple_type = typename type::tuple_type;
    using list_type  = typename type::list_type;
};

}  // namespace impl

template <typename... _Types>
using concat = typename impl::concat<_Types...>::type;

}  // namespace tim

//======================================================================================//
