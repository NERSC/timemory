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

#include <tuple>
#include <type_traits>

namespace tim
{
namespace concepts
{
using false_type = std::false_type;
using true_type  = std::true_type;

//----------------------------------------------------------------------------------//
/// concepts the specifies that a variadic type is empty
///
template <typename T>
struct is_empty : false_type
{};

template <template <typename...> class Tuple>
struct is_empty<Tuple<>> : true_type
{};

//----------------------------------------------------------------------------------//
/// concepts the specifies that a type is a generic variadic wrapper
///
template <typename T>
struct is_variadic : false_type
{};

//----------------------------------------------------------------------------------//
/// concepts the specifies that a type is a timemory variadic wrapper
///
template <typename T>
struct is_wrapper : false_type
{};

//----------------------------------------------------------------------------------//
/// concepts the specifies that a type is a timemory variadic wrapper
/// and components are stack-allocated
///
template <typename T>
struct is_stack_wrapper : false_type
{};

//----------------------------------------------------------------------------------//
/// concepts the specifies that a type is a timemory variadic wrapper
/// and components are heap-allocated
///
template <typename T>
struct is_heap_wrapper : false_type
{};

//----------------------------------------------------------------------------------//
/// concepts the specifies that a type is a timemory variadic wrapper
/// and components are stack- and heap- allocated
///
template <typename T>
struct is_hybrid_wrapper : false_type
{};

//----------------------------------------------------------------------------------//
/// concepts the specifies that a type is a timemory variadic wrapper
/// that does not perform auto start/stop, e.g. component_{tuple,list,hybrid}
///
template <typename T>
struct is_comp_wrapper : false_type
{};

//----------------------------------------------------------------------------------//
/// concepts the specifies that a type is a timemory variadic wrapper
/// that performs auto start/stop, e.g. auto_{tuple,list,hybrid}
///
template <typename T>
struct is_auto_wrapper : false_type
{};

//----------------------------------------------------------------------------------//
/// concepts the specifies that a type is a gotcha type
///
template <typename T>
struct is_gotcha : false_type
{};

//----------------------------------------------------------------------------------//
/// concepts the specifies that a type is a user_bundle type
///
template <typename T>
struct is_user_bundle : false_type
{};

//----------------------------------------------------------------------------------//
/// converts a boolean to an integer
///
template <bool B, typename T = int>
struct bool_int
{
    static constexpr T value = (B) ? 1 : 0;
};

//----------------------------------------------------------------------------------//
/// counts a series of booleans
///
template <bool... B>
struct sum_bool_int;

template <>
struct sum_bool_int<>
{
    using value_type                  = int;
    static constexpr value_type value = 0;
};

template <bool B>
struct sum_bool_int<B>
{
    using value_type                  = int;
    static constexpr value_type value = bool_int<B>::value;
};

template <bool B, bool... BoolTail>
struct sum_bool_int<B, BoolTail...>
{
    using value_type = int;
    static constexpr value_type value =
        bool_int<B>::value + sum_bool_int<BoolTail...>::value;
};

//----------------------------------------------------------------------------------//
/// determines whether variadic structures are compatible
///
template <typename Lhs, typename Rhs>
struct compatible_wrappers
{
    static constexpr int variadic_count = (bool_int<is_variadic<Lhs>::value>::value +
                                           bool_int<is_variadic<Rhs>::value>::value);
    static constexpr int wrapper_count  = (bool_int<is_wrapper<Lhs>::value>::value +
                                          bool_int<is_wrapper<Rhs>::value>::value);
    static constexpr int heap_count     = (bool_int<is_heap_wrapper<Lhs>::value>::value +
                                       bool_int<is_heap_wrapper<Rhs>::value>::value);
    static constexpr int stack_count    = (bool_int<is_stack_wrapper<Lhs>::value>::value +
                                        bool_int<is_stack_wrapper<Rhs>::value>::value);
    static constexpr int hybrid_count = (bool_int<is_hybrid_wrapper<Lhs>::value>::value +
                                         bool_int<is_hybrid_wrapper<Rhs>::value>::value);

    //  valid configs:
    //
    //      1. both heap/stack/hybrid
    //      2. hybrid + stack or heap wrapper
    //      3. wrapper + variadic
    //
    //  invalid configs:
    //
    //      1. hybrid + non-wrapper variadic
    //      2. one stack + one heap
    //      3. zero variadic
    //      4. variadic and zero wrappers
    //

    static constexpr bool valid_1 =
        (hybrid_count == 2 || stack_count == 2 || heap_count == 2);
    static constexpr bool valid_2 =
        (hybrid_count == 1 && (stack_count + heap_count) == 1);
    static constexpr bool valid_3 = (wrapper_count == 1 && variadic_count == 2);

    static constexpr bool invalid_1 = (hybrid_count == 1 && wrapper_count == 1);
    static constexpr bool invalid_2 =
        (hybrid_count == 1 && (stack_count + heap_count) == 1);
    static constexpr bool invalid_3 = (variadic_count == 0);
    static constexpr bool invalid_4 = (variadic_count == 2 && wrapper_count == 0);

    using value_type = bool;

    static constexpr bool value = (!invalid_1 && !invalid_2 && !invalid_3 && !invalid_4 &&
                                   (valid_1 || valid_2 || valid_3))
                                      ? true
                                      : false;

    using type = std::conditional_t<(value), true_type, false_type>;
};

//----------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
struct is_acceptable_conversion
{
    static constexpr bool value =
        (std::is_same<Lhs, Rhs>::value ||
         (std::is_integral<Lhs>::value && std::is_integral<Rhs>::value) ||
         (std::is_floating_point<Lhs>::value && std::is_floating_point<Rhs>::value));
};

//----------------------------------------------------------------------------------//

}  // namespace concepts

template <typename T>
using is_empty_t = typename concepts ::is_empty<T>::type;

template <typename T>
using is_variadic_t = typename concepts ::is_variadic<T>::type;

template <typename T>
using is_wrapper_t = typename concepts ::is_wrapper<T>::type;

template <typename T>
using is_stack_wrapper_t = typename concepts ::is_stack_wrapper<T>::type;

template <typename T>
using is_heap_wrapper_t = typename concepts ::is_heap_wrapper<T>::type;

//----------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#define TIMEMORY_DEFINE_CONCRETE_CONCEPT(CONCEPT, COMPONENT, VALUE)                      \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <>                                                                          \
    struct CONCEPT<COMPONENT> : VALUE                                                    \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_TEMPLATE_CONCEPT(CONCEPT, COMPONENT, VALUE, TYPE)                \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE T>                                                                    \
    struct CONCEPT<COMPONENT<T>> : VALUE                                                 \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_VARIADIC_CONCEPT(CONCEPT, COMPONENT, VALUE, TYPE)                \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE... T>                                                                 \
    struct CONCEPT<COMPONENT<T...>> : VALUE                                              \
    {};                                                                                  \
    }                                                                                    \
    }
