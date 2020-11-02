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

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

//--------------------------------------------------------------------------------------//

#define TIMEMORY_IMPL_IS_CONCEPT(CONCEPT)                                                \
    struct CONCEPT                                                                       \
    {};                                                                                  \
    template <typename Tp>                                                               \
    struct is_##CONCEPT                                                                  \
    {                                                                                    \
    private:                                                                             \
        template <typename, typename = std::true_type>                                   \
        struct have : std::false_type                                                    \
        {};                                                                              \
        template <typename U>                                                            \
        struct have<U, typename std::is_base_of<typename U::CONCEPT, U>::type>           \
        : std::true_type                                                                 \
        {};                                                                              \
        template <typename U>                                                            \
        struct have<U, typename std::is_base_of<typename U::CONCEPT##_type, U>::type>    \
        : std::true_type                                                                 \
        {};                                                                              \
                                                                                         \
    public:                                                                              \
        using type = typename is_##CONCEPT::template have<                               \
            typename std::remove_cv<Tp>::type>::type;                                    \
        static constexpr bool value =                                                    \
            is_##CONCEPT::template have<typename std::remove_cv<Tp>::type>::value;       \
    };
// constexpr operator bool() const noexcept { return value; }

//--------------------------------------------------------------------------------------//

namespace tim
{
//
using true_type  = std::true_type;
using false_type = std::false_type;
//
struct null_type;
//
namespace trait
{
template <typename Tp>
struct is_available;
}
//
namespace
{
template <typename Tp>
struct anonymous
{
    using type = Tp;
};
//
template <typename Tp>
using anonymous_t = typename anonymous<Tp>::type;
}  // namespace
//
namespace component
{
template <size_t Idx, typename Tag>
struct user_bundle;
//
template <size_t Nt, typename ComponentsT, typename DifferentiatorT = anonymous_t<void>>
struct gotcha;
//
template <typename... Types>
struct placeholder;
//
struct nothing;
//
}  // namespace component
//
namespace concepts
{
//
//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_empty
/// \brief concept that specifies that a variadic type is empty
///
TIMEMORY_IMPL_IS_CONCEPT(empty)

template <template <typename...> class Tuple>
struct is_empty<Tuple<>> : true_type
{};

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_null_type
/// \brief concept that specifies that a type is not a useful type
///
TIMEMORY_IMPL_IS_CONCEPT(null_type)

template <template <typename...> class Tuple>
struct is_null_type<Tuple<>> : true_type
{};

template <>
struct is_null_type<void> : true_type
{};

template <>
struct is_null_type<false_type> : true_type
{};

template <>
struct is_null_type<::tim::null_type> : true_type
{};

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_placeholder
/// \brief concept that specifies that a type is not necessarily marked as not available
/// but is still a dummy type
///
TIMEMORY_IMPL_IS_CONCEPT(placeholder)

template <typename... Types>
struct is_placeholder<component::placeholder<Types...>> : true_type
{};

template <>
struct is_placeholder<component::placeholder<component::nothing>> : true_type
{};

template <>
struct is_placeholder<component::nothing> : true_type
{};

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_api
/// \brief concept that specifies that a type is an API. APIs are used to designate
/// different project implementations, different external library tools, etc.
///
TIMEMORY_IMPL_IS_CONCEPT(api)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_variadic
/// \brief concept that specifies that a type is a generic variadic wrapper
///
TIMEMORY_IMPL_IS_CONCEPT(variadic)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
///
TIMEMORY_IMPL_IS_CONCEPT(wrapper)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_stack_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// and components are stack-allocated
///
TIMEMORY_IMPL_IS_CONCEPT(stack_wrapper)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_heap_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// and components are heap-allocated
///
TIMEMORY_IMPL_IS_CONCEPT(heap_wrapper)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_hybrid_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// and components are stack- and heap- allocated
///
TIMEMORY_IMPL_IS_CONCEPT(hybrid_wrapper)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_mixed_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// and variadic types are mix of stack- and heap- allocated
///
TIMEMORY_IMPL_IS_CONCEPT(mixed_wrapper)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_tagged
/// \brief concept that specifies that a type's template parameters include
/// a API specific tag as one of the template parameters (usually first)
///
TIMEMORY_IMPL_IS_CONCEPT(tagged)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_comp_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// that does not perform auto start/stop, e.g. component_{tuple,list,hybrid}
///
TIMEMORY_IMPL_IS_CONCEPT(comp_wrapper)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_auto_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// that performs auto start/stop, e.g. auto_{tuple,list,hybrid}
///
TIMEMORY_IMPL_IS_CONCEPT(auto_wrapper)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_runtime_configurable
/// \brief concept that specifies that a component type supports configurating the
/// set of components that it collects at runtime (e.g. user_bundle)
///
TIMEMORY_IMPL_IS_CONCEPT(runtime_configurable)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::is_external_function_wrapper
/// \brief concept that specifies that a component type wraps external functions
///
TIMEMORY_IMPL_IS_CONCEPT(external_function_wrapper)

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::has_gotcha
/// \brief determines if a variadic wrapper contains a gotcha component
///
template <typename T>
struct has_gotcha : std::false_type
{};

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::has_user_bundle
/// \brief concept that specifies that a type is a user_bundle type
///
template <typename T>
struct has_user_bundle : false_type
{};

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::bool_int
/// \brief converts a boolean to an integer
///
template <bool B, typename T = int>
struct bool_int
{
    static constexpr T value = (B) ? 1 : 0;
};

//----------------------------------------------------------------------------------//
/// \struct tim::concepts::sum_bool_int
/// \brief counts a series of booleans
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
/// \struct tim::concepts::compatible_wrappers
/// \brief determines whether variadic structures are compatible
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

template <typename T>
struct tuple_type
{
    using type = typename T::tuple_type;
};

template <typename T>
struct auto_type
{
    using type = typename T::auto_type;
};

template <typename T>
struct component_type
{
    using type = typename T::component_type;
};

//----------------------------------------------------------------------------------//

}  // namespace concepts

template <typename T>
using is_empty_t = typename concepts::is_empty<T>::type;

template <typename T>
using is_variadic_t = typename concepts::is_variadic<T>::type;

template <typename T>
using is_wrapper_t = typename concepts::is_wrapper<T>::type;

template <typename T>
using is_stack_wrapper_t = typename concepts::is_stack_wrapper<T>::type;

template <typename T>
using is_heap_wrapper_t = typename concepts::is_heap_wrapper<T>::type;

//----------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#if !defined(TIMEMORY_CONCEPT_ALIAS)
#    define TIMEMORY_CONCEPT_ALIAS(ALIAS, TYPE)                                          \
        namespace tim                                                                    \
        {                                                                                \
        namespace concepts                                                               \
        {                                                                                \
        template <typename T>                                                            \
        using ALIAS = typename TYPE<T>::type;                                            \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

TIMEMORY_CONCEPT_ALIAS(is_empty_t, is_empty)
TIMEMORY_CONCEPT_ALIAS(is_variadic_t, is_variadic)
TIMEMORY_CONCEPT_ALIAS(is_wrapper_t, is_wrapper)
TIMEMORY_CONCEPT_ALIAS(is_stack_wrapper_t, is_stack_wrapper)
TIMEMORY_CONCEPT_ALIAS(is_heap_wrapper_t, is_heap_wrapper)

TIMEMORY_CONCEPT_ALIAS(tuple_type_t, tuple_type)
TIMEMORY_CONCEPT_ALIAS(auto_type_t, auto_type)
TIMEMORY_CONCEPT_ALIAS(component_type_t, component_type)

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_CONCRETE_CONCEPT(CONCEPT, SPECIALIZED_TYPE, VALUE)               \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <>                                                                          \
    struct CONCEPT<SPECIALIZED_TYPE> : VALUE                                             \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_TEMPLATE_CONCEPT(CONCEPT, SPECIALIZED_TYPE, VALUE, TYPE)         \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE T>                                                                    \
    struct CONCEPT<SPECIALIZED_TYPE<T>> : VALUE                                          \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_VARIADIC_CONCEPT(CONCEPT, SPECIALIZED_TYPE, VALUE, TYPE)         \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE... T>                                                                 \
    struct CONCEPT<SPECIALIZED_TYPE<T...>> : VALUE                                       \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_CONCRETE_CONCEPT_TYPE(CONCEPT, SPECIALIZED_TYPE, ...)            \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <>                                                                          \
    struct CONCEPT<SPECIALIZED_TYPE>                                                     \
    {                                                                                    \
        using type = __VA_ARGS__;                                                        \
    };                                                                                   \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_TEMPLATE_CONCEPT_TYPE(CONCEPT, SPECIALIZED_TYPE, TYPE, ...)      \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE T>                                                                    \
    struct CONCEPT<SPECIALIZED_TYPE<T>>                                                  \
    {                                                                                    \
        using type = __VA_ARGS__;                                                        \
    };                                                                                   \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_VARIADIC_CONCEPT_TYPE(CONCEPT, SPECIALIZED_TYPE, TYPE, ...)      \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE... T>                                                                 \
    struct CONCEPT<SPECIALIZED_TYPE<T...>>                                               \
    {                                                                                    \
        using type = __VA_ARGS__;                                                        \
    };                                                                                   \
    }                                                                                    \
    }
