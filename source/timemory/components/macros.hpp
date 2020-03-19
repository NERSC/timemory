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

/**
 * \file timemory/components/macros.hpp
 * Define common macros for the components
 */

#pragma once

#include "timemory/components/properties.hpp"
#include "timemory/mpl/types.hpp"

#include <string>
#include <unordered_set>

//======================================================================================//

/**
 * \macro TIMEMORY_DECLARE_COMPONENT
 * \brief Declare a non-templated component type in the tim::component namespace
 */

#if !defined(TIMEMORY_DECLARE_COMPONENT)
#    define TIMEMORY_DECLARE_COMPONENT(NAME)                                             \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        struct NAME;                                                                     \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_BUNDLE_INDEX
 * \brief Declare a bundle index
 */

#if !defined(TIMEMORY_BUNDLE_INDEX)
#    define TIMEMORY_BUNDLE_INDEX(NAME, IDX)                                             \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        static constexpr size_t NAME = IDX;                                              \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_DECLARE_TEMPLATE_COMPONENT
 * \brief Declare a templated component type in the tim::component namespace
 */

#if !defined(TIMEMORY_DECLARE_TEMPLATE_COMPONENT)
#    define TIMEMORY_DECLARE_TEMPLATE_COMPONENT(NAME, ...)                               \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        template <__VA_ARGS__>                                                           \
        struct NAME;                                                                     \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
/**
 * \macro TIMEMORY_COMPONENT_ALIAS
 * \brief Declare a non-templated alias to a component in the tim::component namespace
 */

#if !defined(TIMEMORY_COMPONENT_ALIAS)
#    define TIMEMORY_COMPONENT_ALIAS(NAME, ...)                                          \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        using NAME = __VA_ARGS__;                                                        \
        }                                                                                \
        }
#endif

//======================================================================================//

/**
 * \macro TIMEMORY_PROPERTY_SPECIALIZATION
 * \brief Specialization of the property specialization
 */

#if !defined(TIMEMORY_PROPERTY_SPECIALIZATION)
#    define TIMEMORY_PROPERTY_SPECIALIZATION(TYPE, ENUM, ID, ...)                        \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        template <>                                                                      \
        struct properties<TYPE> : static_properties<TYPE>                                \
        {                                                                                \
            using type                                = TYPE;                            \
            using value_type                          = TIMEMORY_COMPONENT;              \
            static constexpr TIMEMORY_COMPONENT value = ENUM;                            \
            static constexpr const char*        enum_string() { return #ENUM; }          \
            static constexpr const char*        id() { return ID; }                      \
            static const idset_t&               ids()                                    \
            {                                                                            \
                static idset_t _instance{ ID, __VA_ARGS__ };                             \
                return _instance;                                                        \
            }                                                                            \
        };                                                                               \
        template <>                                                                      \
        struct enumerator<ENUM> : properties<TYPE>                                       \
        {                                                                                \
            using type = TYPE;                                                           \
        };                                                                               \
        }                                                                                \
        }
#endif

//======================================================================================//

/**
 * \macro TIMEMORY_TOOLSET_ALIAS
 * \brief Creates an alias for a complex type when declaring the statistics type
 */

#if !defined(TIMEMORY_TOOLSET_ALIAS)
#    define TIMEMORY_TOOLSET_ALIAS(NAME, WRAPPER, ...)                                   \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        namespace aliases                                                                \
        {                                                                                \
        using NAME = WRAPPER<__VA_ARGS__>;                                               \
        }                                                                                \
        }                                                                                \
        }                                                                                \
        using tim::component::aliases::NAME;
#endif

//======================================================================================//
//
//      GENERIC TYPE-TRAIT SPECIALIZATION (for true_type/false_type traits)
//
//======================================================================================//

#if !defined(TIMEMORY_DEFINE_CONCRETE_TRAIT)
#    define TIMEMORY_DEFINE_CONCRETE_TRAIT(TRAIT, COMPONENT, VALUE)                      \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <>                                                                      \
        struct TRAIT<COMPONENT> : VALUE                                                  \
        {};                                                                              \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_DEFINE_TEMPLATE_TRAIT)
#    define TIMEMORY_DEFINE_TEMPLATE_TRAIT(TRAIT, COMPONENT, VALUE, TYPE)                \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TYPE T>                                                                \
        struct TRAIT<COMPONENT<T>> : VALUE                                               \
        {};                                                                              \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_DEFINE_VARIADIC_TRAIT)
#    define TIMEMORY_DEFINE_VARIADIC_TRAIT(TRAIT, COMPONENT, VALUE, TYPE)                \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TYPE... T>                                                             \
        struct TRAIT<COMPONENT<T...>> : VALUE                                            \
        {};                                                                              \
        }                                                                                \
        }
#endif

//======================================================================================//
//
//      STATISTICS TYPE-TRAIT SPECIALIZATION
//
//======================================================================================//

#if !defined(TIMEMORY_STATISTICS_TYPE)
#    define TIMEMORY_STATISTICS_TYPE(COMPONENT, TYPE)                                    \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <>                                                                      \
        struct statistics<COMPONENT>                                                     \
        {                                                                                \
            using type = TYPE;                                                           \
        };                                                                               \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_TEMPLATE_STATISTICS_TYPE)
#    define TIMEMORY_TEMPLATE_STATISTICS_TYPE(COMPONENT, TYPE, TEMPLATE_TYPE)            \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TEMPLATE_TYPE T>                                                       \
        struct statistics<COMPONENT<T>>                                                  \
        {                                                                                \
            using type = TYPE;                                                           \
        };                                                                               \
        }                                                                                \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_VARIADIC_STATISTICS_TYPE)
#    define TIMEMORY_VARIADIC_STATISTICS_TYPE(COMPONENT, TYPE, TEMPLATE_TYPE)            \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TEMPLATE_TYPE... T>                                                    \
        struct statistics<COMPONENT<T...>>                                               \
        {                                                                                \
            using type = TYPE;                                                           \
        };                                                                               \
        }                                                                                \
        }
#endif

//======================================================================================//
//
//      EXTERN TEMPLATE DECLARE AND INSTANTIATE
//
//======================================================================================//

#if !defined(_EXTERN_NAME_COMBINE)
#    define _EXTERN_NAME_COMBINE(X, Y) X##Y
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(_EXTERN_TUPLE_ALIAS)
#    define _EXTERN_TUPLE_ALIAS(Y) _EXTERN_NAME_COMBINE(extern_tuple_, Y)
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(_EXTERN_LIST_ALIAS)
#    define _EXTERN_LIST_ALIAS(Y) _EXTERN_NAME_COMBINE(extern_list_, Y)
#endif

//--------------------------------------------------------------------------------------//

#if !defined(_WINDOWS)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_DECLARE_EXTERN_STORAGE)
#        define TIMEMORY_DECLARE_EXTERN_STORAGE(TYPE, ...)                               \
            namespace tim                                                                \
            {                                                                            \
            extern template class impl::storage<TYPE, implements_storage<TYPE>::value>;  \
            extern template class storage<TYPE>;                                         \
            extern template class singleton<                                             \
                impl::storage<TYPE, implements_storage<TYPE>::value>,                    \
                std::unique_ptr<impl::storage<TYPE, implements_storage<TYPE>::value>,    \
                                impl::storage_deleter<impl::storage<                     \
                                    TYPE, implements_storage<TYPE>::value>>>>;           \
            extern template storage_singleton<storage<TYPE>>*                            \
                                                get_storage_singleton<storage<TYPE>>();  \
            extern template storage_initializer storage_initializer::get<TYPE>();        \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_INSTANTIATE_EXTERN_STORAGE)
#        define TIMEMORY_INSTANTIATE_EXTERN_STORAGE(TYPE, VAR)                           \
            namespace tim                                                                \
            {                                                                            \
            template class impl::storage<TYPE, implements_storage<TYPE>::value>;         \
            template class storage<TYPE>;                                                \
            template class singleton<                                                    \
                impl::storage<TYPE, implements_storage<TYPE>::value>,                    \
                std::unique_ptr<impl::storage<TYPE, implements_storage<TYPE>::value>,    \
                                impl::storage_deleter<impl::storage<                     \
                                    TYPE, implements_storage<TYPE>::value>>>>;           \
            template storage_singleton<storage<TYPE>>*                                   \
                                         get_storage_singleton<storage<TYPE>>();         \
            template storage_initializer storage_initializer::get<TYPE>();               \
            }                                                                            \
            namespace                                                                    \
            {                                                                            \
            using namespace tim::component;                                              \
            namespace component = tim::component;                                        \
            tim::storage_initializer storage_initializer__##VAR =                        \
                tim::storage_initializer::get<TYPE>();                                   \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#else  // elif defined(WINDOWS)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_DECLARE_EXTERN_STORAGE)
#        define TIMEMORY_DECLARE_EXTERN_STORAGE(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_INSTANTIATE_EXTERN_STORAGE)
#        define TIMEMORY_INSTANTIATE_EXTERN_STORAGE(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#endif

//======================================================================================//
//
//      EXTERN OPERATION DECLARE AND INSTANTIATE
//
//======================================================================================//

#if !defined(_WINDOWS)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_DECLARE_EXTERN_OPERATIONS)
#        define TIMEMORY_DECLARE_EXTERN_OPERATIONS(COMPONENT_NAME, HAS_DATA)             \
            namespace tim                                                                \
            {                                                                            \
            namespace operation                                                          \
            {                                                                            \
            extern template struct init_storage<COMPONENT_NAME>;                         \
            extern template struct construct<COMPONENT_NAME>;                            \
            extern template struct set_prefix<COMPONENT_NAME>;                           \
            extern template struct set_flat_profile<COMPONENT_NAME>;                     \
            extern template struct set_timeline_profile<COMPONENT_NAME>;                 \
            extern template struct insert_node<COMPONENT_NAME>;                          \
            extern template struct pop_node<COMPONENT_NAME>;                             \
            extern template struct record<COMPONENT_NAME>;                               \
            extern template struct reset<COMPONENT_NAME>;                                \
            extern template struct measure<COMPONENT_NAME>;                              \
            extern template struct sample<COMPONENT_NAME>;                               \
            extern template struct start<COMPONENT_NAME>;                                \
            extern template struct priority_start<COMPONENT_NAME>;                       \
            extern template struct standard_start<COMPONENT_NAME>;                       \
            extern template struct delayed_start<COMPONENT_NAME>;                        \
            extern template struct stop<COMPONENT_NAME>;                                 \
            extern template struct priority_stop<COMPONENT_NAME>;                        \
            extern template struct standard_stop<COMPONENT_NAME>;                        \
            extern template struct delayed_stop<COMPONENT_NAME>;                         \
            extern template struct mark_begin<COMPONENT_NAME>;                           \
            extern template struct mark_end<COMPONENT_NAME>;                             \
            extern template struct store<COMPONENT_NAME>;                                \
            extern template struct audit<COMPONENT_NAME>;                                \
            extern template struct plus<COMPONENT_NAME>;                                 \
            extern template struct minus<COMPONENT_NAME>;                                \
            extern template struct multiply<COMPONENT_NAME>;                             \
            extern template struct divide<COMPONENT_NAME>;                               \
            extern template struct get<COMPONENT_NAME>;                                  \
            extern template struct base_printer<COMPONENT_NAME>;                         \
            extern template struct print<COMPONENT_NAME>;                                \
            extern template struct print_header<COMPONENT_NAME>;                         \
            extern template struct print_statistics<COMPONENT_NAME>;                     \
            extern template struct print_storage<COMPONENT_NAME>;                        \
            extern template struct add_secondary<COMPONENT_NAME>;                        \
            extern template struct add_statistics<COMPONENT_NAME>;                       \
            extern template struct serialization<COMPONENT_NAME>;                        \
            extern template struct echo_measurement<                                     \
                COMPONENT_NAME, trait::echo_enabled<COMPONENT_NAME>::value>;             \
            extern template struct copy<COMPONENT_NAME>;                                 \
            extern template struct assemble<COMPONENT_NAME>;                             \
            extern template struct dismantle<COMPONENT_NAME>;                            \
            extern template struct finalize::get<COMPONENT_NAME, HAS_DATA>;              \
            extern template struct finalize::mpi_get<COMPONENT_NAME, HAS_DATA>;          \
            extern template struct finalize::upc_get<COMPONENT_NAME, HAS_DATA>;          \
            extern template struct finalize::dmp_get<COMPONENT_NAME, HAS_DATA>;          \
            extern template struct finalize::print<COMPONENT_NAME, HAS_DATA>;            \
            }                                                                            \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS)
#        define TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(COMPONENT_NAME, HAS_DATA)         \
            namespace tim                                                                \
            {                                                                            \
            namespace operation                                                          \
            {                                                                            \
            template struct init_storage<COMPONENT_NAME>;                                \
            template struct construct<COMPONENT_NAME>;                                   \
            template struct set_prefix<COMPONENT_NAME>;                                  \
            template struct set_flat_profile<COMPONENT_NAME>;                            \
            template struct set_timeline_profile<COMPONENT_NAME>;                        \
            template struct insert_node<COMPONENT_NAME>;                                 \
            template struct pop_node<COMPONENT_NAME>;                                    \
            template struct record<COMPONENT_NAME>;                                      \
            template struct reset<COMPONENT_NAME>;                                       \
            template struct measure<COMPONENT_NAME>;                                     \
            template struct sample<COMPONENT_NAME>;                                      \
            template struct start<COMPONENT_NAME>;                                       \
            template struct priority_start<COMPONENT_NAME>;                              \
            template struct standard_start<COMPONENT_NAME>;                              \
            template struct delayed_start<COMPONENT_NAME>;                               \
            template struct stop<COMPONENT_NAME>;                                        \
            template struct priority_stop<COMPONENT_NAME>;                               \
            template struct standard_stop<COMPONENT_NAME>;                               \
            template struct delayed_stop<COMPONENT_NAME>;                                \
            template struct mark_begin<COMPONENT_NAME>;                                  \
            template struct mark_end<COMPONENT_NAME>;                                    \
            template struct store<COMPONENT_NAME>;                                       \
            template struct audit<COMPONENT_NAME>;                                       \
            template struct plus<COMPONENT_NAME>;                                        \
            template struct minus<COMPONENT_NAME>;                                       \
            template struct multiply<COMPONENT_NAME>;                                    \
            template struct divide<COMPONENT_NAME>;                                      \
            template struct get<COMPONENT_NAME>;                                         \
            template struct base_printer<COMPONENT_NAME>;                                \
            template struct print<COMPONENT_NAME>;                                       \
            template struct print_header<COMPONENT_NAME>;                                \
            template struct print_statistics<COMPONENT_NAME>;                            \
            template struct print_storage<COMPONENT_NAME>;                               \
            template struct add_secondary<COMPONENT_NAME>;                               \
            template struct add_statistics<COMPONENT_NAME>;                              \
            template struct serialization<COMPONENT_NAME>;                               \
            template struct echo_measurement<                                            \
                COMPONENT_NAME, trait::echo_enabled<COMPONENT_NAME>::value>;             \
            template struct copy<COMPONENT_NAME>;                                        \
            template struct assemble<COMPONENT_NAME>;                                    \
            template struct dismantle<COMPONENT_NAME>;                                   \
            template struct finalize::get<COMPONENT_NAME, HAS_DATA>;                     \
            template struct finalize::mpi_get<COMPONENT_NAME, HAS_DATA>;                 \
            template struct finalize::upc_get<COMPONENT_NAME, HAS_DATA>;                 \
            template struct finalize::dmp_get<COMPONENT_NAME, HAS_DATA>;                 \
            template struct finalize::print<COMPONENT_NAME, HAS_DATA>;                   \
            }                                                                            \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#else  // elif defined(_WINDOWS)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_DECLARE_EXTERN_OPERATIONS)
#        define TIMEMORY_DECLARE_EXTERN_OPERATIONS(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS)
#        define TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#endif

//======================================================================================//

#if defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_COMPONENT_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
#    if defined(TIMEMORY_SOURCE) && defined(TIMEMORY_COMPONENT_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
#        if !defined(TIMEMORY_EXTERN_OPERATIONS)
#            define TIMEMORY_EXTERN_OPERATIONS(...)                                      \
                TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(__VA_ARGS__)
#        endif
//
//--------------------------------------------------------------------------------------//
//
#        if !defined(TIMEMORY_EXTERN_STORAGE)
#            define TIMEMORY_EXTERN_STORAGE(...)                                         \
                TIMEMORY_INSTANTIATE_EXTERN_STORAGE(__VA_ARGS__)
#        endif
//
//--------------------------------------------------------------------------------------//
//
#        if !defined(TIMEMORY_EXTERN_TEMPLATE)
#            define TIMEMORY_EXTERN_TEMPLATE(...) template __VA_ARGS__;
#        endif
//
//--------------------------------------------------------------------------------------//
//
#    else
//
//--------------------------------------------------------------------------------------//
//
#        if !defined(TIMEMORY_EXTERN_OPERATIONS)
#            define TIMEMORY_EXTERN_OPERATIONS(...)                                      \
                TIMEMORY_DECLARE_EXTERN_OPERATIONS(__VA_ARGS__)
#        endif
//
//--------------------------------------------------------------------------------------//
//
#        if !defined(TIMEMORY_EXTERN_STORAGE)
#            define TIMEMORY_EXTERN_STORAGE(...)                                         \
                TIMEMORY_DECLARE_EXTERN_STORAGE(__VA_ARGS__)
#        endif
//
//--------------------------------------------------------------------------------------//
//
#        if !defined(TIMEMORY_EXTERN_TEMPLATE)
#            define TIMEMORY_EXTERN_TEMPLATE(...) extern template __VA_ARGS__;
#        endif
//
//--------------------------------------------------------------------------------------//
//
#    endif
//
//--------------------------------------------------------------------------------------//
//
#else
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_OPERATIONS)
#        define TIMEMORY_EXTERN_OPERATIONS(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_STORAGE)
#        define TIMEMORY_EXTERN_STORAGE(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_TEMPLATE)
#        define TIMEMORY_EXTERN_TEMPLATE(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#endif

//======================================================================================//
