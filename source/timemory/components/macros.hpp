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
#include "timemory/dll.hpp"

#include <string>
#include <unordered_set>

//======================================================================================//
//
#if defined(TIMEMORY_COMPONENT_SOURCE)
#    define TIMEMORY_COMPONENT_DLL tim_dll_export
#elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_COMPONENT_EXTERN)
#    define TIMEMORY_COMPONENT_DLL tim_dll_import
#else
#    define TIMEMORY_COMPONENT_DLL
#endif
//
//--------------------------------------------------------------------------------------//
//
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

#if !defined(TIMEMORY_PROPERTY_SPECIALIZATION) && !defined(TIMEMORY_DISABLE_PROPERTIES)
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
            using type                  = TYPE;                                          \
            static constexpr bool value = ::tim::trait::is_available<TYPE>::value;       \
        };                                                                               \
        }                                                                                \
        }
#elif !defined(TIMEMORY_PROPERTY_SPECIALIZATION) && defined(TIMEMORY_DISABLE_PROPERTIES)
#    define TIMEMORY_PROPERTY_SPECIALIZATION(...)
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
#        define TIMEMORY_DECLARE_EXTERN_STORAGE(TYPE, ...)                                                            \
            namespace tim                                                                                             \
            {                                                                                                         \
            extern template class TIMEMORY_COMPONENT_DLL                                                              \
                impl::storage<TYPE, implements_storage<TYPE>::value>;                                                 \
            extern template class TIMEMORY_COMPONENT_DLL                                                              \
                                                         storage<TYPE, typename TYPE::value_type>;                    \
            extern template class TIMEMORY_COMPONENT_DLL singleton<                                                   \
                impl::storage<TYPE, implements_storage<TYPE>::value>,                                                 \
                std::unique_ptr<impl::storage<TYPE, implements_storage<TYPE>::value>,                                 \
                                impl::storage_deleter<impl::storage<                                                  \
                                    TYPE, implements_storage<TYPE>::value>>>>;                                        \
            extern template TIMEMORY_COMPONENT_DLL                                                                    \
                                                   storage_singleton<storage<TYPE, typename TYPE::value_type>>*       \
                                                   get_storage_singleton<storage<TYPE, typename TYPE::value_type>>(); \
            extern template TIMEMORY_COMPONENT_DLL storage_initializer                                                \
                                                   storage_initializer::get<TYPE>();                                  \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_INSTANTIATE_EXTERN_STORAGE)
#        define TIMEMORY_INSTANTIATE_EXTERN_STORAGE(TYPE, VAR)                                                 \
            namespace tim                                                                                      \
            {                                                                                                  \
            template class TIMEMORY_COMPONENT_DLL                                                              \
                impl::storage<TYPE, implements_storage<TYPE>::value>;                                          \
            template class TIMEMORY_COMPONENT_DLL                                                              \
                                                  storage<TYPE, typename TYPE::value_type>;                    \
            template class TIMEMORY_COMPONENT_DLL singleton<                                                   \
                impl::storage<TYPE, implements_storage<TYPE>::value>,                                          \
                std::unique_ptr<impl::storage<TYPE, implements_storage<TYPE>::value>,                          \
                                impl::storage_deleter<impl::storage<                                           \
                                    TYPE, implements_storage<TYPE>::value>>>>;                                 \
            template TIMEMORY_COMPONENT_DLL                                                                    \
                                            storage_singleton<storage<TYPE, typename TYPE::value_type>>*       \
                                            get_storage_singleton<storage<TYPE, typename TYPE::value_type>>(); \
            template TIMEMORY_COMPONENT_DLL storage_initializer                                                \
                                            storage_initializer::get<TYPE>();                                  \
            }                                                                                                  \
            namespace                                                                                          \
            {                                                                                                  \
            using namespace tim::component;                                                                    \
            namespace component = tim::component;                                                              \
            tim::storage_initializer storage_initializer__##VAR =                                              \
                tim::storage_initializer::get<TYPE>();                                                         \
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
#        define TIMEMORY_DECLARE_EXTERN_OPERATIONS(COMPONENT_NAME, HAS_DATA)                \
            namespace tim                                                                   \
            {                                                                               \
            namespace operation                                                             \
            {                                                                               \
            extern template struct TIMEMORY_COMPONENT_DLL init_storage<COMPONENT_NAME>;     \
            extern template struct TIMEMORY_COMPONENT_DLL set_prefix<COMPONENT_NAME>;       \
            extern template struct TIMEMORY_COMPONENT_DLL reset<COMPONENT_NAME>;            \
            extern template struct TIMEMORY_COMPONENT_DLL get<COMPONENT_NAME>;              \
            extern template struct TIMEMORY_COMPONENT_DLL print<COMPONENT_NAME>;            \
            extern template struct TIMEMORY_COMPONENT_DLL print_header<COMPONENT_NAME>;     \
            extern template struct TIMEMORY_COMPONENT_DLL                                   \
                                                          print_statistics<COMPONENT_NAME>; \
            extern template struct TIMEMORY_COMPONENT_DLL print_storage<COMPONENT_NAME>;    \
            extern template struct TIMEMORY_COMPONENT_DLL serialization<COMPONENT_NAME>;    \
            extern template struct TIMEMORY_COMPONENT_DLL echo_measurement<                 \
                COMPONENT_NAME, trait::echo_enabled<COMPONENT_NAME>::value>;                \
            extern template struct TIMEMORY_COMPONENT_DLL copy<COMPONENT_NAME>;             \
            extern template struct TIMEMORY_COMPONENT_DLL assemble<COMPONENT_NAME>;         \
            extern template struct TIMEMORY_COMPONENT_DLL derive<COMPONENT_NAME>;           \
            extern template struct TIMEMORY_COMPONENT_DLL                                   \
                finalize::print<COMPONENT_NAME, HAS_DATA>;                                  \
            extern template struct TIMEMORY_COMPONENT_DLL                                   \
                finalize::merge<COMPONENT_NAME, HAS_DATA>;                                  \
            }                                                                               \
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
            template struct TIMEMORY_COMPONENT_DLL init_storage<COMPONENT_NAME>;         \
            template struct TIMEMORY_COMPONENT_DLL set_prefix<COMPONENT_NAME>;           \
            template struct TIMEMORY_COMPONENT_DLL reset<COMPONENT_NAME>;                \
            template struct TIMEMORY_COMPONENT_DLL get<COMPONENT_NAME>;                  \
            template struct TIMEMORY_COMPONENT_DLL print<COMPONENT_NAME>;                \
            template struct TIMEMORY_COMPONENT_DLL print_header<COMPONENT_NAME>;         \
            template struct TIMEMORY_COMPONENT_DLL print_statistics<COMPONENT_NAME>;     \
            template struct TIMEMORY_COMPONENT_DLL print_storage<COMPONENT_NAME>;        \
            template struct TIMEMORY_COMPONENT_DLL serialization<COMPONENT_NAME>;        \
            template struct TIMEMORY_COMPONENT_DLL echo_measurement<                     \
                COMPONENT_NAME, trait::echo_enabled<COMPONENT_NAME>::value>;             \
            template struct TIMEMORY_COMPONENT_DLL copy<COMPONENT_NAME>;                 \
            template struct TIMEMORY_COMPONENT_DLL assemble<COMPONENT_NAME>;             \
            template struct TIMEMORY_COMPONENT_DLL derive<COMPONENT_NAME>;               \
            template struct TIMEMORY_COMPONENT_DLL                                       \
                finalize::print<COMPONENT_NAME, HAS_DATA>;                               \
            template struct TIMEMORY_COMPONENT_DLL                                       \
                finalize::merge<COMPONENT_NAME, HAS_DATA>;                               \
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

#if defined(TIMEMORY_SOURCE) && defined(TIMEMORY_COMPONENT_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_OPERATIONS)
#        define TIMEMORY_EXTERN_OPERATIONS(...)                                          \
            TIMEMORY_INSTANTIATE_EXTERN_OPERATIONS(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_STORAGE)
#        define TIMEMORY_EXTERN_STORAGE(...)                                             \
            TIMEMORY_INSTANTIATE_EXTERN_STORAGE(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_TEMPLATE)
#        define TIMEMORY_EXTERN_TEMPLATE(...) template TIMEMORY_COMPONENT_DLL __VA_ARGS__;
#    endif
//
//--------------------------------------------------------------------------------------//
//
#elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_COMPONENT_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_OPERATIONS)
#        define TIMEMORY_EXTERN_OPERATIONS(...)                                          \
            TIMEMORY_DECLARE_EXTERN_OPERATIONS(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_STORAGE)
#        define TIMEMORY_EXTERN_STORAGE(...) TIMEMORY_DECLARE_EXTERN_STORAGE(__VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_EXTERN_TEMPLATE)
#        define TIMEMORY_EXTERN_TEMPLATE(...)                                            \
            extern template TIMEMORY_COMPONENT_DLL __VA_ARGS__;
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
