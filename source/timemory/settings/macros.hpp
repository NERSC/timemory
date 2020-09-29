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

#include "timemory/compat/macros.h"

#if defined(TIMEMORY_CORE_SOURCE)
#    define TIMEMORY_SETTINGS_SOURCE
#elif defined(TIMEMORY_USE_CORE_EXTERN)
#    define TIMEMORY_USE_SETTINGS_EXTERN
#endif
//
#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_SETTINGS_EXTERN)
#    define TIMEMORY_USE_SETTINGS_EXTERN
#endif
//
#if defined(TIMEMORY_SETTINGS_SOURCE)
#    define TIMEMORY_SETTINGS_COMPILE_MODE
#    define TIMEMORY_SETTINGS_INLINE
#    define TIMEMORY_SETTINGS_LINKAGE(...) __VA_ARGS__
#elif defined(TIMEMORY_USE_SETTINGS_EXTERN)
#    define TIMEMORY_SETTINGS_EXTERN_MODE
#    define TIMEMORY_SETTINGS_INLINE
#    define TIMEMORY_SETTINGS_LINKAGE(...) __VA_ARGS__
#else
#    define TIMEMORY_SETTINGS_HEADER_MODE
#    define TIMEMORY_SETTINGS_INLINE inline
#    define TIMEMORY_SETTINGS_LINKAGE(...) inline __VA_ARGS__
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_STATIC_ACCESSOR)
#    define TIMEMORY_STATIC_ACCESSOR(TYPE, FUNC, INIT)                                   \
    public:                                                                              \
        static TYPE& FUNC() TIMEMORY_VISIBILITY("default")                               \
        {                                                                                \
            return instance()->m__##FUNC;                                                \
        }                                                                                \
                                                                                         \
    private:                                                                             \
        TYPE m__##FUNC = INIT;
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_MEMBER_DECL)
#    define TIMEMORY_SETTINGS_MEMBER_DECL(TYPE, FUNC, ENV_VAR)                           \
    public:                                                                              \
        TYPE& get_##FUNC()                                                               \
        {                                                                                \
            return static_cast<tsettings<TYPE>*>(m_data[ENV_VAR].get())->get();          \
        }                                                                                \
                                                                                         \
        TYPE get_##FUNC() const                                                          \
        {                                                                                \
            auto ret = m_data.find(ENV_VAR);                                             \
            if(ret == m_data.end())                                                      \
                return TYPE{};                                                           \
            if(!ret->second)                                                             \
                return TYPE{};                                                           \
            return static_cast<tsettings<TYPE>*>(ret->second.get())->get();              \
        }                                                                                \
                                                                                         \
        static TYPE& FUNC() TIMEMORY_VISIBILITY("default")                               \
        {                                                                                \
            return instance()->get_##FUNC();                                             \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_REFERENCE_DECL)
#    define TIMEMORY_SETTINGS_REFERENCE_DECL(TYPE, FUNC, ENV_VAR)                        \
    public:                                                                              \
        TYPE& get_##FUNC()                                                               \
        {                                                                                \
            return static_cast<tsettings<TYPE, TYPE&>*>(m_data[ENV_VAR].get())->get();   \
        }                                                                                \
                                                                                         \
        TYPE get_##FUNC() const                                                          \
        {                                                                                \
            auto ret = m_data.find(ENV_VAR);                                             \
            if(ret == m_data.end())                                                      \
                return TYPE{};                                                           \
            if(!ret->second)                                                             \
                return TYPE{};                                                           \
            return static_cast<tsettings<TYPE, TYPE&>*>(ret->second.get())->get();       \
        }                                                                                \
                                                                                         \
        static TYPE& FUNC() TIMEMORY_VISIBILITY("default")                               \
        {                                                                                \
            return instance()->get_##FUNC();                                             \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_MEMBER_IMPL)
#    define TIMEMORY_SETTINGS_MEMBER_IMPL(TYPE, FUNC, ENV_VAR, DESC, INIT)               \
        m_order.push_back(ENV_VAR);                                                      \
        m_data.insert(                                                                   \
            { ENV_VAR, std::make_shared<tsettings<TYPE>>(INIT, #FUNC, ENV_VAR, DESC) });
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_MEMBER_ARG_IMPL)
#    define TIMEMORY_SETTINGS_MEMBER_ARG_IMPL(TYPE, FUNC, ENV_VAR, DESC, INIT, ...)      \
        m_order.push_back(ENV_VAR);                                                      \
        m_data.insert({ ENV_VAR, std::make_shared<tsettings<TYPE>>(                      \
                                     INIT, #FUNC, ENV_VAR, DESC, __VA_ARGS__) });
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_REFERENCE_IMPL)
#    define TIMEMORY_SETTINGS_REFERENCE_IMPL(TYPE, FUNC, ENV_VAR, DESC, INIT)            \
        m_order.push_back(ENV_VAR);                                                      \
        m_data.insert({ ENV_VAR, std::make_shared<tsettings<TYPE, TYPE&>>(               \
                                     INIT, #FUNC, ENV_VAR, DESC) });
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL)
#    define TIMEMORY_SETTINGS_REFERENCE_ARG_IMPL(TYPE, FUNC, ENV_VAR, DESC, INIT, ...)   \
        m_order.push_back(ENV_VAR);                                                      \
        m_data.insert({ ENV_VAR, std::make_shared<tsettings<TYPE, TYPE&>>(               \
                                     INIT, #FUNC, ENV_VAR, DESC, __VA_ARGS__) });
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_ERROR_FUNCTION_MACRO)
#    if defined(__PRETTY_FUNCTION__)
#        define TIMEMORY_ERROR_FUNCTION_MACRO __PRETTY_FUNCTION__
#    else
#        define TIMEMORY_ERROR_FUNCTION_MACRO __FUNCTION__
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_TRY_CATCH_NVP)
#    define TIMEMORY_SETTINGS_TRY_CATCH_NVP(ENV_VAR, FUNC)                               \
        try                                                                              \
        {                                                                                \
            auto& _VAL = FUNC();                                                         \
            ar(cereal::make_nvp(ENV_VAR, _VAL));                                         \
        } catch(...)                                                                     \
        {}
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_EXTERN_TEMPLATE)
//
#    if defined(TIMEMORY_SETTINGS_SOURCE)
//
#        define TIMEMORY_SETTINGS_EXTERN_TEMPLATE(API)                                   \
            namespace tim                                                                \
            {                                                                            \
            template std::shared_ptr<settings> settings::shared_instance<API>();         \
            template settings*                 settings::instance<API>();                \
            template void settings::serialize_settings(cereal::JSONInputArchive&);       \
            template void settings::serialize_settings(                                  \
                cereal::PrettyJSONOutputArchive&);                                       \
            template void settings::serialize_settings(                                  \
                cereal::MinimalJSONOutputArchive&);                                      \
            template void settings::serialize_settings(cereal::JSONInputArchive&,        \
                                                       settings&);                       \
            template void settings::serialize_settings(cereal::PrettyJSONOutputArchive&, \
                                                       settings&);                       \
            template void settings::serialize_settings(                                  \
                cereal::MinimalJSONOutputArchive&, settings&);                           \
            template void settings::save(cereal::PrettyJSONOutputArchive&,               \
                                         const unsigned int) const;                      \
            template void settings::save(cereal::MinimalJSONOutputArchive&,              \
                                         const unsigned int) const;                      \
            template void settings::load(cereal::JSONInputArchive&, const unsigned int); \
            }
//
#    elif defined(TIMEMORY_USE_SETTINGS_EXTERN)
//
#        define TIMEMORY_SETTINGS_EXTERN_TEMPLATE(API)                                   \
            namespace tim                                                                \
            {                                                                            \
            extern template std::shared_ptr<settings> settings::shared_instance<API>();  \
            extern template settings*                 settings::instance<API>();         \
            extern template void                      settings::serialize_settings(      \
                cereal::JSONInputArchive&);                         \
            extern template void settings::serialize_settings(                           \
                cereal::PrettyJSONOutputArchive&);                                       \
            extern template void settings::serialize_settings(                           \
                cereal::MinimalJSONOutputArchive&);                                      \
            extern template void settings::serialize_settings(cereal::JSONInputArchive&, \
                                                              settings&);                \
            extern template void settings::serialize_settings(                           \
                cereal::PrettyJSONOutputArchive&, settings&);                            \
            extern template void settings::serialize_settings(                           \
                cereal::MinimalJSONOutputArchive&, settings&);                           \
            extern template void settings::save(cereal::PrettyJSONOutputArchive&,        \
                                                const unsigned int) const;               \
            extern template void settings::save(cereal::MinimalJSONOutputArchive&,       \
                                                const unsigned int) const;               \
            extern template void settings::load(cereal::JSONInputArchive&,               \
                                                const unsigned int);                     \
            }
//
#    else
//
#        define TIMEMORY_SETTINGS_EXTERN_TEMPLATE(...)
//
#    endif
#endif
//
//======================================================================================//
//
