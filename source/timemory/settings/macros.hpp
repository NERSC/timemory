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
 * \file timemory/settings/macros.hpp
 * \brief Include the macros for settings
 */

#pragma once

#include "timemory/dll.hpp"

//======================================================================================//
//
// Define macros for settings
//
//======================================================================================//
//
#if defined(TIMEMORY_SETTINGS_SOURCE)
//
#    define TIMEMORY_SETTINGS_LINKAGE(...) __VA_ARGS__
//
#else
//
#    if defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_SETTINGS_EXTERN)
//
#        define TIMEMORY_SETTINGS_LINKAGE(...) extern __VA_ARGS__
//
#    else
//
#        define TIMEMORY_SETTINGS_LINKAGE(...) inline __VA_ARGS__
//
#    endif
//
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_SETTINGS_DLL)
#    if defined(TIMEMORY_SETTINGS_SOURCE)
#        define TIMEMORY_SETTINGS_DLL tim_dll_export
#    elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_SETTINGS_EXTERN)
#        define TIMEMORY_SETTINGS_DLL tim_dll_import
#    else
#        define TIMEMORY_SETTINGS_DLL
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_STATIC_ACCESSOR)
#    define TIMEMORY_STATIC_ACCESSOR(TYPE, FUNC, INIT)                                   \
    public:                                                                              \
        static TYPE& FUNC() { return instance()->m__##FUNC; }                            \
                                                                                         \
    private:                                                                             \
        TYPE m__##FUNC = INIT;
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_STATIC_SETTING_INITIALIZER)
#    define TIMEMORY_STATIC_SETTING_INITIALIZER(TYPE, FUNC, ENV_VAR, DESC, INIT)         \
    private:                                                                             \
        static TYPE generate__##FUNC()                                                   \
        {                                                                                \
            auto _parse = []() { FUNC() = tim::get_env<TYPE>(ENV_VAR, FUNC()); };        \
            get_parse_callbacks().insert({ ENV_VAR, _parse });                           \
            get_setting_descriptions().insert({ ENV_VAR, DESC });                        \
            return get_env<TYPE>(ENV_VAR, INIT);                                         \
        }
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_MEMBER_STATIC_ACCESSOR)
#    define TIMEMORY_MEMBER_STATIC_ACCESSOR(TYPE, FUNC, ENV_VAR, DESC, INIT)             \
    public:                                                                              \
        static TYPE& FUNC() { return instance()->m__##FUNC; }                            \
                                                                                         \
    private:                                                                             \
        TYPE generate__##FUNC()                                                          \
        {                                                                                \
            auto _parse = []() { FUNC() = tim::get_env<TYPE>(ENV_VAR, FUNC()); };        \
            get_setting_descriptions().insert({ ENV_VAR, DESC });                        \
            get_parse_callbacks().insert({ ENV_VAR, _parse });                           \
            return get_env<TYPE>(ENV_VAR, INIT);                                         \
        }                                                                                \
        TYPE m__##FUNC = generate__##FUNC();
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_MEMBER_STATIC_REFERENCE)
#    define TIMEMORY_MEMBER_STATIC_REFERENCE(TYPE, FUNC, ENV_VAR, DESC, GETTER, SETTER)  \
    public:                                                                              \
        static TYPE& FUNC() { return *(instance()->m__##FUNC); }                         \
                                                                                         \
    private:                                                                             \
        TYPE& generate__##FUNC()                                                         \
        {                                                                                \
            auto _parse = [&]() {                                                        \
                auto ret = tim::get_env<TYPE>(ENV_VAR, GETTER());                        \
                GETTER() = ret;                                                          \
                SETTER(ret);                                                             \
            };                                                                           \
            get_setting_descriptions().insert({ ENV_VAR, DESC });                        \
            get_parse_callbacks().insert({ ENV_VAR, _parse });                           \
            return GETTER();                                                             \
        }                                                                                \
        TYPE* m__##FUNC = &(generate__##FUNC());
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
            }
//
#    elif defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_SETTINGS_EXTERN)
//
#        define TIMEMORY_SETTINGS_EXTERN_TEMPLATE(API)                                   \
            namespace tim                                                                \
            {                                                                            \
            extern template std::shared_ptr<settings> settings::shared_instance<API>();  \
            extern template settings*                 settings::instance<API>();         \
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
