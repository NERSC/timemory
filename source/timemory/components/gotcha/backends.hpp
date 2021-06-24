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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/**
 * \file timemory/components/gotcha/backends.hpp
 * \brief Implementation of the gotcha functions/utilities
 */

#pragma once

#include "timemory/backends/gotcha.hpp"
#include "timemory/utility/mangler.hpp"
#include "timemory/variadic/types.hpp"

#include <cassert>
#include <cstdint>
#include <string>
#include <tuple>

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//======================================================================================//
//
class gotcha_suppression
{
private:
    template <size_t Nt, typename Components, typename Differentiator>
    friend struct gotcha;

    template <typename Tp, typename Ret>
    struct gotcha_invoker;

    template <typename Tp>
    friend struct operation::init_storage;

    template <size_t, typename Tp>
    friend struct user_bundle;

    friend struct opaque;

    static bool& get()
    {
        static thread_local bool _instance = false;
        return _instance;
    }

public:
    struct auto_toggle
    {
        explicit auto_toggle(bool& _value, bool _if_equal = false)
        : m_value(_value)
        , m_if_equal(_if_equal)
        {
            if(m_value == m_if_equal)
            {
                m_value      = !m_value;
                m_did_toggle = true;
            }
        }

        ~auto_toggle()
        {
            if(m_value != m_if_equal && m_did_toggle)
            {
                m_value = !m_value;
            }
        }

        auto_toggle(const auto_toggle&) = delete;
        auto_toggle(auto_toggle&&)      = delete;
        auto_toggle& operator=(const auto_toggle&) = delete;
        auto_toggle& operator=(auto_toggle&&) = delete;

    private:
        bool& m_value;
        bool  m_if_equal;
        bool  m_did_toggle = false;
    };
};
//
//======================================================================================//
///
/// \struct tim::component::gotcha_invoker
///
///
template <typename Tp, typename Ret>
struct gotcha_invoker
{
    using Type       = Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename FuncT, typename... Args>
    static decltype(auto) invoke(Tp& _obj, FuncT&& _func, Args&&... _args)
    {
        return invoke_sfinae(_obj, std::forward<FuncT>(_func),
                             std::forward<Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  Call:
    //
    //      Ret Type::operator{}(Args...)
    //
    //  instead of gotcha_wrappee
    //
    template <typename FuncT, typename... Args>
    static auto invoke_sfinae_impl(Tp& _obj, int, FuncT&&, Args&&... _args)
        -> decltype(_obj(std::forward<Args>(_args)...))
    {
        return _obj(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Call the original gotcha_wrappee
    //
    template <typename FuncT, typename... Args>
    static auto invoke_sfinae_impl(Tp&, long, FuncT&& _func, Args&&... _args)
        -> decltype(std::forward<FuncT>(_func)(std::forward<Args>(_args)...))
    {
        return std::forward<FuncT>(_func)(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename FuncT, typename... Args>
    static auto invoke_sfinae(Tp& _obj, FuncT&& _func, Args&&... _args)
        -> decltype(invoke_sfinae_impl(_obj, 0, std::forward<FuncT>(_func),
                                       std::forward<Args>(_args)...))
    {
        return invoke_sfinae_impl(_obj, 0, std::forward<FuncT>(_func),
                                  std::forward<Args>(_args)...);
    }
};
//
//======================================================================================//
//
///
/// \struct tim::component::gotcha_data
/// \brief Holds the properties for wrapping and unwrapping a GOTCHA binding
struct gotcha_data
{
    using binding_t     = backend::gotcha::binding_t;
    using wrappee_t     = backend::gotcha::wrappee_t;
    using wrappid_t     = backend::gotcha::string_t;
    using constructor_t = std::function<void()>;
    using destructor_t  = std::function<void()>;

    gotcha_data()  = default;
    ~gotcha_data() = default;

    gotcha_data(const gotcha_data&) = delete;
    gotcha_data(gotcha_data&&)      = delete;
    gotcha_data& operator=(const gotcha_data&) = delete;
    gotcha_data& operator=(gotcha_data&&) = delete;

    bool          ready        = false;        /// ready to be used NOLINT
    bool          filled       = false;        /// structure is populated NOLINT
    bool          is_active    = false;        /// is currently wrapping NOLINT
    bool          is_finalized = false;        /// no more wrapping is allowed NOLINT
    int           priority     = 0;            /// current priority NOLINT
    binding_t     binding      = binding_t{};  /// hold the binder set NOLINT
    wrappee_t     wrapper      = nullptr;      /// the func pointer doing wrapping NOLINT
    wrappee_t     wrappee      = nullptr;      /// the func pointer being wrapped NOLINT
    wrappid_t     wrap_id      = "";           /// function name (possibly mangled) NOLINT
    wrappid_t     tool_id      = "";           /// function name (unmangled) NOLINT
    constructor_t constructor  = []() {};      /// wrap the function NOLINT
    destructor_t  destructor   = []() {};      /// unwrap the function NOLINT
    bool*         suppression  = nullptr;      /// turn on/off some suppression var NOLINT
    bool*         debug        = nullptr;      //  NOLINT
};
}  // namespace component
}  // namespace tim
//
//======================================================================================//
