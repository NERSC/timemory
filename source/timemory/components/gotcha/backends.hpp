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
#include "timemory/operations/types/set_data.hpp"
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
struct gotcha_data;
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

    TIMEMORY_DEFAULT_OBJECT(gotcha_invoker)

    template <typename FuncT, typename... Args>
    static TIMEMORY_INLINE decltype(auto) invoke(Tp& _obj, gotcha_data&& _data,
                                                 FuncT&& _func, Args&&... _args)
    {
        return gotcha_invoker{}(_obj, std::forward<gotcha_data>(_data),
                                std::forward<FuncT>(_func), std::forward<Args>(_args)...);
    }

    template <typename FuncT, typename... Args>
    TIMEMORY_INLINE decltype(auto) operator()(Tp& _obj, gotcha_data&& _data,
                                              FuncT&& _func, Args&&... _args) const
    {
        // if object has set_data(gotcha_data) member function
        operation::set_data<Tp>{}(_obj, std::forward<gotcha_data>(_data));
        // if object has set_data(<function-pointer>) member function
        operation::set_data<Tp>{}(_obj, std::forward<FuncT>(_func));
        //
        return invoke_sfinae(_obj, std::forward<gotcha_data>(_data),
                             std::forward<FuncT>(_func), std::forward<Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //      Ret Type::operator{}(gotcha_data, Args...)
    //
    template <typename DataT, typename FuncT, typename... Args>
    auto sfinae(Tp& _obj, int, int, int, DataT&& _data, FuncT&&, Args&&... _args) const
        -> decltype(_obj(std::forward<DataT>(_data), std::forward<Args>(_args)...))
    {
        return _obj(std::forward<DataT>(_data), std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //      Ret Type::operator{}(<function-pointer>, Args...)
    //
    template <typename DataT, typename FuncT, typename... Args>
    auto sfinae(Tp& _obj, int, int, long, DataT&&, FuncT&& _func, Args&&... _args) const
        -> decltype(_obj(std::forward<FuncT>(_func), std::forward<Args>(_args)...))
    {
        return _obj(std::forward<FuncT>(_func), std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //      Ret Type::operator{}(Args...)
    //
    template <typename DataT, typename FuncT, typename... Args>
    auto sfinae(Tp& _obj, int, long, long, DataT&&, FuncT&&, Args&&... _args) const
        -> decltype(_obj(std::forward<Args>(_args)...))
    {
        return _obj(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Call the original gotcha_wrappee
    //
    template <typename DataT, typename FuncT, typename... Args>
    auto sfinae(Tp&, long, long, long, DataT&&, FuncT&& _func, Args&&... _args) const
        -> decltype(std::forward<FuncT>(_func)(std::forward<Args>(_args)...))
    {
        return std::forward<FuncT>(_func)(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of above
    //
    template <typename DataT, typename FuncT, typename... Args>
    auto invoke_sfinae(Tp& _obj, DataT&& _data, FuncT&& _func, Args&&... _args) const
        -> decltype(sfinae(_obj, 0, 0, 0, std::forward<DataT>(_data),
                           std::forward<FuncT>(_func), std::forward<Args>(_args)...))
    {
        return sfinae(_obj, 0, 0, 0, std::forward<DataT>(_data),
                      std::forward<FuncT>(_func), std::forward<Args>(_args)...);
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
