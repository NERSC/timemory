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
 * \file timemory/operations/types/audit.hpp
 * \brief Definition for various functions for audit in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::audit
///
/// \brief The purpose of this operation class is for a component to provide some extra
/// customization within a GOTCHA function. It allows a GOTCHA component to inspect
/// the arguments and the return type of a wrapped function. To add support to a
/// component, define `void audit(std::string, context, <Args...>)`. The first argument is
/// the function name (possibly mangled), the second is either type \struct
/// audit::incoming or \struct audit::outgoing, and the remaining arguments are the
/// corresponding types
///
/// One such purpose may be to create a custom component that intercepts a malloc and
/// uses the arguments to get the exact allocation size.
///
template <typename Tp>
struct audit
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(audit)

    template <typename... Args>
    audit(type& obj, Args&&... args)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    // resolution #1 (best)
    // operation is supported with given arguments
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.audit(std::forward<Args>(args)...), void())
    {
        obj.audit(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    // resolution #2
    // operation is supported with first argument only
    //
    template <typename Up, typename Arg, typename... Args>
    auto sfinae(Up& obj, int, long, Arg&& arg, Args&&...)
        -> decltype(obj.audit(std::forward<Arg>(arg)), void())
    {
        obj.audit(std::forward<Arg>(arg));
    }

    //----------------------------------------------------------------------------------//
    // resolution #3
    // operation is not supported
    //
    template <typename Up, typename... Args>
    auto sfinae(Up&, long, long, Args&&...) -> decltype(void(), void())
    {}
    //
    //----------------------------------------------------------------------------------//
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
