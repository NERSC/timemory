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
 * \file timemory/operations/types/construct.hpp
 * \brief Definition for various functions for construct in operations
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
/// \struct tim::operation::construct
/// \brief The purpose of this operation class is construct an object with specific args
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct construct
{
    using type = Tp;

    TIMEMORY_DELETED_OBJECT(construct)

    template <typename Arg, typename... Args>
    construct(type& obj, Arg&& arg, Args&&... args);

    template <typename... Args, enable_if_t<sizeof...(Args) == 0, int> = 0>
    construct(type&, Args&&...);

    template <typename... Args,
              enable_if_t<std::is_constructible<Tp, Args...>::value, int> = 0>
    static auto get(Args&&... args)
    {
        return Tp(std::forward<Args>(args)...);
    }

    template <typename... Args, enable_if_t<!std::is_constructible<Tp, Args...>::value &&
                                                std::is_default_constructible<Tp>::value,
                                            int> = 0>
    static auto get(Args&&...)
    {
        return Tp{};
    }

private:
    // resolution #1 (best)
    // construction is possible with given arguments
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, Args&&... args)
        -> decltype(Up(std::forward<Args>(args)...), void())
    {
        obj = Up(std::forward<Args>(args)...);
    }

    // resolution #2
    // construction is not possible with given arguments
    template <typename Up, typename... Args>
    auto sfinae(Up&, long, Args&&...) -> decltype(void(), void())
    {}
};
//
template <typename Tp>
struct construct<Tp*>
{
    using base_type = construct<Tp>;

    template <typename... Args>
    static Tp* get(Args&&...)
    {
        return nullptr;
    }
};
//
template <typename Tp>
struct construct<std::shared_ptr<Tp>>
{
    using base_type = construct<Tp>;

    template <typename... Args>
    static auto get(Args&&...)
    {
        return std::shared_ptr<Tp>(nullptr);
    }
};
//
template <typename Tp, typename... Deleter>
struct construct<std::unique_ptr<Tp, Deleter...>>
{
    using base_type = construct<Tp>;

    template <typename... Args>
    static auto get(Args&&...)
    {
        return std::unique_ptr<Tp, Deleter...>(nullptr);
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Arg, typename... Args>
construct<Tp>::construct(type& obj, Arg&& arg, Args&&... args)
{
    sfinae(obj, 0, std::forward<Arg>(arg), std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args, enable_if_t<sizeof...(Args) == 0, int>>
construct<Tp>::construct(type&, Args&&...)
{}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
