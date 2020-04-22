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
 * \file timemory/operations/types/insert_node.hpp
 * \brief Definition for various functions for insert_node in operations
 */

#pragma once

//======================================================================================//
//
#include "timemory/operations/macros.hpp"
//
#include "timemory/operations/types.hpp"
//
#include "timemory/operations/declaration.hpp"
//
//======================================================================================//

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct insert_node
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(insert_node)

    insert_node(type& obj, scope::config _scope, int64_t _hash)
    {
        if(!trait::runtime_enabled<type>::get())
            return;

        sfinae(obj, 0, _scope, _hash);
    }

private:
    //  satisfies mpl condition and accepts arguments
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, Args&&... args)
        -> decltype(obj.insert_node(std::forward<Args>(args)...), void())
    {
        obj.insert_node(std::forward<Args>(args)...);
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct pop_node
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(pop_node)

    template <typename... Args>
    explicit pop_node(type& obj, Args&&... args)
    {
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    //  satisfies mpl condition and accepts arguments
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.pop_node(std::forward<Args>(args)...), void())
    {
        obj.pop_node(std::forward<Args>(args)...);
    }

    //  satisfies mpl condition but does not accept arguments
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.pop_node(), void())
    {
        obj.pop_node();
    }

    //  no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
