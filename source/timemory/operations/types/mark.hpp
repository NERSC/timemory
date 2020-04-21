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
 * \file timemory/operations/types/mark_begin.hpp
 * \brief Definition for various functions for mark_begin in operations
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
///
/// \class operation::mark_begin
/// \brief This operation class is used for asynchronous routines such as \ref cuda_event
/// and \ref nvtx_marker which are passed cudaStream_t instances
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct mark_begin
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(mark_begin)

    template <typename... Args>
    explicit mark_begin(type& obj, Args&&... args);

private:
    //  The equivalent of supports args
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.mark_begin(std::forward<Args>(args)...), void())
    {
        obj.mark_begin(std::forward<Args>(args)...);
    }

    //  Member function is provided
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.mark_begin(), void())
    {
        obj.mark_begin();
    }

    //  No member function
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
///
/// \class operation::mark_end
/// \brief This operation class is used for asynchronous routines such as \ref cuda_event
/// and \ref nvtx_marker which are passed cudaStream_t instances
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct mark_end
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(mark_end)

    template <typename... Args>
    explicit mark_end(type& obj, Args&&... args);

private:
    //  The equivalent of supports args
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.mark_end(std::forward<Args>(args)...), void())
    {
        obj.mark_end(std::forward<Args>(args)...);
    }

    //  Member function is provided
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.mark_end(), void())
    {
        obj.mark_end();
    }

    //  No member function
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
mark_begin<Tp>::mark_begin(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    sfinae(obj, 0, 0, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
mark_end<Tp>::mark_end(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    sfinae(obj, 0, 0, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
