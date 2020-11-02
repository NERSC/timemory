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
 * \file timemory/operations/types/stop.hpp
 * \brief Definition for various functions for stop in operations
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
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct stop
{
    using type       = Tp;
    using value_type = typename type::value_type;

    template <typename U>
    using base_t = typename U::base_type;

    TIMEMORY_DELETED_OBJECT(stop)

    explicit stop(type& obj) { impl(obj); }

    template <typename Arg, typename... Args>
    stop(type& obj, Arg&& arg, Args&&... args)
    {
        impl(obj, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto operator()(type& obj, Args&&... args)
    {
        using RetT = decltype(do_sfinae(obj, 0, 0, std::forward<Args>(args)...));
        if(trait::runtime_enabled<type>::get() && is_running<Tp, true>{}(obj))
        {
            set_stopped<Tp>{}(obj);
            return do_sfinae(obj, 0, 0, std::forward<Args>(args)...);
        }
        return RetT{};
    }

private:
    template <typename... Args>
    void impl(type& obj, Args&&... args);

    // resolution #1 (best)
    template <typename Up, typename... Args>
    auto do_sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.stop(std::forward<Args>(args)...))
    {
        return obj.stop(std::forward<Args>(args)...);
    }

    // resolution #2
    template <typename Up, typename... Args>
    auto do_sfinae(Up& obj, int, long, Args&&...) -> decltype(obj.stop())
    {
        return obj.stop();
    }

    // resolution #3 (worst) - no member function
    template <typename Up, typename... Args>
    void do_sfinae(Up&, long, long, Args&&...)
    {
        SFINAE_WARNING(type);
    }
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct priority_stop
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(priority_stop)

    template <typename... Args>
    explicit priority_stop(type& obj, Args&&... args);

private:
    //  satisfies mpl condition
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        stop<Tp> _tmp(obj, std::forward<Args>(args)...);
    }

    //  does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, false_type&&, Args&&...)
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
struct standard_stop
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(standard_stop)

    template <typename... Args>
    explicit standard_stop(type& obj, Args&&... args);

private:
    //  satisfies mpl condition
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        stop<Tp> _tmp(obj, std::forward<Args>(args)...);
    }

    //  does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, false_type&&, Args&&...)
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
struct delayed_stop
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(delayed_stop)

    template <typename... Args>
    explicit delayed_stop(type& obj, Args&&... args);

private:
    //  satisfies mpl condition
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        stop<Tp> _tmp(obj, std::forward<Args>(args)...);
    }

    //  does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, false_type&&, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
void
stop<Tp>::impl(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;
    if(is_running<Tp, true>{}(obj))
    {
        set_stopped<Tp>{}(obj);
        do_sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
priority_stop<Tp>::priority_stop(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    using sfinae_type =
        conditional_t<(trait::stop_priority<Tp>::value < 0), true_type, false_type>;
    sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
standard_stop<Tp>::standard_stop(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    using sfinae_type =
        conditional_t<(trait::stop_priority<Tp>::value == 0), true_type, false_type>;
    sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
delayed_stop<Tp>::delayed_stop(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    using sfinae_type =
        conditional_t<(trait::stop_priority<Tp>::value > 0), true_type, false_type>;
    sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
