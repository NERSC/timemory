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
 * \file timemory/operations/types/start.hpp
 * \brief Definition for various functions for start in operations
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
struct start
{
    using type       = Tp;
    using value_type = typename type::value_type;

    template <typename U>
    using base_t = typename U::base_type;

    TIMEMORY_DELETED_OBJECT(start)

    template <typename... Args>
    explicit start(type& obj, Args&&... args);

    template <typename... Args>
    explicit start(type& obj, non_vexing&&, Args&&... args);

private:
    // resolution #1 (best)
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, int, Args&&... args)
        -> decltype(static_cast<base_t<Up>&>(obj).start(crtp::base{},
                                                        std::forward<Args>(args)...),
                    void())
    {
        static_cast<base_t<Up>&>(obj).start(crtp::base{}, std::forward<Args>(args)...);
    }

    // resolution #2
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, int, long, Args&&... args)
        -> decltype(obj.start(std::forward<Args>(args)...), void())
    {
        obj.start(std::forward<Args>(args)...);
    }

    // resolution #3
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, long, Args&&...)
        -> decltype(static_cast<base_t<Up>&>(obj).start(crtp::base{}), void())
    {
        static_cast<base_t<Up>&>(obj).start(crtp::base{});
    }

    // resolution #4
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, int, long, double, Args&&...) -> decltype(obj.start(), void())
    {
        obj.start();
    }

    // resolution #5 (worst) - no member function or does not satisfy mpl condition
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, long, Args&&...)
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
struct priority_start
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(priority_start)

    template <typename... Args>
    explicit priority_start(type& obj, Args&&... args);

private:
    //  satisfies mpl condition
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        start<Tp>(obj, non_vexing{}, std::forward<Args>(args)...);
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
struct standard_start
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(standard_start)

    template <typename... Args>
    explicit standard_start(type& obj, Args&&... args);

private:
    //  satisfies mpl condition
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        start<Tp>(obj, non_vexing{}, std::forward<Args>(args)...);
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
struct delayed_start
{
    using type       = Tp;
    using value_type = typename type::value_type;

    TIMEMORY_DELETED_OBJECT(delayed_start)

    template <typename... Args>
    explicit delayed_start(type& obj, Args&&... args);

private:
    //  satisfies mpl condition
    template <typename Up, typename... Args>
    auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        start<Tp>(obj, non_vexing{}, std::forward<Args>(args)...);
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
start<Tp>::start(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;
    init_storage<Tp>::init();
    sfinae(obj, 0, 0, 0, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
start<Tp>::start(type& obj, non_vexing&&, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;
    init_storage<Tp>::init();
    sfinae(obj, 0, 0, 0, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
priority_start<Tp>::priority_start(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    using sfinae_type =
        conditional_t<(trait::start_priority<Tp>::value < 0), true_type, false_type>;
    sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
standard_start<Tp>::standard_start(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    using sfinae_type =
        conditional_t<(trait::start_priority<Tp>::value == 0), true_type, false_type>;
    sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
delayed_start<Tp>::delayed_start(type& obj, Args&&... args)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    using sfinae_type =
        conditional_t<(trait::start_priority<Tp>::value > 0), true_type, false_type>;
    sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
