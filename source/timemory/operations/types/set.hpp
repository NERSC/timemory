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
 * \file timemory/operations/types/set_prefix.hpp
 * \brief Definition for various functions for set_prefix in operations
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
struct set_prefix
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;
    using string_t   = std::string;

    TIMEMORY_DELETED_OBJECT(set_prefix)

    template <typename Up = Tp, enable_if_t<(trait::requires_prefix<Up>::value), int> = 0>
    set_prefix(type& obj, const string_t& prefix);

    template <typename Up                                            = Tp,
              enable_if_t<!(trait::requires_prefix<Up>::value), int> = 0>
    set_prefix(type& obj, const string_t& prefix);

private:
    //  If the component has a set_prefix(const string_t&) member function
    template <typename U = type>
    auto sfinae(U& obj, int, const string_t& prefix)
        -> decltype(obj.set_prefix(prefix), void())
    {
        if(!trait::runtime_enabled<U>::get())
            return;

        obj.set_prefix(prefix);
    }

    //  If the component does not have a set_prefix(const string_t&) member function
    template <typename U = type>
    auto sfinae(U&, long, const string_t&) -> decltype(void(), void())
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
struct set_flat_profile
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;
    using string_t   = std::string;

    TIMEMORY_DELETED_OBJECT(set_flat_profile)

    set_flat_profile(type& obj, bool flat);

private:
    //  If the component has a set_flat_profile(bool) member function
    template <typename T = type>
    auto sfinae(T& obj, int, bool flat) -> decltype(obj.set_flat_profile(flat), void())
    {
        obj.set_flat_profile(flat);
    }

    //  If the component does not have a set_flat_profile(bool) member function
    template <typename T = type>
    auto sfinae(T&, long, bool) -> decltype(void(), void())
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
struct set_timeline_profile
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;
    using string_t   = std::string;

    TIMEMORY_DELETED_OBJECT(set_timeline_profile)

    set_timeline_profile(type& obj, bool flat);

private:
    //  If the component has a set_timeline_profile(bool) member function
    template <typename T = type>
    auto sfinae(T& obj, int, bool flat)
        -> decltype(obj.set_timeline_profile(flat), void())
    {
        obj.set_timeline_profile(flat);
    }

    //  If the component does not have a set_timeline_profile(bool) member function
    template <typename T = type>
    auto sfinae(T&, long, bool) -> decltype(void(), void())
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up, enable_if_t<(trait::requires_prefix<Up>::value), int>>
set_prefix<Tp>::set_prefix(type& obj, const string_t& prefix)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    obj.set_prefix(prefix);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up, enable_if_t<!(trait::requires_prefix<Up>::value), int>>
set_prefix<Tp>::set_prefix(type& obj, const string_t& prefix)
{
    sfinae(obj, 0, prefix);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
set_flat_profile<Tp>::set_flat_profile(type& obj, bool flat)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    sfinae(obj, 0, flat);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
set_timeline_profile<Tp>::set_timeline_profile(type& obj, bool flat)
{
    if(!trait::runtime_enabled<type>::get())
        return;

    sfinae(obj, 0, flat);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
