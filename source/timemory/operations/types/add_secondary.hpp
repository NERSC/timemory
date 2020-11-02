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
 * \file timemory/operations/types/add_secondary.hpp
 * \brief Definition for various functions for add_secondary in operations
 */

#pragma once

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct operation::add_secondary
/// \brief
/// component contains secondary data resembling the original data
/// but should be another node entry in the graph. These types
/// must provide a get_secondary() member function and that member function
/// must return a pair-wise iterable container, e.g. std::map, of types:
///     - std::string
///     - value_type
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct add_secondary
{
    using type       = Tp;
    using value_type = typename type::value_type;
    using string_t   = std::string;

    //----------------------------------------------------------------------------------//
    // if secondary data explicitly specified
    //
    template <typename Storage, typename Iterator, typename Up = type,
              enable_if_t<trait::secondary_data<Up>::value, int> = 0>
    add_secondary(Storage* _storage, Iterator _itr, const Up& _rhs)
    {
        if(!trait::runtime_enabled<Tp>::get() || _storage == nullptr ||
           !settings::add_secondary())
            return;

        using secondary_data_t = std::tuple<Iterator, const string_t&, value_type>;
        for(const auto& _data : _rhs.get_secondary())
            _storage->append(secondary_data_t{ _itr, _data.first, _data.second });
    }

    //----------------------------------------------------------------------------------//
    // check if secondary data implicitly specified
    //
    template <typename Storage, typename Iterator, typename Up = type,
              enable_if_t<!trait::secondary_data<Up>::value, int> = 0>
    add_secondary(Storage* _storage, Iterator _itr, const Up& _rhs)
    {
        add_secondary_sfinae(_storage, _itr, _rhs, 0);
    }

    //----------------------------------------------------------------------------------//
    // add_secondary called on type
    //
    template <typename... Args>
    add_secondary(type& _rhs, Args&&... args)
    {
        sfinae(_rhs, 0, std::forward<Args>(args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  If the component has a get_secondary() member function
    //
    template <typename Storage, typename Iterator, typename Up = type>
    auto add_secondary_sfinae(Storage* _storage, Iterator _itr, const Up& _rhs, int)
        -> decltype(_rhs.get_secondary(), void())
    {
        if(!trait::runtime_enabled<Tp>::get() || _storage == nullptr ||
           !settings::add_secondary())
            return;

        using secondary_data_t = std::tuple<Iterator, const string_t&, value_type>;
        for(const auto& _data : _rhs.get_secondary())
            _storage->append(secondary_data_t{ _itr, _data.first, _data.second });
    }

    //----------------------------------------------------------------------------------//
    //  If the component does not have a get_secondary() member function
    //
    template <typename Storage, typename Iterator, typename Up = type>
    auto add_secondary_sfinae(Storage*, Iterator, const Up&, long)
        -> decltype(void(), void())
    {}

    //----------------------------------------------------------------------------------//
    //  If the component has a add_secondary(Args...) member function
    //
    template <typename Up, typename... Args>
    auto sfinae(Up& _obj, int, Args&&... args)
        -> decltype(_obj.add_secondary(std::forward<Args>(args)...), void())
    {
        if(!trait::runtime_enabled<Tp>::get() || !settings::add_secondary())
            return;

        _obj.add_secondary(std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
    //  If the component does not have a add_secondary(Args...) member function
    //
    template <typename Up, typename... Args>
    auto sfinae(Up&, long, Args&&...) -> decltype(void(), void())
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
