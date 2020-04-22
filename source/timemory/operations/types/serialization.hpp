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
 * \file timemory/operations/types/serialization.hpp
 * \brief Definition for various functions for serialization in operations
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
struct serialization
{
    using type       = Tp;
    using value_type = typename type::value_type;

    // TIMEMORY_DELETED_OBJECT(serialization)

    template <typename Archive, typename Up = Tp,
              enable_if_t<(is_enabled<Up>::value), char> = 0>
    serialization(const Up& obj, Archive& ar, const unsigned int)
    {
        // clang-format off
        ar(cereal::make_nvp("is_transient", obj.get_is_transient()),
           cereal::make_nvp("laps", obj.get_laps()),
           cereal::make_nvp("value", obj.get_value()),
           cereal::make_nvp("accum", obj.get_accum()),
           cereal::make_nvp("last", obj.get_last()),
           cereal::make_nvp("repr_data", obj.get()),
           cereal::make_nvp("repr_display", obj.get_display()),
           cereal::make_nvp("units", type::get_unit()),
           cereal::make_nvp("display_units", type::get_display_unit()));
        // clang-format on
    }

    template <typename Archive, typename Up = Tp,
              enable_if_t<!(is_enabled<Up>::value), char> = 0>
    serialization(const Up&, Archive&, const unsigned int)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
