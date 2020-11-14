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

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/tpls/cereal/archives.hpp"

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
//
template <typename Tp>
struct serialization
{
    using type       = Tp;
    using value_type = typename type::value_type;

    // TIMEMORY_DELETED_OBJECT(serialization)

    template <typename Archive, typename Up = Tp,
              enable_if_t<is_enabled<Up>::value, char> = 0>
    serialization(const Up& obj, Archive& ar, const unsigned int)
    {
        auto try_catch = [&](const char* key, const auto& val) {
            try
            {
                ar(cereal::make_nvp(key, val));
            } catch(cereal::Exception& e)
            {
                if(settings::debug() || settings::verbose() > -1)
                    fprintf(stderr, "Warning! '%s' threw exception: %s\n", key, e.what());
            }
        };

        try_catch("is_transient", obj.get_is_transient());
        try_catch("laps", obj.get_laps());
        try_catch("value", obj.get_value());
        try_catch("accum", obj.get_accum());
        try_catch("last", obj.get_last());
        try_catch("repr_data", obj.get());
        try_catch("repr_display", obj.get_display());
        // try_catch("units", type::get_unit());
        // try_catch("display_units", type::get_display_unit());
    }

    template <typename Archive, typename Up = Tp,
              enable_if_t<!is_enabled<Up>::value, char> = 0>
    serialization(const Up&, Archive&, const unsigned int)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct extra_serialization
{
    using PrettyJson_t  = cereal::PrettyJSONOutputArchive;
    using MinimalJson_t = cereal::MinimalJSONOutputArchive;

    TIMEMORY_DELETED_OBJECT(extra_serialization)

    explicit extra_serialization(PrettyJson_t& ar, unsigned int ver = 0)
    {
        sfinae(ar, ver, 0, 0);
    }

    explicit extra_serialization(MinimalJson_t& ar, unsigned int ver = 0)
    {
        sfinae(ar, ver, 0, 0);
    }

    template <typename Archive>
    explicit extra_serialization(Archive& ar, unsigned int ver = 0)
    {
        sfinae(ar, ver, 0, 0);
    }

private:
    template <typename Archive, typename Up = Tp>
    auto sfinae(Archive& ar, unsigned int ver, int, int)
        -> decltype(Up::extra_serialization(ar, ver), void())
    {
        Up::extra_serialization(ar, ver);
    }

    template <typename Archive, typename Up = Tp>
    auto sfinae(Archive& ar, unsigned int, int, long)
        -> decltype(Up::extra_serialization(ar), void())
    {
        Up::extra_serialization(ar);
    }

    template <typename Archive, typename Up = Tp>
    auto sfinae(Archive&, unsigned int, long, long)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
