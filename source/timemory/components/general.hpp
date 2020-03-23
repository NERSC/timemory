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
//

/** \file timemory/components/general.hpp
 * \headerfile timemory/components/general.hpp "timemory/components/general.hpp"
 * Defines some short general components
 *
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/data/handler.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/variadic/types.hpp"

#include <cassert>
#include <cstdint>

//======================================================================================//

namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//          General Components with no specific category
//
//--------------------------------------------------------------------------------------//
//
/// \class component::data_tracker
/// \brief This component is provided to facilitate data tracking. The first
/// template parameter is the type of data to be tracked, the second is a custom
/// tag, the third is the implementation for how to track the data.
/// Usage:
///         using tuple_t = tim::auto_tuple<wall_clock, data_tracker<int64>>;
///
///         int64_t num_iter       = 0;
///         double err             = std::numeric_limits<double>::max();
///         const double tolerance = 1.0e-6;
///
///         tuple_t t("iteration_time");
///
///         while(err < tolerance)
///         {
///             num_iter++;
///             // ... do something ...
///         }
///
///         t.store(num_iter);
///
template <typename T, typename Tag = api::native_tag,
          typename Handler = data::handler<T, Tag>>
struct data_tracker : public base<data_tracker<T, Tag, Handler>, T>
{
    using value_type   = T;
    using this_type    = data_tracker<T, Tag, Handler>;
    using base_type    = base<this_type, value_type>;
    using handler_type = Handler;

    using base_type::accum;
    using base_type::value;

    static std::string label()
    {
        std::stringstream ss;
        ss << demangle<Tag>() << "_" << demangle<T>();
        return ss.str();
    }

    static std::string description()
    {
        std::stringstream ss;
        ss << "Data tracker for data of type " << demangle<T>() << " for "
           << demangle<Tag>();
        return ss.str();
    }

    static value_type record() { return T{}; }
    void              start() {}
    void              stop() {}

    void store(const T& val) { handler_type::record(*this, value, val); }

    void mark_begin(const T& val) { handler_type::record(*this, value, val); }

    void mark_end(const T& val) { handler_type::compute(*this, accum, value, val); }

    value_type  get() const { return handler_type::get(*this); }
    std::string get_display() const { return handler_type::get_display(*this); }
};

//--------------------------------------------------------------------------------------//

}  // namespace component

//--------------------------------------------------------------------------------------//
}  // namespace tim
