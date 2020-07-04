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

/** \file timemory/data/handler.hpp
 * \headerfile timemory/data/handler.hpp "timemory/data/handler.hpp"
 * Defines the data_tracker handler component interface
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"

#include <sstream>
#include <string>

namespace tim
{
namespace data
{
/// \struct data::handler<V, Tag>
/// \brief This class is used to provide the implementation for tracking data
/// via the data_tracker component. It is written such that it can be
/// defined for a specific data type (e.g. int) with a tag (e.g. ConvergenceIterations)
/// and shared among multiple template types of data_tracker<T>
///
template <typename V, typename Tag = api::native_tag>
struct handler
{
    using value_type = V;

    template <typename T>
    static void store(T& obj, const value_type& v)
    {
        obj.set_value(v);
    }

    template <typename T, typename Func>
    static auto store(T& obj, Func&& f, const value_type& v)
        -> decltype(obj.set_value(f(obj.get_value(), v)), void())
    {
        obj.set_value(f(obj.get_value(), v));
    }

    template <typename T>
    static void begin(T& obj, const value_type& v)
    {
        store(obj, v);
    }

    template <typename T>
    static void end(T& obj, const value_type& v)
    {
        obj.set_value(v - obj.get_value());
    }

    template <typename T>
    static auto get(const T& obj)
    {
        return obj.load();
    }

    template <typename T>
    static auto get_display(const T& obj)
    {
        return obj.load();
    }
};
}  // namespace data
}  // namespace tim
