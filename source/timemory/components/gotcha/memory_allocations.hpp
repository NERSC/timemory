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

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/gotcha/components.hpp"
#include "timemory/components/gotcha/types.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types/construct.hpp"
#include "timemory/types.hpp"
#include "timemory/variadic/component_tuple.hpp"
#include "timemory/variadic/types.hpp"

#include <memory>
#include <string>

namespace tim
{
namespace component
{
/// \struct memory_allocations
/// \brief This component wraps malloc, calloc, free, cudaMalloc, cudaFree via
/// GOTCHA and tracks the number of bytes requested/freed in each call.
/// This component is useful for detecting the locations where memory re-use
/// would provide a performance benefit.
///
struct memory_allocations
: base<memory_allocations, void>
, public concepts::external_function_wrapper
, public policy::instance_tracker<memory_allocations, true>
{
    using value_type   = void;
    using this_type    = memory_allocations;
    using base_type    = base<this_type, value_type>;
    using tracker_type = policy::instance_tracker<memory_allocations, true>;

    using malloc_gotcha_t = typename malloc_gotcha::gotcha_type<component_tuple_t<>>;
    using malloc_bundle_t = component_tuple_t<malloc_gotcha_t>;
    using data_pointer_t  = std::unique_ptr<malloc_bundle_t>;

    static std::string label() { return "memory_allocations"; }
    static std::string description()
    {
        return "Number of bytes allocated/freed instead of peak/current memory usage: "
               "free(malloc(10)) + free(malloc(10)) would use 10 bytes but this would "
               "report 20 bytes";
    }

    static void global_init() { malloc_gotcha::configure<component_tuple_t<>>(); }
    static void global_finalize() { malloc_gotcha::tear_down<component_tuple_t<>>(); }

    void start()
    {
        auto _cnt = tracker_type::start();
        if(_cnt.first == 0 && _cnt.second == 0 && !get_data())
        {
            get_data().reset(new malloc_bundle_t{});
            get_data()->start();
        }
    }

    void stop()
    {
        auto _cnt = tracker_type::stop();
        if(_cnt.first == 0 && _cnt.second == 0 && get_data())
        {
            get_data()->stop();
            get_data().reset(nullptr);
        }
    }

private:
    static data_pointer_t& get_data()
    {
        static auto _instance = data_pointer_t{};
        return _instance;
    }
};
//
}  // namespace component
}  // namespace tim
