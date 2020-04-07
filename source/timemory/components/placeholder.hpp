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

/** \file components/placeholder.hpp
 * \headerfile components/placeholder.hpp "timemory/components/placeholder.hpp"
 *
 * This is a wrapper around skeletons that inherits from the base type with a void
 * value type and will ensure that the component is always unavailable during
 * template filtering. Also, the placeholder type inherits from the provided
 * types in the event the skeleton type needs to provide callbacks, etc. that
 * may be defined in a project but never use when wrapped in a skeleton type
 *
 */

#pragma once

#include "timemory/components/base.hpp"

#include <cstdint>

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
// provides nothing, useful to inherit from if a component is not available
//
template <typename... Types>
struct placeholder
: public base<placeholder<Types...>, void>
, public Types...
{
    using value_type = void;
    using this_type  = placeholder<Types...>;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "placeholder"; }
    static std::string description() { return "placeholder"; }
    static void        record() {}
    void               start() {}
    void               stop() {}

    template <typename Tp, typename Func>
    static void set_executor_callback(Func&&)
    {}

    //----------------------------------------------------------------------------------//
    // generic configuration
    //
    template <typename... Args>
    static void configure(Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    // specific to gotcha
    //
    template <size_t N, typename... Ret, typename... Args>
    static void configure(Args&&...)
    {}
};

}  // namespace component

namespace trait
{
// always filter placeholder
template <typename... Types>
struct is_available<component::placeholder<Types...>> : std::false_type
{};
}  // namespace trait

}  // namespace tim
