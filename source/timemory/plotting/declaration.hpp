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
 * \file timemory/plotting/declaration.hpp
 * \brief The declaration for the types for plotting without definitions
 */

#pragma once

#include "timemory/plotting/macros.hpp"
#include "timemory/plotting/types.hpp"
#include "timemory/settings/declaration.hpp"

#include <initializer_list>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              plotting
//
//--------------------------------------------------------------------------------------//
//
namespace plotting
{
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PLOTTING_DLL
void
plot(std::string _label, std::string _prefix, const std::string& _dir, bool _echo_dart,
     std::string _json_file);
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types,
          typename std::enable_if<(sizeof...(Types) > 0), int>::type = 0>
void
plot(std::string _prefix = "", const std::string& _dir = settings::output_path(),
     bool _echo_dart = settings::dart_output(), std::string _json_file = "");
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types,
          typename std::enable_if<(sizeof...(Types) == 0), int>::type = 0>
void
plot(std::string _prefix = "", const std::string& _dir = settings::output_path(),
     bool _echo_dart = settings::dart_output(), std::string _json_file = "");
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Arg, typename... Args>
auto
join(const char* sep, Arg&& arg, Args&&... args)
{
    std::stringstream ss;
    ss << std::forward<Arg>(arg);
    auto tmp =
        ::std::initializer_list<int>{ (ss << sep << std::forward<Args>(args), 0)... };
    tim::consume_parameters(tmp);
    return ss.str();
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
//
//--------------------------------------------------------------------------------------//
//
}  // namespace plotting
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
