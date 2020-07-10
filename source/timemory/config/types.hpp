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
 * \file timemory/config/types.hpp
 * \brief Declare the config types
 */

#pragma once

#include "timemory/config/macros.hpp"
#include "timemory/utility/types.hpp"

#include <string>
#include <utility>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              config
//
//--------------------------------------------------------------------------------------//
//
/// initialization (creates manager and configures output path)
//
TIMEMORY_CONFIG_DLL void
timemory_init(int argc, char** argv, const std::string& _prefix = "timemory-",
              const std::string& _suffix = "-output");
//
//--------------------------------------------------------------------------------------//
//
/// initialization (creates manager and configures output path)
TIMEMORY_CONFIG_DLL void
timemory_init(const std::string& exe_name, const std::string& _prefix = "timemory-",
              const std::string& _suffix = "-output");
//
//--------------------------------------------------------------------------------------//
//
/// initialization (creates manager, configures output path, mpi_init)
TIMEMORY_CONFIG_DLL void
timemory_init(int* argc, char*** argv, const std::string& _prefix = "timemory-",
              const std::string& _suffix = "-output");
//
//--------------------------------------------------------------------------------------//
//
/// finalization of the specified types
TIMEMORY_CONFIG_DLL void
timemory_finalize();
//
//--------------------------------------------------------------------------------------//
//
template <typename... Args>
void
init(Args&&... args)
{
    timemory_init(std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
finalize()
{
    timemory_finalize();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types, typename... Args,
          enable_if_t<(sizeof...(Types) > 0 && sizeof...(Args) >= 2), int> = 0>
inline void
timemory_init(Args&&... _args);
//
//--------------------------------------------------------------------------------------//
//
namespace config
{
/// \fn tim::config::read_command_line()
/// \brief this only works on Linux where there is a /proc/<PID>/cmdline file
///
template <typename Func>
void
read_command_line(Func&& _func);
//
}  // namespace config
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
