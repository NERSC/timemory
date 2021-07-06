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

#include "timemory/plotting/declaration.hpp"
#include "timemory/plotting/macros.hpp"
#include "timemory/plotting/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/popen.hpp"

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
#if defined(TIMEMORY_PLOTTING_SOURCE) || !defined(TIMEMORY_USE_PLOTTING_EXTERN)
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PLOTTING_LINKAGE(void)
plot(const string_t& _label, const string_t& _prefix, const string_t& _dir,
     bool _echo_dart, const string_t& _json_file)
{
    auto_lock_t lk(type_mutex<std::ostream>());

    if(settings::debug() || settings::verbose() > 2)
        PRINT_HERE("%s", "");

    if(settings::python_exe().empty())
    {
        fprintf(stderr, "[%s]> Empty '%s' (env: '%s'). Plot generation is disabled...\n",
                _label.c_str(), "tim::settings::python_exe()", "TIMEMORY_PYTHON_EXE");
        return;
    }

    if(settings::debug() || settings::verbose() > 2)
        PRINT_HERE("%s", "");

    auto _info = TIMEMORY_LABEL("");

    const auto& _file = _json_file;
    {
        std::ifstream ifs(_file.c_str());
        bool          exists = ifs.good();
        ifs.close();
        if(!exists)
        {
            fprintf(stderr,
                    "[%s]> file '%s' does not exist. Plot generation is disabled...\n",
                    _label.c_str(), _file.c_str());
            return;
        }
    }

    auto _ctor = get_env<std::string>("TIMEMORY_LIBRARY_CTOR", "");
    auto _bann = get_env<std::string>("TIMEMORY_BANNER", "");
    auto _plot = get_env<std::string>("TIMEMORY_CXX_PLOT_MODE", "");
    // if currently in plotting mode, we dont want to plot again
    if(_plot.length() > 0)
        return;

    // set-up environment such that forking is safe
    set_env<std::string>("TIMEMORY_LIBRARY_CTOR", "OFF", 1);
    set_env<std::string>("TIMEMORY_BANNER", "OFF", 1);
    set_env<std::string>("TIMEMORY_CXX_PLOT_MODE", "1", 1);
    auto cmd =
        operation::join(" ", settings::python_exe(), "-m", "timemory.plotting", "-f",
                        _file, "-t", TIMEMORY_JOIN("\"", "", _prefix, ""), "-o", _dir);

    if(_echo_dart)
        cmd += " -e";

    if(settings::verbose() > 2 || settings::debug())
        PRINT_HERE("PLOT COMMAND: '%s'", cmd.c_str());

    std::stringstream _log{};
    auto _success = launch_process(cmd.c_str(), _info + " plot generation failed", &_log);
    if(_success)
    {
        std::cout << _log.str() << '\n';
    }
    else
    {
        std::cerr << _log.str() << '\n';
    }

    // revert the environment
    set_env<std::string>("TIMEMORY_CXX_PLOT_MODE", _plot, 1);
    set_env<std::string>("TIMEMORY_BANNER", _bann, 1);
    set_env<std::string>("TIMEMORY_LIBRARY_CTOR", _ctor, 1);
}
//
//--------------------------------------------------------------------------------------//
//
#endif  // !defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_PLOTTING_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
}  // namespace plotting
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
