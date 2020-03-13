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

/** \file timemory/plotting.hpp
 * \headerfile timemory/plotting.hpp "timemory/plotting.hpp"
 * Routines for plotting via Python in C++
 *
 */

#pragma once

#include "plotting/definition.hpp"

/*
#include "timemory/mpl/available.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings.hpp"
#include "timemory/types.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <tuple>
#include <type_traits>

namespace tim
{
namespace plotting
{
using string_t     = std::string;
using attributes_t = std::map<string_t, string_t>;

//======================================================================================//

namespace operation
{
//--------------------------------------------------------------------------------------//

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

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
plot(string_t _prefix, const string_t& _dir, bool _echo_dart, string_t _json_file)
{
    auto_lock_t lk(type_mutex<std::ostream>());

    if(settings::debug() || settings::verbose() > 2)
        PRINT_HERE("%s", "");

    if(std::is_same<typename Tp::value_type, void>::value)
    {
        if(settings::debug() || settings::verbose() > 2)
            PRINT_HERE("%s", "");
        return;
    }

    if(!settings::json_output() && !trait::requires_json<Tp>::value)
    {
        if(settings::debug() || settings::verbose() > 2)
            PRINT_HERE("%s", "");
        return;
    }

    if(settings::python_exe().empty())
    {
        fprintf(stderr, "[%s]> Empty '%s' (env: '%s'). Plot generation is disabled...\n",
                demangle<Tp>().c_str(), "tim::settings::python_exe()",
                "TIMEMORY_PYTHON_EXE");
        return;
    }

    if(settings::debug() || settings::verbose() > 2)
        PRINT_HERE("%s", "");

    if(_prefix.empty())
        _prefix = Tp::get_description();

    auto libctor = get_env<std::string>("TIMEMORY_LIBRARY_CTOR", "1");
    auto libdtor = get_env<std::string>("TIMEMORY_LIBRARY_DTOR", "1");

    set_env("TIMEMORY_BANNER", "OFF");
    set_env("TIMEMORY_LIBRARY_CTOR", "0", 1);
    set_env("TIMEMORY_LIBRARY_DTOR", "0", 1);

    if(std::system(nullptr))
    {
        auto _file = _json_file;
        if(_file.empty())
            _file = settings::compose_output_filename(Tp::get_label(), ".json");
        {
            std::ifstream ifs(_file.c_str());
            bool          exists = ifs.good();
            ifs.close();
            if(!exists)
            {
                fprintf(
                    stderr,
                    "[%s]> file '%s' does not exist. Plot generation is disabled...\n",
                    demangle<Tp>().c_str(), _file.c_str());
                return;
            }
        }

        auto cmd = join(" ", settings::python_exe(), "-m", "timemory.plotting", "-f",
                        _file, "-t", "\"" + _prefix, "\"", "-o", _dir);

        if(_echo_dart)
            cmd += " -e";

        if(settings::verbose() > 2 || settings::debug())
            PRINT_HERE("PLOT COMMAND: '%s'", cmd.c_str());

        int sysret = std::system(cmd.c_str());
        if(sysret != 0)
        {
            auto msg = TIMEMORY_JOIN("", "Error generating plots with command: '", cmd,
                                     "'", " Exit code: ", sysret);
            fprintf(stderr, "[%s]> %s\n", TIMEMORY_LABEL("").c_str(), msg.c_str());
        }
    }
    else
    {
        fprintf(stderr, "[%s]> std::system unavailable. Plot generation is disabled...\n",
                demangle<Tp>().c_str());
    }

    set_env("TIMEMORY_LIBRARY_CTOR", libctor, 1);
    set_env("TIMEMORY_LIBRARY_DTOR", libdtor, 1);
}
//
}  // namespace operation
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
//
template <typename... Types>
struct plot
{
    static void generate(string_t _prefix, const string_t& _dir, bool _echo,
                         string_t _json)
    {
        TIMEMORY_FOLD_EXPRESSION(operation::plot<Types>(_prefix, _dir, _echo, _json));
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
struct plot<std::tuple<Types...>> : plot<Types...>
{};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
struct plot<type_list<Types...>> : plot<Types...>
{};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl

//======================================================================================//
//
template <typename... Types, typename std::enable_if<(sizeof...(Types) > 0), int>::type>
void
plot(string_t _prefix, const string_t& _dir, bool _echo_dart, string_t _json_file)
{
    impl::plot<Types...>::generate(_prefix, _dir, _echo_dart, _json_file);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types, typename std::enable_if<(sizeof...(Types) == 0), int>::type>
void
plot(string_t _prefix, const string_t& _dir, bool _echo_dart, string_t _json_file)
{
    impl::plot<tim::available_tuple_t>::generate(_prefix, _dir, _echo_dart, _json_file);
}
//
//======================================================================================//

}  // namespace plotting
}  // namespace tim
*/
