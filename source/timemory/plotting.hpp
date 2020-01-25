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

#include "timemory/components.hpp"
#include "timemory/settings.hpp"
#include "timemory/types.hpp"
#include "timemory/variadic/macros.hpp"

#include <cstdio>
#include <cstdlib>
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

namespace impl
{
//--------------------------------------------------------------------------------------//

template <typename... _Tail, typename... _Args,
          typename std::enable_if<(sizeof...(_Tail) == 0), int>::type = 0>
void
_plot(_Args&&...)
{}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Tail,
          typename std::enable_if<(sizeof...(_Tail) == 0), int>::type = 0>
void
_plot(string_t _prefix = "", const string_t& _dir = settings::output_path(),
      bool echo_dart = settings::dart_output(), string_t _json_file = "")
{
    using storage_type = typename _Tp::storage_type;

    if(settings::debug() || settings::verbose() > 2)
        PRINT_HERE("%s", "");

    if(std::is_same<typename _Tp::value_type, void>::value)
        return;

    if(!settings::json_output() && !trait::requires_json<_Tp>::value)
        return;

    if(settings::python_exe().empty())
    {
        fprintf(stderr, "[%s]> Empty '%s' (env: '%s'). Plot generation is disabled...",
                demangle<_Tp>().c_str(), "tim::settings::python_exe()",
                "TIMEMORY_PYTHON_EXE");
        return;
    }

    auto ret = storage_type::noninit_instance();
    if(!ret)
        return;

    if(ret->empty())
        return;

    if(_prefix.empty())
        _prefix = _Tp::get_description();

    auto libctor = get_env<std::string>("TIMEMORY_LIBRARY_CTOR", "1");
    auto libdtor = get_env<std::string>("TIMEMORY_LIBRARY_DTOR", "1");

    set_env("TIMEMORY_BANNER", "OFF");
    set_env("TIMEMORY_LIBRARY_CTOR", "0", 1);
    set_env("TIMEMORY_LIBRARY_DTOR", "0", 1);

    if(std::system(nullptr))
    {
        auto _file = _json_file;
        if(_file.empty())
            _file = settings::compose_output_filename(_Tp::get_label(), ".json");
        {
            std::ifstream ifs(_file.c_str());
            bool          exists = ifs.good();
            ifs.close();
            if(!exists)
                return;
        }
        auto cmd = TIMEMORY_JOIN(" ", settings::python_exe(), "-m", "timemory.plotting",
                                 "-f", _file, "-t", "\"" + _prefix, "\"", "-o", _dir);
        if(echo_dart)
            cmd += " -e";

        int sysret = std::system(cmd.c_str());
        if(sysret != 0)
        {
            auto msg =
                TIMEMORY_JOIN("", "Error generating plots with command: '", cmd, "'");
            fprintf(stderr, "[%s]> %s\n", TIMEMORY_LABEL("").c_str(), msg.c_str());
        }
    }

    set_env("TIMEMORY_LIBRARY_CTOR", libctor, 1);
    set_env("TIMEMORY_LIBRARY_DTOR", libdtor, 1);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _Tail, typename... _Args,
          typename std::enable_if<(sizeof...(_Tail) > 0), int>::type = 0>
void
_plot(_Args&&... _args)
{
    _plot<_Tp>(std::forward<_Args>(_args)...);
    _plot<_Tail...>(std::forward<_Args>(_args)...);
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct plot
{
    template <typename... _Args>
    static void generate(_Args&&... _args)
    {
        _plot<_Types...>(std::forward<_Args>(_args)...);
    }
};

//--------------------------------------------------------------------------------------//

template <typename... _Types>
struct plot<std::tuple<_Types...>>
{
    template <typename... _Args>
    static void generate(_Args&&... _args)
    {
        _plot<_Types...>(std::forward<_Args>(_args)...);
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace impl

//======================================================================================//

template <typename... _Types, typename... _Args,
          typename std::enable_if<(sizeof...(_Types) > 0), int>::type>
inline void
plot(_Args&&... _args)
{
    impl::plot<_Types...>::generate(std::forward<_Args>(_args)...);
}

//======================================================================================//

template <typename... _Types, typename... _Args,
          typename std::enable_if<(sizeof...(_Types) == 0), int>::type>
inline void
plot(_Args&&... _args)
{
    using tuple_type = tim::available_tuple<tim::complete_tuple_t>;
    impl::plot<tuple_type>::generate(std::forward<_Args>(_args)...);
}

//======================================================================================//

inline void
echo_dart_file(const string_t& filepath, attributes_t attributes)
{
    auto attribute_string = [](const string_t& key, const string_t& item) {
        return TIMEMORY_JOIN("", key, "=", "\"", item, "\"");
    };

    auto lowercase = [](string_t _str) {
        for(auto& itr : _str)
            itr = tolower(itr);
        return _str;
    };

    auto contains = [&lowercase](const string_t& str, std::set<string_t> items) {
        for(const auto& itr : items)
        {
            if(lowercase(str).find(itr) != string_t::npos)
                return true;
        }
        return false;
    };

    auto is_numeric = [](const string_t& str) -> bool {
        return (str.find_first_not_of("0123456789.e+-*/") == string_t::npos);
    };

    if(attributes.find("name") == attributes.end())
    {
        auto name = filepath;
        if(name.find("/") != string_t::npos)
            name = name.substr(name.find_last_of("/") + 1);
        if(name.find("\\") != string_t::npos)
            name = name.substr(name.find_last_of("\\") + 1);
        if(name.find(".") != string_t::npos)
            name.erase(name.find_last_of("."));
        attributes["name"] = name;
    }

    if(attributes.find("type") == attributes.end())
    {
        if(contains(filepath, { ".jpeg", ".jpg" }))
            attributes["type"] = "image/jpeg";
        else if(contains(filepath, { ".png" }))
            attributes["type"] = "image/png";
        else if(contains(filepath, { ".tiff", ".tif" }))
            attributes["type"] = "image/tiff";
        else if(contains(filepath, { ".txt" }))
        {
            bool          numeric_file = true;
            std::ifstream ifs;
            ifs.open(filepath);
            if(ifs)
            {
                while(!ifs.eof())
                {
                    string_t entry;
                    ifs >> entry;
                    if(ifs.eof())
                        break;
                    if(!is_numeric(entry))
                    {
                        numeric_file = false;
                        break;
                    }
                }
            }
            ifs.close();
            if(numeric_file)
                attributes["type"] = "numeric/double";
            else
                attributes["type"] = "text/string";
        }
    }

    std::stringstream ss;
    ss << "<DartMeasurementFile";
    for(const auto& itr : attributes)
        ss << " " << attribute_string(itr.first, itr.second);
    // name=\"" << name << "\ type=\"" << type << "\">"
    ss << ">" << filepath << "</DartMeasurementFile>";
    std::cout << ss.str() << std::endl;
}

//======================================================================================//

}  // namespace plotting
}  // namespace tim
