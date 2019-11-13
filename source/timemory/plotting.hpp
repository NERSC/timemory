// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include "timemory/bits/components.hpp"
#include "timemory/components.hpp"
#include "timemory/settings.hpp"
#include "timemory/variadic/macros.hpp"

#include <cstdio>
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
_plot(const string_t& _prefix, const string_t& _dir = "", bool echo_dart = true)
{
    if(!component::properties<_Tp>::has_storage())
        return;

    if(std::system(nullptr) && tim::settings::json_output() &&
       !tim::trait::external_output_handling<_Tp>::value)
    {
        auto label    = _Tp::label();
        auto descript = _Tp::description();
        auto jname    = tim::settings::compose_output_filename(label, ".json");
        {
            std::ifstream ifs(jname.c_str());
            bool          exists = ifs.good();
            ifs.close();
            if(!exists)
                return;
        }
        auto odir = (_dir == "") ? settings::output_path() : _dir;
        auto cmd =
            TIMEMORY_JOIN(" ", settings::python_exe(), "-m", "timemory.plotting", "-f",
                          jname, "-t", "\"" + _prefix, descript + "\"", "-o", odir);
        if(echo_dart)
            cmd += " -e";
        int ret = std::system(cmd.c_str());
        if(ret != 0)
        {
            auto msg =
                TIMEMORY_JOIN("", "Error generating plots with command: '", cmd, "'");
            fprintf(stderr, "[%s]> %s\n", TIMEMORY_LABEL(""), msg.c_str());
        }
    }
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
          typename std::enable_if<(sizeof...(_Types) > 0), int>::type = 0>
inline void
plot(_Args&&... _args)
{
    impl::plot<_Types...>::generate(std::forward<_Args>(_args)...);
}

//======================================================================================//

template <typename... _Types, typename... _Args,
          typename std::enable_if<(sizeof...(_Types) == 0), int>::type = 0>
inline void
plot(_Args&&... _args)
{
    impl::plot<complete_tuple_t>::generate(std::forward<_Args>(_args)...);
}

//======================================================================================//

inline void
echo_dart(const string_t& filepath, attributes_t attributes)
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
