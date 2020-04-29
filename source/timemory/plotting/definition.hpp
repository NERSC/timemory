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
 * \file timemory/plotting/definition.hpp
 * \brief The definitions for the types in plotting
 */

#pragma once

#include "timemory/plotting/declaration.hpp"
#include "timemory/plotting/macros.hpp"
#include "timemory/plotting/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/types.hpp"

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
#if !(defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_PLOTTING_EXTERN)) ||          \
    defined(TIMEMORY_PLOTTING_SOURCE)

//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_PLOTTING_LINKAGE(void)
plot(string_t _label, string_t _prefix, const string_t& _dir, bool _echo_dart,
     string_t _json_file)
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

    auto _file = _json_file;
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

    auto cmd = operation::join(" ", settings::python_exe(), "-m", "timemory.plotting",
                               "-f", _file, "-t", "\"" + _prefix, "\"", "-o", _dir);

    if(_echo_dart)
        cmd += " -e";

    if(settings::verbose() > 2 || settings::debug())
        PRINT_HERE("PLOT COMMAND: '%s'", cmd.c_str());

    set_env<std::string>("TIMEMORY_BANNER", "OFF");
    set_env<std::string>("TIMEMORY_CXX_PLOT_MODE", "1", 1);
    launch_process(cmd.c_str(), _info + " plot generation failed");
    set_env<std::string>("TIMEMORY_CXX_PLOT_MODE", "0", 1);
}
//
//--------------------------------------------------------------------------------------//
//
#endif  // !defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_PLOTTING_SOURCE)
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
plot(string_t _prefix, const string_t& _dir, bool _echo_dart, string_t _json_file)
{
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

    if(_prefix.empty())
        _prefix = Tp::get_description();

    if(_json_file.empty())
        _json_file = settings::compose_output_filename(Tp::get_label(), ".json");

    auto _label = demangle<Tp>();

    plot(_label, _prefix, _dir, _echo_dart, _json_file);
}
//
//--------------------------------------------------------------------------------------//
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
//
//--------------------------------------------------------------------------------------//
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
//--------------------------------------------------------------------------------------//
//
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
    ss << ">" << filepath << "</DartMeasurementFile>";
    std::cout << ss.str() << std::endl;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace plotting
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
