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
#include "timemory/types.hpp"

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
void
plot(const std::string& _label, const std::string& _prefix, const std::string& _dir,
     bool _echo_dart, const std::string& _json_file) TIMEMORY_VISIBILITY("default");
//
void
echo_dart_file(const string_t& filepath, attributes_t attributes)
    TIMEMORY_VISIBILITY("default");
//
template <typename... Types>
std::enable_if_t<(sizeof...(Types) > 0), void>
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
template <typename... Types>
std::enable_if_t<(sizeof...(Types) > 0), void>
plot(const string_t& _prefix, const string_t& _dir, bool _echo_dart,
     const string_t& _json_file)
{
    impl::plot<Types...>::generate(_prefix, _dir, _echo_dart, _json_file);
}
//
//--------------------------------------------------------------------------------------//
//
inline void
echo_dart_file(const string_t& filepath, attributes_t attributes)
{
    auto attribute_string = [](const string_t& key, const string_t& item) {
        return operation::join("", key, "=", "\"", item, "\"");
    };

    auto lowercase = [](string_t _str) {
        for(auto& itr : _str)
            itr = tolower(itr);
        return _str;
    };

    auto contains = [&lowercase](const string_t& str, const std::set<string_t>& items) {
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
        if(name.find('/') != string_t::npos)
            name = name.substr(name.find_last_of('/') + 1);
        if(name.find('\\') != string_t::npos)
            name = name.substr(name.find_last_of('\\') + 1);
        if(name.find('.') != string_t::npos)
            name.erase(name.find_last_of('.'));
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
