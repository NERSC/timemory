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

#include "timemory/settings/vsettings.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/settings/types.hpp"

#include <algorithm>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tim
{
//
TIMEMORY_SETTINGS_INLINE
vsettings::vsettings(std::string _name, std::string _env_name, std::string _descript,
                     std::vector<std::string> _cmdline, int32_t _count,
                     int32_t _max_count, std::vector<std::string> _choices)
: m_count(_count)
, m_max_count(_max_count)
, m_name(std::move(_name))
, m_env_name(std::move(_env_name))
, m_description(std::move(_descript))
, m_cmdline(std::move(_cmdline))
, m_choices(std::move(_choices))
{}
//
TIMEMORY_SETTINGS_LINKAGE(vsettings::display_map_t)
vsettings::get_display(std::ios::fmtflags fmt, int _w, int _p)
{
    display_map_t _data;
    auto          _as_str = [&](auto _val) {
        std::stringstream _ss;
        _ss.setf(fmt);
        if(_w > -1)
            _ss << std::setw(_w);
        if(_p > -1)
            _ss << std::setprecision(_p);
        _ss << std::boolalpha << _val;
        return _ss.str();
    };

    auto _arr_as_str = [&](auto _val) -> std::string {
        if(_val.empty())
            return "";
        std::stringstream _ss;
        for(size_t i = 0; i < _val.size(); ++i)
            _ss << ", " << _as_str(_val.at(i));
        return _ss.str().substr(2);
    };

    _data["name"]         = _as_str(m_name);
    _data["count"]        = _as_str(m_count);
    _data["max_count"]    = _as_str(m_max_count);
    _data["env_name"]     = _as_str(m_env_name);
    _data["description"]  = _as_str(m_description);
    _data["command_line"] = _arr_as_str(m_cmdline);
    _data["choices"]      = _arr_as_str(m_choices);
    return _data;
}
//
TIMEMORY_SETTINGS_LINKAGE(bool)
vsettings::matches(const std::string& inp, bool exact) const
{
    // an exact match to name or env-name should always be checked
    if(inp == m_env_name || inp == m_name)
        return true;

    if(exact)
    {
        // match the command-line option w/ or w/o leading dashes
        auto _cmd_line_exact = [&](const std::string& itr) {
            // don't match short-options
            if(itr.length() == 2)
                return false;
            auto _with_dash = (itr == inp);
            auto _pos       = itr.find_first_not_of('-');
            if(_with_dash || _pos == std::string::npos)
                return _with_dash;
            return (itr.substr(_pos) == inp);
        };

        return std::any_of(m_cmdline.begin(), m_cmdline.end(), _cmd_line_exact);
    }
    else
    {
        const auto       cre = std::regex_constants::icase;
        const std::regex re(inp, cre);

        if(std::regex_search(m_env_name, re) || std::regex_search(m_name, re))
            return true;

        auto _cmd_line_regex = [&](const std::string& itr) {
            // don't match short-options
            if(itr.length() == 2)
                return false;
            return std::regex_search(itr, re);
        };

        return std::any_of(m_cmdline.begin(), m_cmdline.end(), _cmd_line_regex);
    }
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SETTINGS_LINKAGE(void)
vsettings::clone(std::shared_ptr<vsettings> rhs)
{
    m_name        = rhs->m_name;
    m_env_name    = rhs->m_env_name;
    m_description = rhs->m_description;
    m_cmdline     = rhs->m_cmdline;
    m_count       = rhs->m_count;
    m_max_count   = rhs->m_max_count;
}
//
}  // namespace tim
