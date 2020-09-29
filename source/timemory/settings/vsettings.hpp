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

#include "timemory/api.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/types.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace tim
{
/// \struct tim::base::vsettings
/// \brief Base class for storing settings
struct vsettings
{
    using parser_t      = argparse::argument_parser;
    using display_map_t = std::map<std::string, std::string>;

    vsettings(const std::string& _name = "", const std::string& _env_name = "",
              const std::string& _descript = "", std::vector<std::string> _cmdline = {},
              int32_t _count = -1, int32_t _max_count = -1)
    : m_count(_count)
    , m_max_count(_max_count)
    , m_name(_name)
    , m_env_name(_env_name)
    , m_description(_descript)
    , m_cmdline(_cmdline)
    {}

    virtual ~vsettings() = default;

    virtual void                       parse()                                  = 0;
    virtual void                       parse(const std::string&)                = 0;
    virtual void                       add_argument(argparse::argument_parser&) = 0;
    virtual std::shared_ptr<vsettings> clone()                                  = 0;
    virtual void                       clone(std::shared_ptr<vsettings> rhs);

    virtual display_map_t get_display(std::ios::fmtflags fmt = {}, int _w = -1,
                                      int _p = -1)
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

        _data["name"]        = _as_str(m_name);
        _data["count"]       = _as_str(m_count);
        _data["max_count"]   = _as_str(m_max_count);
        _data["env_name"]    = _as_str(m_env_name);
        _data["description"] = _as_str(m_description);
        return _data;
    }

    const auto& get_name() const { return m_name; }
    const auto& get_env_name() const { return m_env_name; }
    const auto& get_description() const { return m_description; }
    const auto& get_command_line() const { return m_cmdline; }
    const auto& get_count() const { return m_count; }
    const auto& get_max_count() const { return m_max_count; }

    void set_count(int32_t v) { m_count = v; }
    void set_max_count(int32_t v) { m_max_count = v; }

    auto get_type_index() const { return m_type_index; }
    auto get_value_index() const { return m_value_index; }

    virtual bool matches(const std::string&) const;

protected:
    std::type_index          m_type_index  = std::type_index(typeid(void));
    std::type_index          m_value_index = std::type_index(typeid(void));
    int32_t                  m_count       = -1;
    int32_t                  m_max_count   = -1;
    std::string              m_name        = "";
    std::string              m_env_name    = "";
    std::string              m_description = "";
    std::vector<std::string> m_cmdline     = {};
};
//
}  // namespace tim

#if defined(TIMEMORY_SETTINGS_HEADER_MODE)
#    include "timemory/settings/vsettings.cpp"
#endif
