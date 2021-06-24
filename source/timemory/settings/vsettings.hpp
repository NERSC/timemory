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
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/environment/types.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/utility/types.hpp"

#include <cstdint>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

namespace tim
{
//
namespace argparse
{
struct argument_parser;
}
//
/// \struct tim::vsettings
/// \brief Virtual base class for storing settings
struct vsettings
{
    using parser_t      = argparse::argument_parser;
    using parser_func_t = std::function<void(parser_t&)>;
    using display_map_t = std::map<std::string, std::string>;

    struct noparse
    {};

    vsettings(std::string _name = "", std::string _env_name = "",
              std::string _descript = "", std::vector<std::string> _cmdline = {},
              int32_t _count = -1, int32_t _max_count = -1,
              std::vector<std::string> _choices = {});

    virtual ~vsettings() = default;

    vsettings(const vsettings&) = default;
    vsettings(vsettings&&)      = default;

    vsettings& operator=(const vsettings&) = default;
    vsettings& operator=(vsettings&&) = default;

    virtual std::string                as_string() const                        = 0;
    virtual void                       reset()                                  = 0;
    virtual void                       parse()                                  = 0;
    virtual void                       parse(const std::string&)                = 0;
    virtual void                       add_argument(argparse::argument_parser&) = 0;
    virtual std::shared_ptr<vsettings> clone()                                  = 0;
    virtual void                       clone(std::shared_ptr<vsettings> rhs);

    virtual display_map_t get_display(std::ios::fmtflags fmt = {}, int _w = -1,
                                      int _p = -1);

    const auto& get_name() const { return m_name; }
    const auto& get_env_name() const { return m_env_name; }
    const auto& get_description() const { return m_description; }
    const auto& get_command_line() const { return m_cmdline; }
    const auto& get_choices() const { return m_choices; }
    const auto& get_count() const { return m_count; }
    const auto& get_max_count() const { return m_max_count; }

    void set_count(int32_t v) { m_count = v; }
    void set_max_count(int32_t v) { m_max_count = v; }
    void set_choices(const std::vector<std::string>& v) { m_choices = v; }
    void set_command_line(const std::vector<std::string>& v) { m_cmdline = v; }

    auto get_type_index() const { return m_type_index; }
    auto get_value_index() const { return m_value_index; }

    virtual bool matches(const std::string&, bool exact = true) const;

    template <typename Tp>
    std::pair<bool, Tp> get() const;

    template <typename Tp>
    bool get(Tp& _val) const;

    template <typename Tp, enable_if_t<std::is_fundamental<decay_t<Tp>>::value> = 0>
    bool set(const Tp& _val);
    void set(const std::string& _val) { parse(_val); }

    virtual parser_func_t get_action(TIMEMORY_API) = 0;

    template <typename Tp>
    static auto cast(std::shared_ptr<vsettings>& _val)
    {
        return static_cast<tsettings<decay_t<Tp>, Tp>*>(_val.get());
    }

    template <typename Tp>
    static auto cast(const std::shared_ptr<vsettings>& _val)
    {
        return static_cast<const tsettings<decay_t<Tp>, Tp>*>(_val.get());
    }

protected:
    static int get_debug()
    {
        static bool _bool_val = get_env("TIMEMORY_DEBUG_SETTINGS", false);
        static int  _int_val  = get_env("TIMEMORY_DEBUG_SETTINGS", 0);
        return (_bool_val) ? _int_val : 0;
    }

    template <typename Tp>
    void report_change(Tp _old, const Tp& _new);

protected:
    std::type_index          m_type_index  = std::type_index(typeid(void));  // NOLINT
    std::type_index          m_value_index = std::type_index(typeid(void));  // NOLINT
    int32_t                  m_count       = -1;                             // NOLINT
    int32_t                  m_max_count   = -1;                             // NOLINT
    std::string              m_name        = "";                             // NOLINT
    std::string              m_env_name    = "";                             // NOLINT
    std::string              m_description = "";                             // NOLINT
    std::vector<std::string> m_cmdline     = {};                             // NOLINT
    std::vector<std::string> m_choices     = {};                             // NOLINT
};
//
template <typename Tp>
std::pair<bool, Tp>
vsettings::get() const
{
    auto _ref = dynamic_cast<const tsettings<Tp, Tp&>*>(this);
    if(_ref)
    {
        return { true, _ref->get() };
    }

    auto _nref = dynamic_cast<const tsettings<Tp, Tp>*>(this);
    if(_nref)
        return { true, _nref->get() };

    return { false, Tp{} };
}
//
template <typename Tp>
bool
vsettings::get(Tp& _val) const
{
    auto&& _ret = this->get<Tp>();
    if(_ret.first)
        _val = _ret.second;
    return _ret.first;
}
//
template <typename Tp, enable_if_t<std::is_fundamental<decay_t<Tp>>::value>>
bool
vsettings::set(const Tp& _val)
{
    auto _ref = dynamic_cast<tsettings<Tp, Tp&>*>(this);
    if(_ref)
    {
        _ref->set(_val);
        return true;
    }

    auto _nref = dynamic_cast<tsettings<Tp, Tp>*>(this);
    if(_nref)
    {
        _nref->set(_val);
        return true;
    }

    return false;
}
//
template <typename Tp>
void
vsettings::report_change(Tp _old, const Tp& _new)
{
    if(get_debug() < 1)
        return;

    if(_old != _new)
    {
        std::ostringstream oss;
        oss << std::boolalpha;
        oss << "[timemory::settings] " << m_name << " (env: " << m_env_name
            << ") changed: " << _old << " --> " << _new << "\n";
        if(get_debug() > 1)
        {
            print_demangled_backtrace<6, 3>(oss);
        }
        std::cerr << oss.str() << std::flush;
    }
}
//
}  // namespace tim

#if defined(TIMEMORY_SETTINGS_HEADER_MODE)
#    include "timemory/settings/vsettings.cpp"
#endif
