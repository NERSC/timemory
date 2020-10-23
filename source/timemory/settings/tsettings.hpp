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

#include "timemory/settings/types.hpp"
#include "timemory/settings/vsettings.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/types.hpp"

#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

namespace tim
{
//
/// \struct tim::tsettings
/// \brief Implements a specific setting
template <typename Tp, typename Vp>
struct tsettings : public vsettings
{
    using type          = Tp;
    using value_type    = Vp;
    using base_type     = vsettings;
    using parser_t      = base_type::parser_t;
    using parser_func_t = base_type::parser_func_t;

    template <typename Up = Vp, enable_if_t<!std::is_reference<Up>::value> = 0>
    tsettings()
    : base_type()
    , m_value(Tp{})
    {}

    template <typename... Args>
    tsettings(Vp _value, Args&&... _args)
    : base_type(std::forward<Args>(_args)...)
    , m_value(_value)
    {
        m_type_index  = std::type_index(typeid(type));
        m_value_index = std::type_index(typeid(value_type));
    }

    Tp&       get() { return m_value; }
    const Tp& get() const { return m_value; }
    void      set(const Tp& _value) { m_value = _value; }
    // void      set(Tp&& _value) { m_value = std::forward<Tp>(_value); }

    virtual std::string as_string() const override
    {
        std::stringstream ss;
        ss << std::boolalpha;
        ss << m_value;
        return ss.str();
    }
    virtual void parse() final { m_value = get_env<decay_t<Tp>>(m_env_name, m_value); }
    virtual void parse(const std::string& v) final
    {
        m_value = get_value<decay_t<Tp>>(v);
    }

    virtual void add_argument(argparse::argument_parser& p) final
    {
        if(!m_cmdline.empty())
        {
            if(std::is_same<Tp, bool>::value)
                m_max_count = 1;
            p.add_argument(m_cmdline, m_description)
                .action(get_action())
                .count(m_count)
                .max_count(m_max_count)
                .choices(m_choices);
        }
    }

    virtual std::shared_ptr<vsettings> clone() final
    {
        return std::make_shared<tsettings<Tp>>(m_value, m_name, m_env_name, m_description,
                                               m_cmdline, m_count, m_max_count);
    }

    virtual void clone(std::shared_ptr<vsettings> rhs) final
    {
        vsettings::clone(rhs);
        if(dynamic_cast<tsettings<Tp>*>(rhs.get()))
            set(dynamic_cast<tsettings<Tp>*>(rhs.get())->get());
        else if(dynamic_cast<tsettings<Tp, Tp&>*>(rhs.get()))
            set(dynamic_cast<tsettings<Tp, Tp&>*>(rhs.get())->get());
    }

    virtual display_map_t get_display(std::ios::fmtflags fmt = {}, int _w = -1,
                                      int _p = -1) override
    {
        auto _data   = vsettings::get_display(fmt, _w, _p);
        auto _as_str = [&](auto _val) {
            std::stringstream _ss;
            _ss.setf(fmt);
            if(_w > -1)
                _ss << std::setw(_w);
            if(_p > -1)
                _ss << std::setprecision(_p);
            _ss << std::boolalpha << _val;
            return _ss.str();
        };
        _data["value"] = _as_str(m_value);
        _data["type"]  = _as_str(demangle<Tp>());
        return _data;
    }

    template <typename Archive, typename Up = Vp,
              enable_if_t<!std::is_reference<Up>::value> = 0>
    void save(Archive& ar, const unsigned int) const
    {
        ar(cereal::make_nvp("name", m_name));
        ar(cereal::make_nvp("environ", m_env_name));
        ar(cereal::make_nvp("description", m_description));
        ar(cereal::make_nvp("count", m_count));
        ar(cereal::make_nvp("max_count", m_max_count));
        ar(cereal::make_nvp("cmdline", m_cmdline));
        ar(cereal::make_nvp("value", m_value));
    }

    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        try
        {
            ar(cereal::make_nvp("name", m_name));
            ar(cereal::make_nvp("environ", m_env_name));
            ar(cereal::make_nvp("description", m_description));
            ar(cereal::make_nvp("count", m_count));
            ar(cereal::make_nvp("max_count", m_max_count));
            ar(cereal::make_nvp("cmdline", m_cmdline));
        } catch(...)
        {}
        ar(cereal::make_nvp("value", m_value));
    }

    template <typename Up = decay_t<Tp>>
    enable_if_t<std::is_same<Up, bool>::value, Up> get_value(const std::string& val)
    {
        if(!val.empty())
        {
            namespace regex_const             = std::regex_constants;
            const auto        regex_constants = regex_const::egrep | regex_const::icase;
            const std::string pattern         = "^(off|false|no|n|f|0)$";
            bool              _match          = false;
            try
            {
                _match = std::regex_match(val, std::regex(pattern, regex_constants));
            } catch(std::bad_cast&)
            {
                auto _val = val;
                for(auto& itr : _val)
                    itr = tolower(itr);
                for(const auto& itr : { "off", "false", "no", "n", "f", "0" })
                {
                    if(_val == itr)
                    {
                        _match = true;
                        break;
                    }
                }
            }
            if(_match)
                return false;
            else
                return true;
        }
        return true;
    }

    template <typename Up = decay_t<Tp>>
    enable_if_t<!std::is_same<Up, bool>::value && !std::is_same<Up, std::string>::value,
                Up>
    get_value(const std::string& str)
    {
        std::stringstream ss;
        ss << str;
        Up val{};
        ss >> val;
        return val;
    }

    template <typename Up = decay_t<Tp>>
    enable_if_t<std::is_same<Up, std::string>::value, Up> get_value(
        const std::string& val)
    {
        return val;
    }

    virtual parser_func_t get_action(project::timemory) override { return get_action(); }

private:
    template <typename Up = decay_t<Tp>, enable_if_t<std::is_same<Up, bool>::value> = 0>
    auto get_action()
    {
        return [&](parser_t& p) {
            std::string id  = m_cmdline.back();
            auto        pos = m_cmdline.back().find_first_not_of('-');
            if(pos != std::string::npos)
                id = id.substr(pos);
            auto val = p.get<std::string>(id);
            if(val.empty())
                m_value = true;
            else
            {
                namespace regex_const      = std::regex_constants;
                const auto regex_constants = regex_const::egrep | regex_const::icase;
                const std::string pattern  = "^(off|false|no|n|f|0)$";
                bool              _match   = false;
                try
                {
                    _match = std::regex_match(val, std::regex(pattern, regex_constants));
                } catch(std::bad_cast&)
                {
                    for(auto& itr : val)
                        itr = tolower(itr);
                    for(const auto& itr : { "off", "false", "no", "n", "f", "0" })
                    {
                        if(val == itr)
                        {
                            _match = true;
                            break;
                        }
                    }
                }
                if(_match)
                    m_value = false;
                else
                    m_value = true;
            }
        };
    }

    template <typename Up                                        = decay_t<Tp>,
              enable_if_t<!std::is_same<Up, bool>::value &&
                          !std::is_same<Up, std::string>::value> = 0>
    auto get_action()
    {
        return [&](parser_t& p) {
            std::string id  = m_cmdline.back();
            auto        pos = m_cmdline.back().find_first_not_of('-');
            if(pos != std::string::npos)
                id = id.substr(pos);
            m_value = p.get<Up>(id);
        };
    }

    template <typename Up                                       = decay_t<Tp>,
              enable_if_t<std::is_same<Up, std::string>::value> = 0>
    auto get_action()
    {
        return [&](parser_t& p) {
            std::string id  = m_cmdline.back();
            auto        pos = m_cmdline.back().find_first_not_of('-');
            if(pos != std::string::npos)
                id = id.substr(pos);
            auto _vec = p.get<std::vector<std::string>>(id);
            if(_vec.empty())
                m_value = "";
            else
            {
                std::stringstream ss;
                for(auto& itr : _vec)
                    ss << ", " << itr;
                m_value = ss.str().substr(2);
            }
        };
    }

private:
    using base_type::m_count;
    using base_type::m_description;
    using base_type::m_env_name;
    using base_type::m_max_count;
    using base_type::m_name;
    using base_type::m_type_index;
    value_type m_value;
};
//
}  // namespace tim

namespace cereal
{
template <typename Archive, typename Tp>
void
save(Archive& ar, std::shared_ptr<tim::tsettings<Tp, Tp&>> obj)
{
    auto _obj = obj->clone();
    ar(_obj);
}
}  // namespace cereal
