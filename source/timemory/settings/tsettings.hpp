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

#include "timemory/environment/declaration.hpp"
#include "timemory/settings/types.hpp"
#include "timemory/settings/vsettings.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/argparse.hpp"
#include "timemory/utility/types.hpp"

#include <cstdlib>
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
private:
    template <typename Up>
    static constexpr bool is_bool_type()
    {
        return std::is_same<decay_t<Up>, bool>::value;
    }

    template <typename Up>
    static constexpr bool is_string_type()
    {
        return std::is_same<decay_t<Up>, std::string>::value;
    }

    template <typename Up>
    static constexpr bool is_else_type()
    {
        return !is_bool_type<Up>() && !is_string_type<Up>();
    }

public:
    using type          = Tp;
    using value_type    = Vp;
    using base_type     = vsettings;
    using parser_t      = base_type::parser_t;
    using parser_func_t = base_type::parser_func_t;
    using vpointer_t    = std::shared_ptr<vsettings>;

public:
    template <typename Up = Vp, enable_if_t<!std::is_reference<Up>::value> = 0>
    tsettings();

    template <typename... Args>
    tsettings(Vp, Args&&...);

    ~tsettings() {}

    tsettings(const tsettings&) = default;
    tsettings(tsettings&&)      = default;
    tsettings& operator=(const tsettings&) = default;
    tsettings& operator=(tsettings&&) = default;

public:
    Tp&         get();
    const Tp&   get() const;
    Tp          get_value(const std::string& val) const;
    std::string as_string() const override;

    void set(Tp);
    void parse() final;
    void parse(const std::string& v) final;
    void clone(std::shared_ptr<vsettings> rhs) final;
    void add_argument(argparse::argument_parser& p) final;

    vpointer_t    clone() final;
    parser_func_t get_action(TIMEMORY_API) override;
    display_map_t get_display(std::ios::fmtflags fmt = {}, int _w = -1,
                              int _p = -1) override;

    template <typename Archive, typename Up = Vp>
    void save(Archive& ar, const unsigned int,
              enable_if_t<!std::is_reference<Up>::value, int> = 0) const;

    template <typename Archive>
    void load(Archive& ar, const unsigned int);

private:
    template <typename Up>
    auto get_action(enable_if_t<is_bool_type<Up>(), int> = 0)
        TIMEMORY_VISIBILITY("hidden");

    template <typename Up>
    auto get_action(enable_if_t<is_string_type<Up>(), long> = 0)
        TIMEMORY_VISIBILITY("hidden");

    template <typename Up>
    auto get_action(enable_if_t<is_else_type<Up>(), long long> = 0)
        TIMEMORY_VISIBILITY("hidden");

    template <typename Up>
    Up get_value(const std::string& val, enable_if_t<is_bool_type<Up>(), int> = 0) const
        TIMEMORY_VISIBILITY("hidden");

    template <typename Up>
    Up get_value(const std::string& val,
                 enable_if_t<is_string_type<Up>(), long> = 0) const
        TIMEMORY_VISIBILITY("hidden");

    template <typename Up>
    Up get_value(const std::string& str,
                 enable_if_t<is_else_type<Up>(), long long> = 0) const
        TIMEMORY_VISIBILITY("hidden");

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
template <typename Tp, typename Vp>
template <typename Up, enable_if_t<!std::is_reference<Up>::value, int>>
tsettings<Tp, Vp>::tsettings()
: base_type{}
, m_value{ Tp{} }
{}
//
template <typename Tp, typename Vp>
template <typename... Args>
tsettings<Tp, Vp>::tsettings(Vp _value, Args&&... _args)  // NOLINT
: base_type{ std::forward<Args>(_args)... }
, m_value{ _value }  // NOLINT
{
    m_type_index  = std::type_index(typeid(type));
    m_value_index = std::type_index(typeid(value_type));
}
//
template <typename Tp, typename Vp>
Tp&
tsettings<Tp, Vp>::get()
{
    return m_value;
}
//
template <typename Tp, typename Vp>
const Tp&
tsettings<Tp, Vp>::get() const
{
    return m_value;
}
//
template <typename Tp, typename Vp>
Tp
tsettings<Tp, Vp>::get_value(const std::string& val) const
{
    return get_value<Tp>(val);
}
//
template <typename Tp, typename Vp>
void
tsettings<Tp, Vp>::set(Tp _value)
{
    m_value = std::move(_value);
}
//
template <typename Tp, typename Vp>
std::string
tsettings<Tp, Vp>::as_string() const
{
    std::stringstream ss;
    ss << std::boolalpha;
    ss << m_value;
    return ss.str();
}
//
template <typename Tp, typename Vp>
void
tsettings<Tp, Vp>::parse()
{
    if(!m_env_name.empty())
    {
        char* c_env_val = std::getenv(m_env_name.c_str());
        if(c_env_val)
            parse(std::string{ c_env_val });
    }
}
//
template <typename Tp, typename Vp>
void
tsettings<Tp, Vp>::parse(const std::string& v)
{
    m_value = get_value<decay_t<Tp>>(v);
}
//
template <typename Tp, typename Vp>
void
tsettings<Tp, Vp>::add_argument(argparse::argument_parser& p)
{
    if(!m_cmdline.empty())
    {
        if(std::is_same<Tp, bool>::value)
            m_max_count = 1;
        p.add_argument(m_cmdline, m_description)
            .action(get_action(TIMEMORY_API{}))
            .count(m_count)
            .max_count(m_max_count)
            .choices(m_choices);
    }
}
//
template <typename Tp, typename Vp>
void
tsettings<Tp, Vp>::clone(std::shared_ptr<vsettings> rhs)
{
    vsettings::clone(rhs);
    if(dynamic_cast<tsettings<Tp>*>(rhs.get()))
    {
        set(dynamic_cast<tsettings<Tp>*>(rhs.get())->get());
    }
    else if(dynamic_cast<tsettings<Tp, Tp&>*>(rhs.get()))
    {
        set(dynamic_cast<tsettings<Tp, Tp&>*>(rhs.get())->get());
    }
}
//
template <typename Tp, typename Vp>
std::shared_ptr<vsettings>
tsettings<Tp, Vp>::clone()
{
    return std::make_shared<tsettings<Tp>>(m_value, m_name, m_env_name, m_description,
                                           m_cmdline, m_count, m_max_count);
}
//
template <typename Tp, typename Vp>
typename tsettings<Tp, Vp>::parser_func_t tsettings<Tp, Vp>::get_action(TIMEMORY_API)
{
    return get_action<Tp>();
}
//
template <typename Tp, typename Vp>
typename tsettings<Tp, Vp>::display_map_t
tsettings<Tp, Vp>::get_display(std::ios::fmtflags fmt, int _w, int _p)
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
//
template <typename Tp, typename Vp>
template <typename Archive, typename Up>
void
tsettings<Tp, Vp>::save(Archive& ar, const unsigned int,
                        enable_if_t<!std::is_reference<Up>::value, int>) const
{
    std::string _dtype =
        (std::is_same<Tp, std::string>::value) ? "string" : demangle<Tp>();
    ar(cereal::make_nvp("name", m_name));
    ar(cereal::make_nvp("environ", m_env_name));
    ar(cereal::make_nvp("description", m_description));
    ar(cereal::make_nvp("count", m_count));
    ar(cereal::make_nvp("max_count", m_max_count));
    ar(cereal::make_nvp("cmdline", m_cmdline));
    ar(cereal::make_nvp("data_type", _dtype));
    ar(cereal::make_nvp("value", m_value));
}
//
template <typename Tp, typename Vp>
template <typename Archive>
void
tsettings<Tp, Vp>::load(Archive& ar, const unsigned int)
{
    try
    {
        std::string _dtype{};
        ar(cereal::make_nvp("name", m_name));
        ar(cereal::make_nvp("environ", m_env_name));
        ar(cereal::make_nvp("description", m_description));
        ar(cereal::make_nvp("count", m_count));
        ar(cereal::make_nvp("max_count", m_max_count));
        ar(cereal::make_nvp("cmdline", m_cmdline));
        ar(cereal::make_nvp("data_type", _dtype));
    } catch(...)
    {}
    ar(cereal::make_nvp("value", m_value));
}
//
template <typename Tp, typename Vp>
template <typename Up>
auto
tsettings<Tp, Vp>::get_action(enable_if_t<is_bool_type<Up>(), int>)
{
    return [&](parser_t& p) {
        std::string id  = m_cmdline.back();
        auto        pos = m_cmdline.back().find_first_not_of('-');
        if(pos != std::string::npos)
            id = id.substr(pos);
        auto val = p.get<std::string>(id);
        if(val.empty())
        {
            m_value = true;
        }
        else
        {
            m_value = get_bool(val, true);
        }
    };
}
//
template <typename Tp, typename Vp>
template <typename Up>
auto
tsettings<Tp, Vp>::get_action(enable_if_t<is_string_type<Up>(), long>)
{
    return [&](parser_t& p) {
        std::string id  = m_cmdline.back();
        auto        pos = m_cmdline.back().find_first_not_of('-');
        if(pos != std::string::npos)
            id = id.substr(pos);
        auto _vec = p.get<std::vector<std::string>>(id);
        if(_vec.empty())
        {
            m_value = "";
        }
        else
        {
            std::stringstream ss;
            for(auto& itr : _vec)
                ss << ", " << itr;
            m_value = ss.str().substr(2);
        }
    };
}
//
template <typename Tp, typename Vp>
template <typename Up>
auto
tsettings<Tp, Vp>::get_action(enable_if_t<is_else_type<Up>(), long long>)
{
    return [&](parser_t& p) {
        std::string id  = m_cmdline.back();
        auto        pos = m_cmdline.back().find_first_not_of('-');
        if(pos != std::string::npos)
            id = id.substr(pos);
        m_value = p.get<decay_t<Up>>(id);
    };
}
//
template <typename Tp, typename Vp>
template <typename Up>
Up
tsettings<Tp, Vp>::get_value(const std::string& val,
                             enable_if_t<is_bool_type<Up>(), int>) const
{
    if(!val.empty())
        return get_bool(val, true);
    return true;
}
//
template <typename Tp, typename Vp>
template <typename Up>
Up
tsettings<Tp, Vp>::get_value(const std::string& val,
                             enable_if_t<is_string_type<Up>(), long>) const
{
    return val;
}
//
template <typename Tp, typename Vp>
template <typename Up>
Up
tsettings<Tp, Vp>::get_value(const std::string& str,
                             enable_if_t<is_else_type<Up>(), long long>) const
{
    std::stringstream ss;
    ss << str;
    Up val{};
    ss >> val;
    return val;
}
//
}  // namespace tim

namespace tim
{
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
}  // namespace tim
