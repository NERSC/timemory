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

/** \file component/properties.hpp
 * \headerfile component/properties.hpp "timemory/component/properties.hpp"
 * Provides properties for components
 *
 */

#pragma once

#include "timemory/enum.h"

#include <cstring>
#include <set>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

namespace tim
{
namespace component
{
//
template <typename... Types>
struct placeholder;
//
using idset_t = std::initializer_list<std::string>;
//
struct nothing;
//
struct opaque;
//
namespace factory
{
//
template <typename Toolset, typename Arg, typename... Args>
opaque
get_opaque(Arg&& arg, Args&&... args);
//
template <typename Toolset>
opaque
get_opaque();
//
template <typename Toolset>
opaque
get_opaque(bool);
//
template <typename Toolset>
std::set<size_t>
get_typeids();
//
}  // namespace factory
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct static_properties;
//
template <typename Tp>
struct properties;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct state
{
    static bool& has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};
//--------------------------------------------------------------------------------------//
//
/*
struct dynamic_properties
{
    using properties_map_t = std::unordered_map<std::type_index, dynamic_properties*>;

    dynamic_properties(int _idx, const char* _estr, const char* _id, const idset_t& _ids)
    : m_idx(_idx)
    , m_enum_string(_estr)
    , m_id(_id)
    , m_ids(_ids)
    {}

    int         get_value() const { return m_idx; }
    const char* get_enum_string() const { return m_enum_string.c_str(); }
    const char* get_id() const { return m_id.c_str(); }
    idset_t     get_ids() const { return m_ids; }

    bool operator==(int _idx) { return _idx == m_idx; }
    bool operator==(const char* _key)
    {
        return (strcmp(_key, m_enum_string.c_str()) == 0 ||
                m_ids.find(_key) != m_ids.end());
    }
    bool operator==(const std::string& _key)
    {
        return (m_enum_string == _key || m_ids.find(_key) != m_ids.end());
    }

    static const properties_map_t& get_properties_map()
    {
        return get_private_properties_map();
    }

protected:
    static properties_map_t& get_private_properties_map()
    {
        static properties_map_t _instance{};
        return _instance;
    }

protected:
    int         m_idx;
    std::string m_enum_string;
    std::string m_id;
    idset_t     m_ids;
};*/
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct static_properties
// : dynamic_properties
{
    static_properties();

    static constexpr bool matches(int _idx) { return (_idx == properties<Tp>::value); }

    static bool matches(const std::string& _key)
    {
        return (strcmp(_key.c_str(), properties<Tp>::enum_string()) == 0 ||
                properties<Tp>::ids().find(_key) != properties<Tp>::ids().end());
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct properties : static_properties<Tp>
{
    using type                                = Tp;
    using value_type                          = TIMEMORY_COMPONENT;
    static constexpr TIMEMORY_COMPONENT value = TIMEMORY_COMPONENTS_END;
    static constexpr const char* enum_string() { return "TIMEMORY_COMPONENTS_END"; }
    static constexpr const char* id() { return ""; }
    static idset_t               ids() { return idset_t{}; }
};
//
//--------------------------------------------------------------------------------------//
//
template <int Idx>
struct enumerator : properties<placeholder<nothing>>
{
    using type                  = placeholder<nothing>;
    static constexpr bool value = false;

    bool operator==(int) const { return false; }
    bool operator==(const char*) const { return false; }
    bool operator==(const std::string&) const { return false; }
};
//
//--------------------------------------------------------------------------------------//
//
/*
template <typename Tp>
static_properties<Tp>::static_properties()
: dynamic_properties(properties<Tp>::value, properties<Tp>::enum_string(),
                     properties<Tp>::id(), properties<Tp>::ids())
{
    get_private_properties_map().insert({ std::type_index(typeid(Tp)), this });
}*/
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
static_properties<Tp>::static_properties()
{}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
