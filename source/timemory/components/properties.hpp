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
#include "timemory/mpl/concepts.hpp"

#include <cstring>
#include <regex>
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
using idset_t = std::set<std::string>;
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
template <typename Tp, bool PlaceHolder = concepts::is_placeholder<Tp>::value>
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
//
//--------------------------------------------------------------------------------------//
//
//  non-placeholder types
//
template <typename Tp>
struct static_properties<Tp, false>
{
    using ptype = properties<Tp>;

    static bool matches(int _idx) { return (_idx == ptype{}()); }
    static bool matches(const std::string& _key) { return matches(_key.c_str()); }
    static bool matches(const char* _key)
    {
        // don't allow checks for placeholder types
        static_assert(!concepts::is_placeholder<Tp>::value,
                      "static_properties is instantiating a placeholder type");

        const auto regex_consts = std::regex_constants::ECMAScript |
                                  std::regex_constants::icase |
                                  std::regex_constants::optimize;

        auto get_pattern = [](const std::string& _option) {
            return std::string("^(.*[,;: \t\n\r]+|)") + _option + "([,;: \t\n\r]+.*|$)";
        };
        if(std::regex_match(_key,
                            std::regex(get_pattern(ptype::enum_string()), regex_consts)))
            return true;
        for(const auto& itr : ptype::ids())
        {
            if(std::regex_match(_key, std::regex(get_pattern(itr), regex_consts)))
                return true;
        }
        return false;
    }
};
//
//  placeholder types
//
template <typename Tp>
struct static_properties<Tp, true>
{
    static bool matches(int) { return false; }
    static bool matches(const char*) { return false; }
    static bool matches(const std::string&) { return false; }
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
    template <typename Archive>
    void serialize(Archive&, const unsigned int)
    {}
    TIMEMORY_COMPONENT operator()() { return TIMEMORY_COMPONENTS_END; }
    //
    constexpr operator TIMEMORY_COMPONENT() const { return TIMEMORY_COMPONENTS_END; }
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
template <int Idx>
using enumerator_t = typename enumerator<Idx>::type;
//
}  // namespace component
}  // namespace tim
