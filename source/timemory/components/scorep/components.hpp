//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/**
 * \file timemory/components/scorep/components.hpp
 * \brief Implementation of the scorep component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/scorep/backends.hpp"
#include "timemory/components/scorep/types.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
struct scorep : public base<scorep, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = scorep;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "scorep"; }
    static std::string description()
    {
        return "Forwards markers to Score-P instrumentation "
               "(via SCOREP_USER_REGION_BEGIN and SCOREP_USER_REGION_END)";
    }
    static value_type record() {}

    scorep(const std::string& _prefix = "")
    : m_prefix(_prefix){};

    void start()
    {
#if defined(TIMEMORY_USE_SCOREP)
        // set the region type
        auto region_type = (tim::settings::timeline_profile())
                               ? SCOREP_USER_REGION_TYPE_DYNAMIC
                               : SCOREP_USER_REGION_TYPE_COMMON;
        // begin marker
        SCOREP_USER_REGION_BEGIN(get_handle(m_prefix), m_prefix.c_str(), region_type)
#endif
    }

    void stop()
    {
#if defined(TIMEMORY_USE_SCOREP)
        // end marker
        SCOREP_USER_REGION_END(get_handle(m_prefix))
#endif
    }

    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }

    //----------------------------------------------------------------------------------//
    //
    // Member Variables
    //
    //----------------------------------------------------------------------------------//
private:
    std::string m_prefix = "";

#if defined(TIMEMORY_USE_SCOREP)
private:
    static handle_type& get_handle(const std::string& _key)
    {
        using uomap_t = std::unordered_map<std::string, handle_type>;
        static thread_local uomap_t handle_map{};
        if(handle_map.find(_key) == handle_map.end())
            handle_map.insert({ _key, SCOREP_USER_INVALID_REGION });
        return handle_map[_key];
    }
#endif
};
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
