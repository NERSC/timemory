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
 * \file timemory/components/caliper/components.hpp
 * \brief Implementation of the caliper component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/caliper/backends.hpp"
#include "timemory/components/caliper/types.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
struct caliper : public base<caliper, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = caliper;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "caliper"; }
    static std::string description() { return "Caliper instrumentation markers"; }
    static value_type  record() {}

    static void global_init(storage_type*) { backend::cali::init(); }

    caliper(const std::string& _channel = get_channel(),
            const int& _attributes = get_attributes(), const std::string& _prefix = "")
    : channel(_channel)
    , attributes(_attributes)
    , id(backend::cali::create_attribute(_channel, CALI_TYPE_STRING, _attributes))
    , prefix(_prefix)
    {}

    void start() { backend::cali::begin(id, prefix.c_str()); }
    void stop() { backend::cali::end(id); }

    void set_prefix(const std::string& _prefix) { prefix = _prefix; }

    //----------------------------------------------------------------------------------//
    //
    // Custom functions
    //
    //----------------------------------------------------------------------------------//
    using attributes_t = int;
    static std::string  get_default_channel() { return "timemory"; }
    static std::string& get_channel()
    {
        static std::string _instance = get_default_channel();
        return _instance;
    }
    static attributes_t get_default_attributes()
    {
        return (CALI_ATTR_NESTED | CALI_ATTR_SCOPE_THREAD);
    }
    static attributes_t& get_attributes()
    {
        static attributes_t _instance = get_default_attributes();
        return _instance;
    }
    static void enable_process_scope()
    {
        get_attributes() = (CALI_ATTR_NESTED | CALI_ATTR_SCOPE_PROCESS);
    }
    static void enable_thread_scope()
    {
        get_attributes() = (CALI_ATTR_NESTED | CALI_ATTR_SCOPE_THREAD);
    }
    static void enable_task_scope()
    {
        get_attributes() = (CALI_ATTR_NESTED | CALI_ATTR_SCOPE_TASK);
    }

    //----------------------------------------------------------------------------------//
    //
    // Member Variables
    //
    //----------------------------------------------------------------------------------//
private:
    std::string         channel    = get_channel();
    int                 attributes = get_attributes();
    backend::cali::id_t id =
        backend::cali::create_attribute(channel, CALI_TYPE_STRING, attributes);
    std::string prefix = "";
};

}  // namespace component
}  // namespace tim
//
//======================================================================================//
