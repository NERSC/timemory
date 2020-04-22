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
 * \file timemory/components/craypat/components.hpp
 * \brief Implementation of the craypat component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
//
#include "timemory/components/craypat/backends.hpp"
#include "timemory/components/craypat/types.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
struct craypat_record
: base<craypat_record, void>
, policy::instance_tracker<craypat_record>
{
    using this_type    = craypat_record;
    using value_type   = void;
    using base_type    = base<this_type, void>;
    using tracker_type = policy::instance_tracker<this_type>;

    static std::string label() { return "craypat_record"; }
    static std::string description()
    {
        return "Toggles CrayPAT recording on calling thread";
    }

    static void global_init(storage_type*) { backend::craypat::record(PAT_STATE_OFF); }

    static void global_finalize(storage_type*)
    {
        backend::craypat::record(PAT_STATE_OFF);
    }

    void start()
    {
        tracker_type::start();
        if(tracker_type::m_thr == 0)
            backend::craypat::record(PAT_STATE_ON);
    }

    void stop()
    {
        tracker_type::stop();
        if(tracker_type::m_thr == 0)
            backend::craypat::record(PAT_STATE_OFF);
    }
};
//
//--------------------------------------------------------------------------------------//
//
struct craypat_region
: base<craypat_region, void>
, policy::instance_tracker<craypat_region, false>
{
    using tracker_type = policy::instance_tracker<craypat_region, false>;

    static std::string label() { return "craypat_region"; }
    static std::string description() { return "Adds region labels to CrayPAT output"; }

    void start()
    {
        if(m_label == nullptr)
            return;
        tracker_type::start();
        backend::craypat::region_begin(tracker_type::m_tot, m_label);
    }

    void stop()
    {
        if(m_label == nullptr)
            return;
        backend::craypat::region_end(tracker_type::m_tot);
        tracker_type::stop();
    }

    void set_prefix(const char* _label) { m_label = _label; }

private:
    const char* m_label = nullptr;
};
//
//--------------------------------------------------------------------------------------//
//
struct craypat_counters : base<craypat_counters, std::vector<unsigned long>>
{
    using value_type   = std::vector<unsigned long>;
    using this_type    = craypat_counters;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using strvector_t  = std::vector<std::string>;

    static std::string label() { return "craypat_counters"; }

    static std::string description()
    {
        return "Returns the names and current count value of any counter events that "
               "have been set to count on the hardware category";
    }

    static void configure()
    {
        std::set<int> _empty;
        int           _idx = 0;
        for(auto& citr : get_persistent_data().m_categories)
        {
            int _category = std::get<0>(citr);
            // temporary data
            const char**   _names   = nullptr;
            unsigned long* _values  = nullptr;
            int*           _nevents = nullptr;
            // get data from craypat
            backend::craypat::counters(_category, _names, _values, _nevents);
            if(_names && _values && _nevents)
            {
                std::get<1>(citr) = *_nevents;
                for(int i = 0; i < *_nevents; ++i)
                {
                    std::get<2>(citr).push_back(_names[i]);
                    get_persistent_data().m_labels.push_back(_names[i]);
                }
            }
            else
            {
                _empty.insert(_idx);
            }
            ++_idx;
        }

        get_persistent_data().m_events = get_persistent_data().m_labels.size();

        // erase the unused categories
        for(auto ritr = _empty.rbegin(); ritr != _empty.rend(); ++ritr)
        {
            auto itr = get_persistent_data().m_categories.begin();
            std::advance(itr, *ritr);
            get_persistent_data().m_categories.erase(itr);
        }
    }

    static void global_init(storage_type*) { configure(); }

    static value_type record()
    {
        value_type _data;
        for(const auto& citr : get_persistent_data().m_categories)
        {
            int _category = std::get<0>(citr);
            // temporary data
            const char**   _names   = nullptr;
            unsigned long* _values  = nullptr;
            int*           _nevents = nullptr;
            // get data from craypat
            backend::craypat::counters(_category, _names, _values, _nevents);
            if(_names && _values && _nevents)
            {
                // current beginning index
                auto _off = _data.size();
                // make sure data size is consistent
                _data.resize(_off + std::get<1>(citr), 0);
                // compute the size of loop
                int _n = std::min(std::get<1>(citr), *_nevents);
                for(int i = 0; i < _n; ++i)
                    _data[_off + i] = _values[i];
            }
        }
        return _data;
    }

    static strvector_t label_array() { return get_persistent_data().m_labels; }

    static strvector_t description_array()
    {
        return strvector_t(get_persistent_data().m_events, "");
    }

    static std::vector<unsigned long> unit_array()
    {
        return std::vector<unsigned long>(get_persistent_data().m_events, 1);
    }

    static strvector_t display_unit_array()
    {
        return strvector_t(get_persistent_data().m_events, "");
    }

    value_type get() const { return base_type::load(); }
    value_type get_display() const { return base_type::load(); }

    void start() { value = record(); }

    void stop()
    {
        value = (record() - value);
        accum += value;
    }

private:
    struct persistent_data
    {
        using category_tuple_t  = std::tuple<int, int, strvector_t>;
        using category_vector_t = std::vector<category_tuple_t>;
        category_vector_t m_categories =
            category_vector_t({ category_tuple_t{ PAT_CTRS_CPU, 0, strvector_t{} },
                                category_tuple_t{ PAT_CTRS_NETWORK, 0, strvector_t{} },
                                category_tuple_t{ PAT_CTRS_ACCEL, 0, strvector_t{} },
                                category_tuple_t{ PAT_CTRS_RAPL, 0, strvector_t{} },
                                category_tuple_t{ PAT_CTRS_PM, 0, strvector_t{} },
                                category_tuple_t{ PAT_CTRS_UNCORE, 0, strvector_t{} } });
        strvector_t m_labels = {};
        size_t      m_events = 0;
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
struct craypat_heap_stats : base<craypat_heap_stats, void>
{
    static std::string label() { return "craypat_heap_stats"; }
    static std::string description() { return "Undocumented by 'pat_api.h'"; }

    void start() {}
    void stop() { backend::craypat::heap_stats(); }
};
//
//--------------------------------------------------------------------------------------//
//
struct craypat_flush_buffer : base<craypat_flush_buffer, unsigned long>
{
    using value_type = unsigned long;
    using this_type  = craypat_flush_buffer;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "craypat_flush_buffer"; }
    static std::string description()
    {
        return "Writes all the recorded contents in the data buffer to the experiment "
               "data file for the calling PE and calling thread. Returns the number of "
               "bytes flushed";
    }

    static value_type record()
    {
        unsigned long _nbytes;
        backend::craypat::flush_buffer(&_nbytes);
        return _nbytes;
    }

    double get() const { return m_nbytes; }
    auto   get_display() const { return get(); }
    void   start() {}
    void   stop()
    {
        value = record();
        accum += record();
    }

private:
    unsigned long m_nbytes = 0;
};
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
