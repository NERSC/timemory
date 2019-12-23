// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file base_storage.hpp
 * \headerfile base_storage.hpp "timemory/utility/base_storage.hpp"
 * Storage of the call-graph for each component. Each component has a thread-local
 * singleton that holds the call-graph. When a worker thread is deleted, it merges
 * itself back into the master thread storage. When the master thread is deleted,
 * it handles I/O (i.e. text file output, JSON output, stdout output). This class
 * needs some refactoring for clarity and probably some non-templated inheritance
 * for common features because this is probably the most convoluted piece of
 * development in the entire repository.
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/gperf.hpp"
#include "timemory/general/hash.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"

//--------------------------------------------------------------------------------------//

#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

//--------------------------------------------------------------------------------------//

namespace tim
{
class manager;

//--------------------------------------------------------------------------------------//

namespace base
{
//======================================================================================//
//
//              Storage class for types that implement it
//
//======================================================================================//

class storage
{
public:
    //----------------------------------------------------------------------------------//
    //
    using string_t  = std::string;
    using this_type = storage;

public:
    //----------------------------------------------------------------------------------//
    //
    storage(bool _is_master, int64_t _instance_id, const std::string& _label)
    : m_initialized(false)
    , m_finalized(false)
    , m_global_init(false)
    , m_thread_init(false)
    , m_data_init(false)
    , m_is_master(_is_master)
    , m_node_init(dmp::is_initialized())
    , m_node_rank(dmp::rank())
    , m_node_size(dmp::size())
    , m_instance_id(_instance_id)
    , m_label(_label)
    , m_hash_ids(::tim::get_hash_ids())
    , m_hash_aliases(::tim::get_hash_aliases())
    {
        if(m_is_master && m_instance_id > 0)
        {
            int _id = m_instance_id;
            PRINT_HERE("%s: %i (%s)",
                       "Warning! base::storage is master but is not zero instance", _id,
                       m_label.c_str());
        }

        if(!m_is_master && m_instance_id == 0)
        {
            int _id = m_instance_id;
            PRINT_HERE("%s: %i (%s)",
                       "Warning! base::storage is not master but is zero instance", _id,
                       m_label.c_str());
        }

        if(settings::debug())
            PRINT_HERE("%s: %i (%s)", "base::storage instance created",
                       (int) m_instance_id, m_label.c_str());
    }

    //----------------------------------------------------------------------------------//
    //
    virtual ~storage()
    {
        if(settings::debug())
            PRINT_HERE("%s: %i (%s)", "base::storage instance deleted",
                       (int) m_instance_id, m_label.c_str());
    }

    //----------------------------------------------------------------------------------//
    //
    explicit storage(const this_type&) = delete;
    explicit storage(this_type&&)      = delete;
    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

    virtual void print()       = 0;
    virtual void cleanup()     = 0;
    virtual void stack_clear() = 0;
    virtual void initialize()  = 0;
    virtual void finalize()    = 0;

public:
    const graph_hash_map_ptr_t&   get_hash_ids() const { return m_hash_ids; }
    const graph_hash_alias_ptr_t& get_hash_aliases() const { return m_hash_aliases; }

    hash_result_type add_hash_id(const std::string& _prefix)
    {
        return ::tim::add_hash_id(m_hash_ids, _prefix);
    }

    void add_hash_id(uint64_t _lhs, uint64_t _rhs)
    {
        ::tim::add_hash_id(m_hash_ids, m_hash_aliases, _lhs, _rhs);
    }

    bool is_initialized() const { return m_initialized; }

    int64_t instance_id() const { return m_instance_id; }

    void free_shared_manager();

protected:
    void add_text_output(const string_t& _label, const string_t& _file);
    void add_json_output(const string_t& _label, const string_t& _file);

    static std::atomic<int>& storage_once_flag()
    {
        static std::atomic<int> _instance(0);
        return _instance;
    }

    static void stop_profiler()
    {
        // disable gperf if profiling
#if defined(TIMEMORY_USE_GPERF) || defined(TIMEMORY_USE_GPERF_CPU_PROFILER) ||           \
    defined(TIMEMORY_USE_GPERF_HEAP_PROFILER)
        try
        {
            if(storage_once_flag()++ == 0)
                gperf::profiler_stop();
        } catch(std::exception& e)
        {
            std::cerr << "Error calling gperf::profiler_stop(): " << e.what()
                      << ". Continuing..." << std::endl;
        }
#endif
    }

protected:
    bool                     m_initialized  = false;
    bool                     m_finalized    = false;
    bool                     m_global_init  = false;
    bool                     m_thread_init  = false;
    bool                     m_data_init    = false;
    bool                     m_is_master    = false;
    bool                     m_node_init    = dmp::is_initialized();
    int32_t                  m_node_rank    = dmp::rank();
    int32_t                  m_node_size    = dmp::size();
    int64_t                  m_instance_id  = -1;
    string_t                 m_label        = "";
    graph_hash_map_ptr_t     m_hash_ids     = ::tim::get_hash_ids();
    graph_hash_alias_ptr_t   m_hash_aliases = ::tim::get_hash_aliases();
    std::shared_ptr<manager> m_manager;
};

//======================================================================================//

}  // namespace base
}  // namespace tim

//======================================================================================//
