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

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/types.hpp"

#include <atomic>
#include <memory>
#include <string>

namespace tim
{
//
class manager;
struct settings;
//
namespace base
{
class storage
{
public:
    using string_t  = std::string;
    using this_type = storage;

public:
    storage(bool _is_master, int64_t _instance_id, std::string _label);
    virtual ~storage();

    explicit storage(const this_type&) = delete;
    explicit storage(this_type&&)      = delete;
    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

    virtual void print() {}
    virtual void cleanup() {}
    virtual void stack_clear() {}
    virtual void disable() {}
    virtual void initialize() {}
    virtual void finalize() {}
    virtual bool global_init() { return false; }
    virtual bool thread_init() { return false; }
    virtual bool data_init() { return false; }

    template <typename Tp>
    static this_type* base_instance();

public:
    const graph_hash_map_ptr_t&   get_hash_ids() const { return m_hash_ids; }
    const graph_hash_alias_ptr_t& get_hash_aliases() const { return m_hash_aliases; }

    hash_value_type add_hash_id(const std::string& _prefix);
    void            add_hash_id(uint64_t _lhs, uint64_t _rhs);

    bool    is_initialized() const { return m_initialized; }
    int64_t instance_id() const { return m_instance_id; }
    void    free_shared_manager();

protected:
    void add_file_output(const string_t& _category, const string_t& _label,
                         const string_t& _file);
    void add_text_output(const string_t& _label, const string_t& _file);
    void add_json_output(const string_t& _label, const string_t& _file);

    static std::atomic<int>& storage_once_flag();
    static void              stop_profiler();

protected:
    bool                      m_initialized  = false;                      // NOLINT
    bool                      m_finalized    = false;                      // NOLINT
    bool                      m_global_init  = false;                      // NOLINT
    bool                      m_thread_init  = false;                      // NOLINT
    bool                      m_data_init    = false;                      // NOLINT
    bool                      m_is_master    = false;                      // NOLINT
    bool                      m_node_init    = dmp::is_initialized();      // NOLINT
    int32_t                   m_node_rank    = dmp::rank();                // NOLINT
    int32_t                   m_node_size    = dmp::size();                // NOLINT
    int64_t                   m_instance_id  = -1;                         // NOLINT
    int64_t                   m_thread_idx   = threading::get_id();        // NOLINT
    std::thread::id           m_thread_tid   = threading::get_tid();       // NOLINT
    string_t                  m_label        = "";                         // NOLINT
    graph_hash_map_ptr_t      m_hash_ids     = ::tim::get_hash_ids();      // NOLINT
    graph_hash_alias_ptr_t    m_hash_aliases = ::tim::get_hash_aliases();  // NOLINT
    std::shared_ptr<manager>  m_manager      = {};                         // NOLINT
    std::shared_ptr<settings> m_settings     = {};                         // NOLINT
};
//
template <typename Tp>
inline storage*
storage::base_instance()
{
    using storage_type = tim::storage<Tp>;

    // thread-local variable
    static thread_local base::storage* _ret = nullptr;

    // return nullptr is disabled
    if(!trait::runtime_enabled<Tp>::get())
        return nullptr;

    // if nullptr, try to get instance
    if(_ret == nullptr)
    {
        // thread will copy the hash-table so use a lock here
        auto_lock_t _lk{ type_mutex<base::storage>() };
        _ret = static_cast<base::storage*>(storage_type::instance());
    }

    // return pointer
    return _ret;
}
//
}  // namespace base
}  // namespace tim
