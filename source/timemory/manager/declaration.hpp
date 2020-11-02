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

/**
 * \file timemory/manager/declaration.hpp
 * \brief The declaration for the types for manager without definitions
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/manager/macros.hpp"
#include "timemory/manager/types.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/tpls/cereal/cereal.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              manager
//
//--------------------------------------------------------------------------------------//
//
class manager
{
public:
    using this_type        = manager;
    using pointer_t        = std::shared_ptr<this_type>;
    using pointer_pair_t   = std::pair<pointer_t, pointer_t>;
    using size_type        = std::size_t;
    using string_t         = std::string;
    using comm_group_t     = std::tuple<mpi::comm_t, int32_t>;
    using mutex_t          = std::mutex;
    using auto_lock_t      = std::unique_lock<mutex_t>;
    using auto_lock_ptr_t  = std::shared_ptr<std::unique_lock<mutex_t>>;
    using finalizer_func_t = std::function<void()>;
    using finalizer_pair_t = std::pair<std::string, finalizer_func_t>;
    using finalizer_list_t = std::deque<finalizer_pair_t>;
    using finalizer_void_t = std::multimap<void*, finalizer_func_t>;
    using settings_ptr_t   = std::shared_ptr<settings>;
    using filemap_t        = std::map<string_t, std::map<string_t, std::set<string_t>>>;
    using enum_set_t       = std::set<TIMEMORY_COMPONENT>;
    template <typename Tp>
    using enum_map_t = std::map<TIMEMORY_COMPONENT, Tp>;

public:
    // Constructor and Destructors
    manager();
    ~manager();

    manager(const manager&) = delete;
    manager(manager&&)      = delete;

    manager& operator=(const manager&) = delete;
    manager& operator=(manager&&) = delete;

    // storage-types add functors to destroy the instances
    template <typename Func>
    void add_cleanup(void*, Func&&);
    template <typename Func>
    void add_cleanup(const std::string&, Func&&);
    template <typename StackFunc, typename FinalFunc>
    void add_finalizer(const std::string&, StackFunc&&, FinalFunc&&, bool);
    void remove_cleanup(const std::string&);
    void remove_finalizer(const std::string&);
    void cleanup(const std::string&);
    void cleanup();
    void finalize();
    void read_command_line();
    bool is_finalized() { return m_is_finalized; }

    void add_file_output(const string_t& _category, const string_t& _label,
                         const string_t& _file);
    void add_text_output(const string_t& _label, const string_t& _file);
    void add_json_output(const string_t& _label, const string_t& _file);

    /// \fn set_write_metadata
    /// \brief Set to 0 for yes if other output, -1 for never, or 1 for yes
    void    set_write_metadata(short v) { m_write_metadata = v; }
    void    write_metadata(const char* = "");
    void    update_metadata_prefix();
    int32_t get_rank() const { return m_rank; }
    bool    is_finalizing() const { return m_is_finalizing; }
    void    is_finalizing(bool v) { m_is_finalizing = v; }
    void    add_entries(uint64_t n) { m_num_entries += n; }

public:
    // Public static functions
    static pointer_t instance() TIMEMORY_VISIBILITY("default");
    static pointer_t master_instance() TIMEMORY_VISIBILITY("default");
    static int32_t   total_instance_count() { return f_manager_instance_count().load(); }
    static void      use_exit_hook(bool val) { f_use_exit_hook() = val; }
    static void      exit_hook();
    static int32_t   get_thread_count() { return f_thread_counter().load(); }

private:
    template <typename Tp>
    void do_init_storage();

    template <typename Tp>
    void do_print_storage(const enum_set_t& = {});

    template <typename Tp>
    void do_clear(const enum_set_t& = {});

    template <typename Arch, typename Tp>
    void do_serialize(Arch& ar, const enum_set_t& = {});

    template <typename Tp>
    void do_size(uint64_t& _sz);  // total size for all components

    template <typename Tp>
    void do_size(enum_map_t<uint64_t>& _sz);  // size for individual components

    //----------------------------------------------------------------------------------//
    // used to expand a tuple in settings
    //
    template <typename... Types>
    struct filtered_get_storage
    {
        static void        initialize(pointer_t _manager = {});
        static std::string serialize(pointer_t _manager = {}, const enum_set_t& = {});
        static void        clear(pointer_t _manager = {}, const enum_set_t& = {});
        static void        print(pointer_t _manager = {}, const enum_set_t& = {});
        static uint64_t    size(pointer_t _manager = {});
        static enum_map_t<uint64_t> size(pointer_t _manager, const enum_set_t&);

        static std::string serialize(const enum_set_t& _types)
        {
            return serialize(pointer_t{ nullptr }, _types);
        }

        static void clear(const enum_set_t& _types)
        {
            clear(pointer_t{ nullptr }, _types);
        }

        static void print(const enum_set_t& _types)
        {
            print(pointer_t{ nullptr }, _types);
        }

        static enum_map_t<uint64_t> size(const enum_set_t& _types)
        {
            return size(pointer_t{ nullptr }, _types);
        }
    };

    //----------------------------------------------------------------------------------//
    //
    template <template <typename...> class Tuple, typename... Types>
    struct filtered_get_storage<Tuple<Types...>> : public filtered_get_storage<Types...>
    {
        using base_type = filtered_get_storage<Types...>;
        using base_type::clear;
        using base_type::initialize;
        using base_type::print;
        using base_type::serialize;
        using base_type::size;
    };

public:
    //----------------------------------------------------------------------------------//
    //
    template <typename... Types>
    struct get_storage : public filtered_get_storage<implemented_t<Types...>>
    {
        using base_type = filtered_get_storage<implemented_t<Types...>>;
        using base_type::clear;
        using base_type::initialize;
        using base_type::print;
        using base_type::serialize;
        using base_type::size;
    };

    //----------------------------------------------------------------------------------//
    //
    template <template <typename...> class Tuple, typename... Types>
    struct get_storage<Tuple<Types...>>
    : public filtered_get_storage<implemented_t<Types...>>
    {
        using base_type = filtered_get_storage<implemented_t<Types...>>;
        using base_type::clear;
        using base_type::initialize;
        using base_type::print;
        using base_type::serialize;
        using base_type::size;
    };

    //----------------------------------------------------------------------------------//
    //
    /// used by storage classes to ensure that the singleton instance is managed
    /// via the master thread of holding the manager instance
    //
    template <typename Tp>
    auto get_singleton() -> decltype(Tp::instance())
    {
        return Tp::instance();
    }

private:
    template <typename... Types>
    friend struct get_storage;

    template <typename... Types>
    friend struct filtered_get_storage;

public:
    // Public member functions
    int32_t instance_count() const { return m_instance_count; }
    int64_t get_tid() const { return m_thread_index; }

protected:
    // protected static functions
    static comm_group_t get_communicator_group();

protected:
    // protected functions
    string_t get_prefix() const;

private:
    /// notifies that it is finalizing
    bool            m_is_finalizing   = false;
    bool            m_is_finalized    = false;
    short           m_write_metadata  = 0;
    int32_t         m_instance_count  = 0;
    int32_t         m_rank            = 0;
    uint64_t        m_num_entries     = 0;
    int64_t         m_thread_index    = threading::get_id();
    std::thread::id m_thread_id       = threading::get_tid();
    string_t        m_metadata_prefix = "";
    mutex_t         m_mutex;
    auto_lock_ptr_t m_lock = auto_lock_ptr_t{ nullptr };
    /// increment the shared_ptr count here to ensure these instances live
    /// for the entire lifetime of the manager instance
    graph_hash_map_ptr_t   m_hash_ids           = get_hash_ids();
    graph_hash_alias_ptr_t m_hash_aliases       = get_hash_aliases();
    finalizer_list_t       m_finalizer_cleanups = {};
    finalizer_list_t       m_master_cleanup     = {};
    finalizer_list_t       m_worker_cleanup     = {};
    finalizer_list_t       m_master_finalizers  = {};
    finalizer_list_t       m_worker_finalizers  = {};
    finalizer_void_t       m_pointer_fini       = {};
    filemap_t              m_output_files       = {};
    settings_ptr_t         m_settings = settings::shared_instance<TIMEMORY_API>();

private:
    struct persistent_data
    {
        persistent_data()  = default;
        ~persistent_data() = default;

        persistent_data(const persistent_data&) = delete;
        persistent_data(persistent_data&&)      = delete;
        persistent_data& operator=(const persistent_data&) = delete;
        persistent_data& operator=(persistent_data&&) = delete;

        std::atomic<int32_t>      instance_count{ 0 };
        std::atomic<int32_t>      thread_count{ 0 };
        bool                      use_exit_hook = true;
        pointer_t                 master_instance;
        bool&                     debug   = settings::debug();
        int&                      verbose = settings::verbose();
        std::shared_ptr<settings> config  = settings::shared_instance<TIMEMORY_API>();
    };

    /// single instance of all the global static data
    static persistent_data& f_manager_persistent_data();
    /// number of timing manager instances
    static std::atomic<int32_t>& f_manager_instance_count()
    {
        return f_manager_persistent_data().instance_count;
    }
    /// num-threads based on number of managers created
    static std::atomic<int32_t>& f_thread_counter()
    {
        return f_manager_persistent_data().thread_count;
    }
    /// suppresses the exit hook during termination
    static bool& f_use_exit_hook() { return f_manager_persistent_data().use_exit_hook; }
    static auto& f_debug() { return f_manager_persistent_data().debug; }
    static auto& f_verbose() { return f_manager_persistent_data().verbose; }
    static auto  f_settings() { return f_manager_persistent_data().config; }

public:
    static void set_persistent_master(pointer_t _pinst)
    {
        tim::manager::f_manager_persistent_data().master_instance = std::move(_pinst);
    }

    static void update_settings(const settings& _settings)
    {
        f_settings().reset(new settings(_settings));
    }

    static settings swap_settings(settings _settings)
    {
        settings _tmp         = std::move(*(f_settings().get()));
        *(f_settings().get()) = std::move(_settings);
        return _tmp;
    }
};
//
//----------------------------------------------------------------------------------//
//
template <typename Func>
void
manager::add_cleanup(void* _key, Func&& _func)
{
    // insert into map
    m_pointer_fini.insert({ _key, std::forward<Func>(_func) });
}
//
//----------------------------------------------------------------------------------//
//
template <typename Func>
void
manager::add_cleanup(const std::string& _key, Func&& _func)
{
    // ensure there are no duplicates
    remove_cleanup(_key);
    // insert into map
    auto _entry = finalizer_pair_t{ _key, std::forward<Func>(_func) };
    m_finalizer_cleanups.push_back(_entry);
}
//
//----------------------------------------------------------------------------------//
//
template <typename StackFunc, typename FinalFunc>
void
manager::add_finalizer(const std::string& _key, StackFunc&& _stack_func,
                       FinalFunc&& _inst_func, bool _is_master)
{
    // ensure there are no duplicates
    remove_finalizer(_key);

    m_metadata_prefix = settings::get_global_output_prefix(true);

    if(m_write_metadata == 0)
        m_write_metadata = 1;

    auto _stack_entry = finalizer_pair_t{ _key, std::forward<StackFunc>(_stack_func) };
    auto _final_entry = finalizer_pair_t{ _key, std::forward<FinalFunc>(_inst_func) };

    if(_is_master)
    {
        m_master_cleanup.push_back(_stack_entry);
        m_master_finalizers.push_back(_final_entry);
    }
    else
    {
        m_worker_cleanup.push_back(_stack_entry);
        m_worker_finalizers.push_back(_final_entry);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_init_storage()
{
    using storage_type = typename Tp::storage_type;

    static thread_local auto tmp = [&]() {
        if(!component::state<Tp>::has_storage())
        {
            auto ret = storage_type::instance();
            if(ret)
                ret->initialize();

            if(f_debug())
                printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
                       demangle<Tp>().c_str(), (void*) ret,
                       (component::state<Tp>::has_storage()) ? "true" : "false",
                       (ret) ? ((ret->empty()) ? "true" : "false") : "false");
        }
        return true;
    }();
    tim::consume_parameters(tmp);
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_print_storage(const enum_set_t& _types)
{
    using storage_type = typename Tp::storage_type;

    if(!_types.empty() && _types.count(component::properties<Tp>{}()) == 0)
        return;

    auto ret = storage_type::noninit_instance();
    if(ret && !ret->empty())
        ret->print();

    if(f_debug())
        printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
               demangle<Tp>().c_str(), (void*) ret,
               (component::state<Tp>::has_storage()) ? "true" : "false",
               (ret) ? ((ret->empty()) ? "true" : "false") : "false");
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_clear(const enum_set_t& _types)
{
    using storage_type = typename Tp::storage_type;

    if(!_types.empty() && _types.count(component::properties<Tp>{}()) == 0)
        return;

    auto ret = storage_type::noninit_instance();
    if(ret)
        ret->reset();

    if(f_debug())
        printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
               demangle<Tp>().c_str(), (void*) ret,
               (component::state<Tp>::has_storage()) ? "true" : "false",
               (ret) ? ((ret->empty()) ? "true" : "false") : "false");
}
//
//----------------------------------------------------------------------------------//
//
template <typename Archive, typename Tp>
void
manager::do_serialize(Archive& ar, const enum_set_t& _types)
{
    using storage_type = typename Tp::storage_type;

    if(!_types.empty() && _types.count(component::properties<Tp>{}()) == 0)
        return;

    auto ret = storage_type::noninit_instance();
    if(ret && !ret->empty())
        ret->do_serialize(ar);

    if(f_debug())
        printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
               demangle<Tp>().c_str(), (void*) ret,
               (component::state<Tp>::has_storage()) ? "true" : "false",
               (ret) ? ((ret->empty()) ? "true" : "false") : "false");
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_size(uint64_t& _sz)
{
    auto label         = tim::demangle<Tp>();
    using storage_type = typename Tp::storage_type;
    label += std::string(" (") + tim::demangle<storage_type>() + ")";

    auto ret = storage_type::noninit_instance();
    if(ret && !ret->empty())
        _sz += ret->size();

    if(f_debug())
        printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
               demangle<Tp>().c_str(), (void*) ret,
               (component::state<Tp>::has_storage()) ? "true" : "false",
               (ret) ? ((ret->empty()) ? "true" : "false") : "false");
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
void
manager::do_size(enum_map_t<uint64_t>& _sz)
{
    using storage_type = typename Tp::storage_type;

    auto itr = _sz.find(component::properties<Tp>{}());
    if(itr == _sz.end())
        return;

    auto ret = storage_type::noninit_instance();
    if(ret && !ret->empty())
        itr->second = ret->true_size();
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
void
manager::filtered_get_storage<Types...>::initialize(pointer_t _manager)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return;
    TIMEMORY_FOLD_EXPRESSION(_manager->do_init_storage<Types>());
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
std::string
manager::filtered_get_storage<Types...>::serialize(pointer_t         _manager,
                                                   const enum_set_t& _types)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return "";
    std::stringstream ss;
    {
        using archive_type = trait::output_archive_t<manager>;
        using policy_type  = policy::output_archive_t<manager>;
        auto oa            = policy_type::get(ss);
        oa->setNextName("timemory");
        oa->startNode();
        {
            oa->setNextName("ranks");
            oa->startNode();
            oa->makeArray();
            TIMEMORY_FOLD_EXPRESSION(
                _manager->do_serialize<archive_type, Types>(*oa, _types));
            oa->finishNode();
        }
        oa->finishNode();
    }
    return ss.str();
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
void
manager::filtered_get_storage<Types...>::clear(pointer_t         _manager,
                                               const enum_set_t& _types)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return;
    TIMEMORY_FOLD_EXPRESSION(_manager->do_clear<Types>(_types));
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
void
manager::filtered_get_storage<Types...>::print(pointer_t         _manager,
                                               const enum_set_t& _types)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return;
    TIMEMORY_FOLD_EXPRESSION(_manager->do_print_storage<Types>(_types));
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
uint64_t
manager::filtered_get_storage<Types...>::size(pointer_t _manager)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();
    if(!_manager)
        return 0;
    uint64_t _sz = 0;
    TIMEMORY_FOLD_EXPRESSION(_manager->do_size<Types>(_sz));
    return _sz;
}
//
//----------------------------------------------------------------------------------//
//
template <typename... Types>
manager::enum_map_t<uint64_t>
manager::filtered_get_storage<Types...>::size(pointer_t         _manager,
                                              const enum_set_t& _types)
{
    if(_manager.get() == nullptr)
        _manager = manager::instance();

    enum_map_t<uint64_t> _sz{};
    if(_manager)
    {
        for(auto& itr : _types)
            _sz.insert({ itr, 0 });
        TIMEMORY_FOLD_EXPRESSION(_manager->do_size<Types>(_sz));
    }
    return _sz;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//

#if !defined(_TIMEMORY_INTEL)
CEREAL_CLASS_VERSION(tim::manager, 0)
#endif
