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

/** \file timemory/manager.hpp
 * \headerfile timemory/manager.hpp "timemory/manager.hpp"
 * Static singleton handler that is not templated. In general, this is the
 * first object created and last object destroyed. It should be utilized to
 * store type-independent data
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/backends/mpi.hpp"
#include "timemory/data/base_storage.hpp"
#include "timemory/general/hash.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/utility.hpp"

//--------------------------------------------------------------------------------------//

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

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

template <typename... Types>
class component_tuple;

//--------------------------------------------------------------------------------------//

class tim_api manager
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
    using finalizer_func_t = std::function<void()>;
    using finalizer_pair_t = std::pair<std::string, finalizer_func_t>;
    using finalizer_list_t = std::deque<finalizer_pair_t>;
    using strmap_t         = std::map<string_t, std::set<string_t>>;

public:
    // Constructor and Destructors
    manager();
    ~manager();

    manager(const manager&) = delete;
    manager(manager&&)      = delete;

    manager& operator=(const manager&) = delete;
    manager& operator=(manager&&) = delete;

    // storage-types add functors to destroy the instances
    template <typename _Func>
    void add_cleanup(const std::string&, _Func&&);
    template <typename _Func>
    void add_finalizer(const std::string&, _Func&&, bool);
    void remove_cleanup(const std::string&);
    void remove_finalizer(const std::string&);
    void cleanup();
    void finalize();

    void add_text_output(const string_t& _label, const string_t& _file)
    {
        m_text_files[_label].insert(_file);
    }

    void add_json_output(const string_t& _label, const string_t& _file)
    {
        m_json_files[_label].insert(_file);
    }

    /// \fn set_write_metadata
    /// \brief Set to 0 for yes if other output, -1 for never, or 1 for yes
    void    set_write_metadata(short v) { m_write_metadata = v; }
    void    write_metadata(const char* = "");
    int32_t get_rank() const { return m_rank; }

public:
    // Public static functions
    static pointer_t instance();
    static pointer_t master_instance();
    static int32_t   total_instance_count() { return f_manager_instance_count().load(); }
    static void      use_exit_hook(bool val) { f_use_exit_hook() = val; }
    static void      exit_hook();

private:
    //----------------------------------------------------------------------------------//
    //
    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _init_storage()
    {
        using storage_type = typename _Tp::storage_type;

        static thread_local auto tmp = []() {
            if(!component::state<_Tp>::has_storage())
            {
                auto ret = storage_type::instance();
                if(ret)
                    ret->initialize();

                if(settings::debug())
                    printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
                           demangle<_Tp>().c_str(), (void*) ret,
                           (component::state<_Tp>::has_storage()) ? "true" : "false",
                           (ret) ? ((ret->empty()) ? "true" : "false") : "false");
            }
            return true;
        }();
        tim::consume_parameters(tmp);
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void _init_storage()
    {
        _init_storage<_Tp>();
        _init_storage<_Tail...>();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _print_storage()
    {
        using storage_type = typename _Tp::storage_type;

        auto ret = storage_type::noninit_instance();
        if(ret && !ret->empty())
            ret->print();

        if(settings::debug())
            printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
                   demangle<_Tp>().c_str(), (void*) ret,
                   (component::state<_Tp>::has_storage()) ? "true" : "false",
                   (ret) ? ((ret->empty()) ? "true" : "false") : "false");
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void _print_storage()
    {
        _print_storage<_Tp>();
        _print_storage<_Tail...>();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _clear()
    {
        using storage_type = typename _Tp::storage_type;

        auto ret = storage_type::noninit_instance();
        if(ret)
            ret->data().reset();

        if(settings::debug())
            printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
                   demangle<_Tp>().c_str(), (void*) ret,
                   (component::state<_Tp>::has_storage()) ? "true" : "false",
                   (ret) ? ((ret->empty()) ? "true" : "false") : "false");
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void _clear()
    {
        _clear<_Tp>();
        _clear<_Tail...>();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Archive, typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _serialize(_Archive& ar)
    {
        using storage_type = typename _Tp::storage_type;

        auto ret = storage_type::noninit_instance();
        if(ret && !ret->empty())
            ret->_serialize(ar);

        if(settings::debug())
            printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
                   demangle<_Tp>().c_str(), (void*) ret,
                   (component::state<_Tp>::has_storage()) ? "true" : "false",
                   (ret) ? ((ret->empty()) ? "true" : "false") : "false");
    }

    template <typename _Archive, typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void _serialize(_Archive& ar)
    {
        _serialize<_Archive, _Tp>(ar);
        _serialize<_Archive, _Tail...>(ar);
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _size(uint64_t& _sz)
    {
        auto label         = tim::demangle<_Tp>();
        using storage_type = typename _Tp::storage_type;
        label += std::string(" (") + tim::demangle<storage_type>() + ")";

        auto ret = storage_type::noninit_instance();
        if(ret && !ret->empty())
            _sz += ret->size();

        if(settings::debug())
            printf("[%s]> pointer: %p. has storage: %s. empty: %s...\n",
                   demangle<_Tp>().c_str(), (void*) ret,
                   (component::state<_Tp>::has_storage()) ? "true" : "false",
                   (ret) ? ((ret->empty()) ? "true" : "false") : "false");
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void _size(uint64_t& _sz)
    {
        _size<_Tp>(_sz);
        _size<_Tail...>(_sz);
    }

    //----------------------------------------------------------------------------------//
    // used to expand a tuple in settings
    //
    template <typename... _Types>
    struct filtered_get_storage
    {
        static std::string serialize(pointer_t _manager = pointer_t(nullptr))
        {
            if(_manager.get() == nullptr)
                _manager = manager::instance();
            if(!_manager)
                return "";
            std::stringstream ss;
            {
                using archive_type = trait::output_archive<manager>::type;
                auto oa            = trait::output_archive<manager>::get(ss);
                oa->setNextName("timemory");
                oa->startNode();
                {
                    oa->setNextName("ranks");
                    oa->startNode();
                    oa->makeArray();
                    _manager->_serialize<archive_type, _Types...>(*oa);
                    oa->finishNode();
                }
                oa->finishNode();
            }
            return ss.str();
        }

        static void initialize(pointer_t _manager = pointer_t(nullptr))
        {
            if(_manager.get() == nullptr)
                _manager = manager::instance();
            if(!_manager)
                return;
            _manager->_init_storage<_Types...>();
        }

        static void clear(pointer_t _manager = pointer_t(nullptr))
        {
            if(_manager.get() == nullptr)
                _manager = manager::instance();
            if(!_manager)
                return;
            _manager->_clear<_Types...>();
        }

        static void print(pointer_t _manager = pointer_t(nullptr))
        {
            if(_manager.get() == nullptr)
                _manager = manager::instance();
            if(!_manager)
                return;
            _manager->_print_storage<_Types...>();
        }

        static uint64_t size(pointer_t _manager = pointer_t(nullptr))
        {
            uint64_t _sz = 0;
            if(_manager.get() == nullptr)
                _manager = manager::instance();
            if(_manager)
                _manager->_size<_Types...>(_sz);
            return _sz;
        }
    };

    //----------------------------------------------------------------------------------//
    //
    template <typename... _Types, template <typename...> class _Tuple>
    struct filtered_get_storage<_Tuple<_Types...>>
    : public filtered_get_storage<_Types...>
    {
        using base_type = filtered_get_storage<_Types...>;
        using base_type::clear;
        using base_type::initialize;
        using base_type::print;
        using base_type::serialize;
        using base_type::size;
    };

public:
    //----------------------------------------------------------------------------------//
    //
    template <typename... _Types>
    struct get_storage : public filtered_get_storage<implemented<_Types...>>
    {
        using base_type = filtered_get_storage<implemented<_Types...>>;
        using base_type::clear;
        using base_type::initialize;
        using base_type::print;
        using base_type::serialize;
        using base_type::size;
    };

    //----------------------------------------------------------------------------------//
    //
    template <typename... _Types, template <typename...> class _Tuple>
    struct get_storage<_Tuple<_Types...>>
    : public filtered_get_storage<implemented<_Types...>>
    {
        using base_type = filtered_get_storage<implemented<_Types...>>;
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
    template <typename _Tp>
    auto get_singleton() -> decltype(_Tp::instance())
    {
        return _Tp::instance();
    }

private:
    template <typename... _Types>
    friend struct get_storage;

    template <typename... _Types>
    friend struct filtered_get_storage;

public:
    // Public member functions
    int32_t instance_count() const { return m_instance_count; }

protected:
    // protected static functions
    static comm_group_t get_communicator_group();

protected:
    // protected functions
    string_t get_prefix() const;

private:
    /// notifies that it is finalizing
    bool  m_is_finalizing  = false;
    short m_write_metadata = 0;
    /// instance id
    int32_t         m_instance_count;
    int32_t         m_rank;
    string_t        m_metadata_prefix = "";
    std::thread::id m_thread_id;
    /// increment the shared_ptr count here to ensure these instances live
    /// for the entire lifetime of the manager instance
    graph_hash_map_ptr_t   m_hash_ids     = get_hash_ids();
    graph_hash_alias_ptr_t m_hash_aliases = get_hash_aliases();
    finalizer_list_t       m_finalizer_cleanups;
    finalizer_list_t       m_master_finalizers;
    finalizer_list_t       m_worker_finalizers;
    mutex_t                m_mutex;
    auto_lock_t*           m_lock = nullptr;
    strmap_t               m_text_files;
    strmap_t               m_json_files;

private:
    struct persistent_data
    {
        persistent_data()  = default;
        ~persistent_data() = default;

        persistent_data(const persistent_data&) = delete;
        persistent_data(persistent_data&&)      = delete;
        persistent_data& operator=(const persistent_data&) = delete;
        persistent_data& operator=(persistent_data&&) = delete;

        std::atomic<int32_t> instance_count{ 0 };
        std::atomic<int32_t> thread_count{ 0 };
        bool                 use_exit_hook = true;
        pointer_t            master_instance;
    };

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

public:
    static void set_persistent_master(pointer_t _pinst)
    {
        tim::manager::f_manager_persistent_data().master_instance = _pinst;
    }
};

//======================================================================================//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/bits/manager.hpp"
#include "timemory/utility/macros.hpp"

//--------------------------------------------------------------------------------------//

#if !defined(__library_ctor__)
#    if !defined(_WINDOWS)
#        define __library_ctor__ __attribute__((constructor))
#    else
#        define __library_ctor__ static
#    endif
#endif

//--------------------------------------------------------------------------------------//

#if !defined(__library_dtor__)
#    if !defined(_WINDOWS)
#        define __library_dtor__ __attribute__((destructor))
#    else
#        define __library_dtor__ static
#    endif
#endif

//--------------------------------------------------------------------------------------//

extern "C"
{
#if defined(TIMEMORY_EXTERN_INIT)

    extern ::tim::manager*       timemory_manager_master_instance();
    __library_ctor__ extern void timemory_library_constructor();
    __library_dtor__ extern void timemory_library_destructor();

#else

#    if defined(_WINDOWS)
    static
#    endif
        ::tim::manager*
        timemory_manager_master_instance()
    {
        using manager_t     = tim::manager;
        static auto& _pinst = tim::get_shared_ptr_pair<manager_t>();
        tim::manager::set_persistent_master(_pinst.first);
        return _pinst.first.get();
    }

    __library_ctor__ void timemory_library_constructor()
    {
        auto library_ctor = tim::get_env<bool>("TIMEMORY_LIBRARY_CTOR", true);
        if(!library_ctor)
            return;

        auto _debug   = tim::settings::debug();
        auto _verbose = tim::settings::verbose();

        if(_debug || _verbose > 3)
            printf("[%s]> initializing...\n", __FUNCTION__);

        auto        _inst        = timemory_manager_master_instance();
        static auto _dir         = tim::settings::output_path();
        static auto _prefix      = tim::settings::output_prefix();
        static auto _time_output = tim::settings::time_output();
        static auto _time_format = tim::settings::time_format();
        tim::consume_parameters(_dir, _prefix, _time_output, _time_format);

        static auto              _master = tim::manager::master_instance();
        static thread_local auto _worker = tim::manager::instance();

        if(!_master && _inst)
            _master.reset(_inst);
        else if(!_master)
            _master = tim::manager::master_instance();

        if(_worker != _master)
            printf("[%s]> tim::manager :: master != worker : %p vs. %p\n", __FUNCTION__,
                   (void*) _master.get(), (void*) _worker.get());

        std::atexit(tim::timemory_finalize);
    }

    __library_dtor__ void timemory_library_destructor() {}

#endif
}

//--------------------------------------------------------------------------------------//
