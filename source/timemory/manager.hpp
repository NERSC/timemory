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

/** \file manager.hpp
 * \headerfile manager.hpp "timemory/manager.hpp"
 * Static singleton handler that is not templated. In general, this is the
 * first object created and last object destroyed. It should be utilized to
 * store type-independent data
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/backends/papi.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/storage.hpp"
#include "timemory/utility/utility.hpp"

//--------------------------------------------------------------------------------------//

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>

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
    using finalizer_list_t = std::deque<finalizer_func_t>;

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
    void add_finalizer(_Func&&, bool);

    void finalize();

public:
    // Public static functions
    static pointer_t instance();
    static pointer_t master_instance();
    static void      enable(bool val = true) { settings::enabled() = val; }
    static void      disable(bool val = true) { settings::enabled() = !val; }
    static bool      is_enabled() { return settings::enabled(); }
    static void      max_depth(const int32_t& val) { settings::max_depth() = val; }
    static int32_t   max_depth() { return settings::max_depth(); }
    static int32_t   total_instance_count() { return f_manager_instance_count().load(); }

    static void exit_hook();

private:
    //----------------------------------------------------------------------------------//
    //
    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _init_storage()
    {
        if(!component::properties<_Tp>::has_storage())
        {
            using storage_type = typename _Tp::storage_type;
            auto ret           = storage_type::instance();
            ret->initialize();
        }
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
        if(component::properties<_Tp>::has_storage())
        {
            using storage_type = typename _Tp::storage_type;
            auto ret           = storage_type::noninit_instance();
            if(ret && !ret->empty())
                ret->print();
        }
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
        if(component::properties<_Tp>::has_storage())
        {
            using storage_type = typename _Tp::storage_type;
            auto ret           = storage_type::noninit_instance();
            if(ret)
                ret->data().reset();
        }
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
        if(component::properties<_Tp>::has_storage())
        {
            using storage_type = typename _Tp::storage_type;
            auto ret           = storage_type::noninit_instance();
            if(ret && !ret->empty())
                ret->_serialize(ar);
        }
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
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void _size(uint64_t& _sz)
    {
        _size<_Tp>(_sz);
        _size<_Tail...>(_sz);
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _size(uint64_t& _sz)
    {
        if(component::properties<_Tp>::has_storage())
        {
            using storage_type = typename _Tp::storage_type;
            auto ret           = storage_type::noninit_instance();
            if(ret && !ret->empty())
                _sz += ret->size();
        }
    }

    //----------------------------------------------------------------------------------//
    // used to expand a tuple in settings
    //
    template <typename... _Types>
    struct filtered_get_storage
    {
        using indent                  = cereal::JSONOutputArchive::Options::IndentChar;
        static constexpr auto spacing = indent::space;

        static std::string serialize(pointer_t _manager = pointer_t(nullptr))
        {
            if(_manager.get() == nullptr)
                _manager = manager::instance();
            if(!_manager)
                return "";
            std::stringstream ss;
            {
                // args: precision, spacing, indent size
                cereal::JSONOutputArchive::Options opts(12, spacing, 4);
                cereal::JSONOutputArchive          oa(ss, opts);
                oa.setNextName("rank");
                oa.startNode();
                auto rank = mpi::rank();
                oa(cereal::make_nvp("rank_id", rank));
                _manager->_serialize<decltype(oa), _Types...>(oa);
                oa.finishNode();
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
    /// number of timing manager instances
    static std::atomic<int32_t>& f_manager_instance_count();
    /// instance id
    int32_t m_instance_count;
    /// increment the shared_ptr count here to ensure these instances live
    /// for the entire lifetime of the manager instance
    graph_hash_map_ptr_t   m_hash_ids     = get_hash_ids();
    graph_hash_alias_ptr_t m_hash_aliases = get_hash_aliases();
    finalizer_list_t       m_master_finalizers;
    finalizer_list_t       m_worker_finalizers;
    mutex_t                m_mutex;

private:
    /// num-threads based on number of managers created
    static std::atomic<int32_t>& f_thread_counter();
};

//======================================================================================//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/bits/manager.hpp"

//--------------------------------------------------------------------------------------//

#if !defined(__library_ctor__)
#    if !defined(_WIN32) && !defined(_WIN64)
#        define __library_ctor__ __attribute__((constructor))
#    else
#        define __library_ctor__
#    endif
#endif

//--------------------------------------------------------------------------------------//

#if !defined(__library_dtor__)
#    if !defined(_WIN32) && !defined(_WIN64)
#        define __library_dtor__ __attribute__((destructor))
#    else
#        define __library_dtor__
#    endif
#endif

//--------------------------------------------------------------------------------------//
