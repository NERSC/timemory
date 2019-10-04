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

namespace details
{
struct tim_api manager_deleter;
}

//--------------------------------------------------------------------------------------//

class tim_api manager
{
public:
    using this_type     = manager;
    using pointer_t     = std::unique_ptr<this_type, details::manager_deleter>;
    using singleton_t   = singleton<this_type, pointer_t>;
    using size_type     = std::size_t;
    using string_t      = std::string;
    using comm_group_t  = std::tuple<mpi::comm_t, int32_t>;
    using mutex_t       = std::mutex;
    using auto_lock_t   = std::unique_lock<mutex_t>;
    using pointer       = singleton_t::pointer;
    using smart_pointer = singleton_t::smart_pointer;
    using void_counter  = counted_object<void>;

public:
    // Constructor and Destructors
    manager();
    ~manager();

public:
    // Public static functions
    static pointer instance();
    static pointer master_instance();
    static pointer noninit_instance();
    static pointer noninit_master_instance();
    static void    enable(bool val = true) { void_counter::enable(val); }
    static void    disable(bool val = true) { void_counter::enable(!val); }
    static bool    is_enabled() { return void_counter::enable(); }
    static void    max_depth(const int32_t& val) { void_counter::set_max_depth(val); }
    static int32_t max_depth() { return void_counter::max_depth(); }
    static int32_t total_instance_count() { return f_manager_instance_count().load(); }

    static void exit_hook();

private:
    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _init_storage()
    {
        using storage_type = typename _Tp::storage_type;
        auto ret           = storage_type::instance();
        consume_parameters(ret);
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void _init_storage()
    {
        _init_storage<_Tp>();
        _init_storage<_Tail...>();
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _print_storage()
    {
        using storage_type = typename _Tp::storage_type;
        auto ret           = storage_type::noninit_instance();
        if(ret && !ret->empty())
            ret->print();
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void _print_storage()
    {
        _print_storage<_Tp>();
        _print_storage<_Tail...>();
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) == 0), int> = 0>
    void _clear()
    {
        using storage_type = typename _Tp::storage_type;
        auto ret           = storage_type::noninit_instance();
        if(ret)
            ret->data().clear();
    }

    template <typename _Tp, typename... _Tail,
              enable_if_t<(sizeof...(_Tail) > 0), int> = 0>
    void _clear()
    {
        _clear<_Tp>();
        _clear<_Tail...>();
    }

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

public:
    // used to expand a tuple in settings
    template <typename... _Types>
    struct get_storage
    {
        using indent                  = cereal::JSONOutputArchive::Options::IndentChar;
        static constexpr auto spacing = indent::space;
        static std::string    serialize()
        {
            manager*          _manager = manager::instance();
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

        static void initialize()
        {
            manager* _manager = manager::instance();
            _manager->_init_storage<_Types...>();
        }

        static void clear()
        {
            manager* _manager = manager::instance();
            _manager->_clear<_Types...>();
        }

        static void print()
        {
            manager* _manager = manager::instance();
            _manager->_print_storage<_Types...>();
        }
    };

    template <typename... _Types>
    struct get_storage<std::tuple<_Types...>>
    {
        using indent                  = cereal::JSONOutputArchive::Options::IndentChar;
        static constexpr auto spacing = indent::space;

        static std::string serialize(manager* _manager = nullptr)
        {
            if(_manager == nullptr)
                _manager = manager::instance();
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

        static void initialize(manager* _manager = nullptr)
        {
            if(_manager == nullptr)
                _manager = manager::instance();
            _manager->_init_storage<_Types...>();
        }

        static void clear(manager* _manager = nullptr)
        {
            if(_manager == nullptr)
                _manager = manager::instance();
            _manager->_clear<_Types...>();
        }

        static void print(manager* _manager = nullptr)
        {
            if(_manager == nullptr)
                _manager = manager::instance();
            _manager->_print_storage<_Types...>();
        }
    };

private:
    template <typename... _Types>
    friend struct get_storage;

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
    // private static variables
    /// for temporary enabling/disabling
    // static bool f_enabled();
    /// number of timing manager instances
    static std::atomic<int32_t>& f_manager_instance_count();

private:
    // private variables
    /// instance id
    int32_t m_instance_count;

private:
    /// num-threads based on number of managers created
    static std::atomic<int32_t>& f_thread_counter()
    {
        static std::atomic<int32_t> _instance;
        return _instance;
    }
};

//======================================================================================//

namespace details
{
//--------------------------------------------------------------------------------------//

struct manager_deleter
{
    using Type        = tim::manager;
    using pointer_t   = std::unique_ptr<Type, manager_deleter>;
    using singleton_t = singleton<Type, pointer_t>;

    void operator()(Type* ptr)
    {
        Type*           master     = singleton_t::master_instance_ptr();
        std::thread::id master_tid = singleton_t::master_thread_id();
        std::thread::id this_tid   = std::this_thread::get_id();

        if(ptr && master && ptr != master)
        {
        }
        else
        {
            if(ptr)
            {
                // ptr->print();
            }
            else if(master)
            {
                // master->print();
            }
        }

        if(this_tid == master_tid)
        {
            // delete ptr;
        }
        else
        {
            if(master && ptr != master)
            {
                singleton_t::remove(ptr);
            }
            delete ptr;
        }
    }
};

//--------------------------------------------------------------------------------------//

inline manager::singleton_t&
manager_singleton()
{
    static manager::singleton_t _instance = manager::singleton_t::instance();
    return _instance;
}

//--------------------------------------------------------------------------------------//

}  // namespace details
/*
template <typename... _Types>
struct manager::initialize<std::tuple<_Types...>>
{
    static void storage()
    {
        manager* _manager = manager::instance();
        _manager->initialize_storage<_Types...>();
    }
};
*/
//======================================================================================//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/details/manager.hpp"

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

#if !defined(TIMEMORY_EXTERN_INIT)
/*
//--------------------------------------------------------------------------------------//

#include "timemory/bits/timemory.hpp"

//--------------------------------------------------------------------------------------//
//
static void
timemory_library_constructor() __library_ctor__;

//--------------------------------------------------------------------------------------//
//
void
timemory_library_constructor()
{
#if defined(DEBUG)
    auto _debug   = tim::settings::debug();
    auto _verbose = tim::settings::verbose();
#endif

#if defined(DEBUG)
    if(_debug || _verbose > 3)
        printf("[%s]> initializing manager...\n", __FUNCTION__);
#endif

    // fully initialize manager
    auto _master   = tim::manager::master_instance();
    auto _instance = tim::manager::instance();

    if(_instance != _master)
        printf("[%s]> master_instance() != instance() : %p vs. %p\n", __FUNCTION__,
               (void*) _instance, (void*) _master);

#if defined(DEBUG)
    if(_debug || _verbose > 3)
        printf("[%s]> initializing storage...\n", __FUNCTION__);
#endif

    // initialize storage
    using tuple_type = tim::available_tuple<tim::complete_tuple_t>;
    tim::manager::get_storage<tuple_type>::initialize(_instance);
}

//--------------------------------------------------------------------------------------//
*/
#endif
