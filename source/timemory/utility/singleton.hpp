//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file utility/singleton.hpp
 * \headerfile utility/singleton.hpp "timemory/utility/singleton.hpp"
 * This is the C++ class provides thread-local singleton functionality with
 * the ability to overload the destruction of the singleton (critical to automatic
 * output at the end of the application)
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/macros/attributes.hpp"

#include <cstddef>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

//======================================================================================//

namespace tim
{
//
template <typename Type,
          typename PointerT = std::unique_ptr<Type, std::default_delete<Type>>,
          typename TagT     = TIMEMORY_API>
class singleton;
//
template <>
class singleton<void, void, void>
{
public:
    using thread_t    = std::thread;
    using thread_id_t = std::thread::id;

    // returns whether current thread is primary thread
    static bool is_main_thread()
    {
        return (std::this_thread::get_id() == f_main_thread());
    }

    // the thread the main instance was created on
    static thread_id_t main_thread_id() { return f_main_thread(); }

    // function which just sets the main thread
    static bool init()
    {
        (void) f_main_thread();
        return true;
    }

private:
    template <typename TypeT, typename PointerT, typename TagT>
    friend class singleton;

    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE thread_id_t& f_main_thread()
    {
        static auto _instance = std::this_thread::get_id();
        return _instance;
    }
};
//
namespace internal
{
namespace
{
// ensure assigned before main
bool singleton_main_thread_assigned = ::tim::singleton<void, void, void>::init();
}  // namespace
}  // namespace internal
//
template <typename Type, typename PointerT, typename TagT>
class singleton : private singleton<void, void, void>
{
public:
    using common_type    = singleton<void, void, void>;
    using this_type      = singleton<Type, PointerT, TagT>;
    using thread_id_t    = std::thread::id;
    using mutex_t        = std::recursive_mutex;
    using auto_lock_t    = std::unique_lock<mutex_t>;
    using pointer        = Type*;
    using children_t     = std::set<pointer>;
    using children_tid_t = std::vector<std::pair<pointer, thread_id_t>>;
    using smart_pointer  = PointerT;
    using deleter_t      = std::function<void(Type*&)>;

    template <bool B, typename T = int>
    using enable_if_t = typename std::enable_if<B, T>::type;

public:
    // Constructor and Destructors
    singleton();
    ~singleton();

    singleton(const singleton&) = delete;
    singleton(singleton&&)      = delete;
    singleton& operator=(const singleton&) = delete;
    singleton& operator=(singleton&&) = delete;

public:
    // instance functions that initialize if nullptr
    static pointer instance();
    static pointer master_instance();

    // instance functions that do not initialize
    smart_pointer&        smart_instance() { return _local_instance(); }
    static smart_pointer& smart_master_instance() { return _master_instance(); }

    // for checking but not allocating
    static pointer instance_ptr()
    {
        return is_master_thread() ? f_master_instance() : _local_instance().get();
    }
    static pointer master_instance_ptr() { return f_master_instance(); }

    // the thread the master instance was created on
    static thread_id_t master_thread_id() { return common_type::f_main_thread(); }

    static children_t     children() { return f_children(); }
    static children_tid_t children_tids() { return f_children_tids(); }
    static bool           is_master(pointer ptr) { return ptr == master_instance_ptr(); }
    static bool           is_master_thread() { return common_type::is_main_thread(); }
    static mutex_t&       get_mutex() { return f_mutex(); }

    using common_type::is_main_thread;
    using common_type::main_thread_id;

    static void insert(pointer itr);
    static void remove(pointer itr);
    void        reset(pointer ptr);
    void        reset();

    // since we are overloading delete we overload new
    /*void* operator new(size_t)
    {
        this_type* ptr = ::new this_type();
        return static_cast<void*>(ptr);
    }*/

    // overload delete so that f_master_instance is guaranteed to be
    // a nullptr after deletion
    void operator delete(void* ptr)
    {
        if(ptr && f_master_instance() && f_master_instance() == ptr)
        {
            this_type* _instance = (this_type*) (ptr);
            ::delete _instance;
            f_master_instance() = nullptr;
        }
        else if(ptr)
        {
            this_type* _instance = (this_type*) (ptr);
            ::delete _instance;
        }
        if(std::this_thread::get_id() == f_master_thread())
            f_master_instance() = nullptr;
    }

private:
    void initialize();

    // Private functions
    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE smart_pointer& _local_instance()
    {
        static thread_local auto _instance = smart_pointer{};
        return _instance;
    }

    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE smart_pointer& _master_instance()
    {
        static auto _instance = smart_pointer{ f_master_instance() };
        return _instance;
    }

    void* operator new[](std::size_t) noexcept { return nullptr; }
    void  operator delete[](void*) noexcept {}

    template <typename PtrT = PointerT>
    deleter_t& get_deleter(
        enable_if_t<std::is_same<PtrT, std::shared_ptr<Type>>::value> = 0)
    {
        static deleter_t _instance = [](Type*&) {};
        return _instance;
    }

    template <typename PtrT = PointerT>
    deleter_t& get_deleter(
        enable_if_t<!std::is_same<PtrT, std::shared_ptr<Type>>::value> = 0)
    {
        static deleter_t _instance = [](Type*& _master) {
            if(_master_instance() && _master_instance().get() == _master)
            {
                _master_instance().reset();
                _master = nullptr;
            }
            else if(_master)
            {
                auto& _del = _master_instance().get_deleter();
                _del(_master);
                _master = nullptr;
            }
        };
        return _instance;
    }

private:
    // Private variables
    struct persistent_data
    {
        thread_id_t    m_master_thread = common_type::f_main_thread();
        mutex_t        m_mutex{};
        pointer        m_master_instance = nullptr;
        children_t     m_children        = {};
        children_tid_t m_children_tids   = {};

        persistent_data()                       = default;
        ~persistent_data()                      = default;
        persistent_data(const persistent_data&) = delete;
        persistent_data(persistent_data&&)      = delete;
        persistent_data& operator=(const persistent_data&) = delete;
        persistent_data& operator=(persistent_data&&) = delete;

        void reset()
        {
            m_master_instance = nullptr;
            m_children.clear();
        }
    };

    bool                     m_is_master = false;
    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE thread_id_t& f_master_thread();
    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE mutex_t& f_mutex();
    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE pointer& f_master_instance();
    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE children_t& f_children();
    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE children_tid_t& f_children_tids();

    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE persistent_data& f_persistent_data()
    {
        static persistent_data _instance{};
        return _instance;
    }
};

//======================================================================================//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::thread_id_t&
singleton<Type, PointerT, TagT>::f_master_thread()
{
    return common_type::f_main_thread();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::pointer&
singleton<Type, PointerT, TagT>::f_master_instance()
{
    return f_persistent_data().m_master_instance;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::mutex_t&
singleton<Type, PointerT, TagT>::f_mutex()
{
    return f_persistent_data().m_mutex;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::children_t&
singleton<Type, PointerT, TagT>::f_children()
{
    return f_persistent_data().m_children;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::children_tid_t&
singleton<Type, PointerT, TagT>::f_children_tids()
{
    return f_persistent_data().m_children_tids;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
singleton<Type, PointerT, TagT>::singleton()
{
    if(!f_master_instance())
    {
        f_master_instance() = new Type{};
        (void) _master_instance();
        assert(f_master_instance() == _master_instance().get());
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
singleton<Type, PointerT, TagT>::~singleton()
{
    reset();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::pointer
singleton<Type, PointerT, TagT>::instance()
{
    if(is_master_thread())
    {
        return master_instance();
    }
    if(!_local_instance())
    {
        _local_instance().reset(new Type{});
        insert(_local_instance().get());
    }
    return _local_instance().get();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::pointer
singleton<Type, PointerT, TagT>::master_instance()
{
    if(!f_master_instance())
    {
        f_master_instance() = new Type{};
        (void) _master_instance();
        assert(f_master_instance() == _master_instance().get());
    }
    return f_master_instance();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::insert(pointer itr)
{
    auto_lock_t _lk{ f_mutex(), std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();
    f_children().insert(itr);
    f_children_tids().emplace_back(itr, std::this_thread::get_id());
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::remove(pointer itr)
{
    auto_lock_t _lk{ f_mutex(), std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();
    f_children().erase(itr);
    auto& _v = f_children_tids();
    _v.erase(std::remove_if(_v.begin(), _v.end(),
                            [itr](const auto& _entry) { return _entry.first == itr; }),
             _v.end());
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::reset(pointer ptr)
{
    if(is_master(ptr) || is_master_thread())
    {
        if(_master_instance())
        {
            _master_instance().reset();
            f_master_instance() = nullptr;
        }
        else if(f_master_instance())
        {
            get_deleter()(f_master_instance());
        }
        f_persistent_data().reset();
    }
    else
    {
        _local_instance().reset();
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::reset()
{
    if(is_master_thread())
    {
        if(_master_instance())
        {
            _master_instance().reset();
            f_master_instance() = nullptr;
        }
        else if(f_master_instance())
        {
            get_deleter()(f_master_instance());
        }
    }
    _local_instance().reset();
    f_persistent_data().reset();
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
