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
#include <list>
#include <map>
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
    static bool is_main_thread();

    // the thread the main instance was created on
    static thread_id_t main_thread_id() { return f_main_thread(); }

    // function which just sets the main thread
    static bool init();

private:
    template <typename TypeT, typename PointerT, typename TagT>
    friend class singleton;

    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE thread_id_t& f_main_thread()
    {
        static auto _instance = std::this_thread::get_id();
        return _instance;
    }
};

//--------------------------------------------------------------------------------------//

inline bool
singleton<void, void, void>::is_main_thread()
{
    return (std::this_thread::get_id() == f_main_thread());
}

//--------------------------------------------------------------------------------------//

inline bool
singleton<void, void, void>::init()
{
    (void) f_main_thread();
    return true;
}

//--------------------------------------------------------------------------------------//

using common_singleton = singleton<void, void, void>;

//======================================================================================//

/// \class tim::singleton
/// \brief Thread-safe singleton management
///
template <typename Type, typename PointerT, typename TagT>
class singleton
{
public:
    using this_type     = singleton<Type, PointerT, TagT>;
    using thread_id_t   = std::thread::id;
    using mutex_t       = std::recursive_mutex;
    using auto_lock_t   = std::unique_lock<mutex_t>;
    using pointer       = Type*;
    using children_t    = std::set<pointer>;
    using smart_pointer = PointerT;
    using deleter_t     = std::function<void(PointerT&)>;
    using dtor_map_t    = std::map<pointer, std::function<void()>>;

    template <bool B, typename T = int>
    using enable_if_t = typename std::enable_if<B, T>::type;

public:
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
    static smart_pointer& smart_instance() { return _local_instance(); }
    static smart_pointer& smart_master_instance() { return _master_instance(); }

    // for checking but not allocating
    static pointer instance_ptr();
    static pointer master_instance_ptr() { return f_master_instance(); }

    // the thread the master instance was created on
    static thread_id_t master_thread_id() { return f_master_thread(); }

    static children_t children() { return f_children(); }
    static bool       is_master(pointer ptr) { return ptr == master_instance_ptr(); }
    static bool       is_master_thread();
    static void       insert(smart_pointer& itr);
    static void       remove(pointer itr);
    static mutex_t&   get_mutex() { return f_mutex(); }

    // since we are overloading delete we overload new
    void* operator new(size_t);

    // overload delete so that f_master_instance is guaranteed to be
    // a nullptr after deletion
    void operator delete(void* ptr);

    void initialize();
    void reset(pointer ptr);
    void reset();

private:
    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE smart_pointer& _local_instance()
    {
        static thread_local smart_pointer _instance = smart_pointer();
        return _instance;
    }
    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE smart_pointer& _master_instance()
    {
        static smart_pointer _instance = smart_pointer();
        return _instance;
    }

    void* operator new[](std::size_t) noexcept { return nullptr; }
    void  operator delete[](void*) noexcept {}

    template <typename PtrT = PointerT>
    deleter_t& get_deleter(
        enable_if_t<std::is_same<PtrT, std::shared_ptr<Type>>::value> = 0);

    template <typename PtrT = PointerT>
    deleter_t& get_deleter(
        enable_if_t<!std::is_same<PtrT, std::shared_ptr<Type>>::value> = 0);

private:
    struct persistent_data
    {
        mutex_t     m_mutex;
        thread_id_t m_master_thread   = std::this_thread::get_id();
        pointer     m_master_instance = nullptr;
        children_t  m_children        = {};
        dtor_map_t  m_dtors           = {};

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

    static TIMEMORY_NOINLINE TIMEMORY_NOCLONE persistent_data& f_persistent_data()
    {
        static persistent_data _instance{};
        return _instance;
    }

    static thread_id_t& f_master_thread();
    static mutex_t&     f_mutex();
    static pointer&     f_master_instance();
    static children_t&  f_children();
    static dtor_map_t&  f_dtors();
};

//======================================================================================//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::thread_id_t&
singleton<Type, PointerT, TagT>::f_master_thread()
{
    return f_persistent_data().m_master_thread;
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
typename singleton<Type, PointerT, TagT>::dtor_map_t&
singleton<Type, PointerT, TagT>::f_dtors()
{
    return f_persistent_data().m_dtors;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
singleton<Type, PointerT, TagT>::singleton()
{
    initialize();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
singleton<Type, PointerT, TagT>::~singleton()
{
    auto& del = get_deleter();
    if(del)
        del(_master_instance());
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::initialize()
{
    if(!f_master_instance())
    {
        f_master_thread()   = std::this_thread::get_id();
        f_master_instance() = new Type{};
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::pointer
singleton<Type, PointerT, TagT>::instance()
{
    if(std::this_thread::get_id() == f_master_thread())
    {
        return master_instance();
    }
    if(!_local_instance().get())
    {
        _local_instance().reset(new Type{});
        insert(_local_instance());
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
        f_master_thread()   = std::this_thread::get_id();
        f_master_instance() = new Type{};
    }
    return f_master_instance();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
typename singleton<Type, PointerT, TagT>::pointer
singleton<Type, PointerT, TagT>::instance_ptr()
{
    return is_master_thread() ? f_master_instance() : _local_instance().get();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void* singleton<Type, PointerT, TagT>::operator new(size_t)
{
    this_type* ptr = ::new this_type();
    return static_cast<void*>(ptr);
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::operator delete(void* ptr)
{
    if(f_master_instance() && ptr && f_master_instance() == ptr)
    {
        this_type* _instance = (this_type*) (ptr);
        ::delete _instance;
        f_master_instance() = nullptr;
    }
    else if(f_master_instance() && f_master_instance() != ptr)
    {
        this_type* _instance = (this_type*) (ptr);
        ::delete _instance;
    }
    if(std::this_thread::get_id() == f_master_thread())
        f_master_instance() = nullptr;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
bool
singleton<Type, PointerT, TagT>::is_master_thread()
{
    return std::this_thread::get_id() == f_master_thread();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::insert(smart_pointer& itr)
{
    auto_lock_t _lk{ f_mutex(), std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();
    f_children().insert(itr.get());
    f_dtors().emplace(itr.get(), [&]() { itr.reset(); });
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::remove(pointer itr)
{
    if(!itr)
        return;
    auto_lock_t _lk{ f_mutex(), std::defer_lock };
    if(!_lk.owns_lock())
        _lk.lock();
    for(auto litr = f_children().begin(); litr != f_children().end(); ++litr)
    {
        if(*litr == itr)
        {
            f_children().erase(litr);
            break;
        }
    }
    auto ditr = f_dtors().find(itr);
    if(ditr != f_dtors().end())
        f_dtors().erase(ditr);
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::reset(pointer ptr)
{
    if(is_master(ptr))
    {
        if(!f_dtors().empty())
        {
            dtor_map_t _dtors{};
            std::swap(f_dtors(), _dtors);
            for(auto& itr : _dtors)
                itr.second();
        }

        if(_master_instance().get())
        {
            _master_instance().reset();
        }
        else if(f_master_instance())
        {
            auto& del = get_deleter();
            del(_master_instance());
            f_master_instance() = nullptr;
        }
        f_persistent_data().reset();
    }
    else
    {
        remove(_local_instance().get());
        _local_instance().reset();
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
void
singleton<Type, PointerT, TagT>::reset()
{
    if(_local_instance())
    {
        remove(_local_instance().get());
        _local_instance().reset();
    }

    if(is_master_thread())
    {
        if(!f_dtors().empty())
        {
            dtor_map_t _dtors{};
            std::swap(f_dtors(), _dtors);
            for(auto& itr : _dtors)
                itr.second();
        }

        if(_master_instance().get())
        {
            _master_instance().reset();
        }
        else if(f_master_instance())
        {
            auto& del = get_deleter();
            del(_master_instance());
            f_master_instance() = nullptr;
        }
    }
    f_persistent_data().reset();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
template <typename PtrT>
typename singleton<Type, PointerT, TagT>::deleter_t&
    singleton<Type, PointerT, TagT>::get_deleter(
        enable_if_t<std::is_same<PtrT, std::shared_ptr<Type>>::value>)
{
    static deleter_t _instance = [](PointerT&) {};
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename PointerT, typename TagT>
template <typename PtrT>
typename singleton<Type, PointerT, TagT>::deleter_t&
    singleton<Type, PointerT, TagT>::get_deleter(
        enable_if_t<!std::is_same<PtrT, std::shared_ptr<Type>>::value>)
{
    static deleter_t _instance = [](PointerT& _master) {
        auto& del = _master.get_deleter();
        del(_master.get());
        _master.reset(nullptr);
    };
    return _instance;
}

//--------------------------------------------------------------------------------------//

namespace internal
{
namespace
{
// ensure assigned before main
// NOLINTNEXTLINE
bool singleton_main_thread_assigned = ::tim::singleton<void, void, void>::init();
}  // namespace
}  // namespace internal

//--------------------------------------------------------------------------------------//

}  // namespace tim
