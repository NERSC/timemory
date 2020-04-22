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

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

//======================================================================================//

namespace tim
{
//======================================================================================//

template <typename Type,
          typename Pointer = std::unique_ptr<Type, std::default_delete<Type>>>
class singleton
{
public:
    using this_type     = singleton<Type, Pointer>;
    using thread_id_t   = std::thread::id;
    using mutex_t       = std::recursive_mutex;
    using auto_lock_t   = std::unique_lock<mutex_t>;
    using pointer       = Type*;
    using list_t        = std::set<pointer>;
    using smart_pointer = Pointer;
    using deleter_t     = std::function<void(Pointer&)>;

    template <bool B, typename T = int>
    using enable_if_t = typename std::enable_if<B, T>::type;

public:
    // Constructor and Destructors
    singleton();
    singleton(pointer);
    ~singleton();

    singleton(const singleton&) = delete;
    singleton(singleton&&)      = delete;
    singleton& operator=(const singleton&) = delete;
    singleton& operator=(singleton&&) = delete;

public:
    // public member function
    void initialize();
    void initialize(pointer);
    void destroy();

    // instance functions that initialize if nullptr
    static pointer instance();
    static pointer master_instance();

    // instance functions that do not initialize
    smart_pointer&        smart_instance() { return _local_instance(); }
    static smart_pointer& smart_master_instance() { return _master_instance(); }

    // for checking but not allocating
    pointer instance_ptr()
    {
        return is_master_thread() ? f_master_instance() : _local_instance().get();
    }
    static pointer master_instance_ptr() { return f_master_instance(); }

    // the thread the master instance was created on
    static thread_id_t master_thread_id() { return f_master_thread(); }

    // since we are overloading delete we overload new
    void* operator new(size_t)
    {
        this_type* ptr = ::new this_type();
        return static_cast<void*>(ptr);
    }

    // overload delete so that f_master_instance is guaranteed to be
    // a nullptr after deletion
    void operator delete(void* ptr)
    {
        this_type* _instance = (this_type*) (ptr);
        ::delete _instance;
        if(std::this_thread::get_id() == f_master_thread())
            f_master_instance() = nullptr;
    }

    static list_t children() { return f_children(); }
    static bool   is_master(pointer ptr) { return ptr == master_instance_ptr(); }
    static bool   is_master_thread()
    {
        return std::this_thread::get_id() == f_master_thread();
    }

    static void insert(pointer itr)
    {
        auto_lock_t l(f_mutex());
        f_children().insert(itr);
    }

    static void remove(pointer itr)
    {
        auto_lock_t l(f_mutex());
        for(auto litr = f_children().begin(); litr != f_children().end(); ++litr)
        {
            if(*litr == itr)
            {
                f_children().erase(litr);
                break;
            }
        }
    }

    static mutex_t& get_mutex() { return f_mutex(); }

    void reset(pointer ptr)
    {
        if(is_master(ptr))
        {
            if(_master_instance().get())
                _master_instance().reset();
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
            _local_instance().reset();
        }
    }

    void reset()
    {
        if(is_master_thread())
            _master_instance().reset();
        _local_instance().reset();
        f_persistent_data().reset();
    }

private:
    // Private functions
    static smart_pointer& _local_instance()
    {
        static thread_local smart_pointer _instance = smart_pointer();
        return _instance;
    }

    static smart_pointer& _master_instance()
    {
        static smart_pointer _instance = smart_pointer();
        return _instance;
    }

    void* operator new[](std::size_t) noexcept { return nullptr; }
    void  operator delete[](void*) noexcept {}

    template <typename Tp = Type, typename PtrT = Pointer,
              enable_if_t<(std::is_same<PtrT, std::shared_ptr<Tp>>::value)> = 0>
    deleter_t& get_deleter()
    {
        static deleter_t _instance = [](Pointer&) {};
        return _instance;
    }

    template <typename Tp = Type, typename PtrT = Pointer,
              enable_if_t<!(std::is_same<PtrT, std::shared_ptr<Tp>>::value)> = 0>
    deleter_t& get_deleter()
    {
        static deleter_t _instance = [](Pointer& _master) {
            auto& del = _master.get_deleter();
            del(_master.get());
            _master.reset(nullptr);
        };
        return _instance;
    }

private:
    // Private variables
    struct persistent_data
    {
        thread_id_t m_master_thread = std::this_thread::get_id();
        mutex_t     m_mutex;
        pointer     m_master_instance = nullptr;
        list_t      m_children        = {};

        persistent_data()                       = default;
        ~persistent_data()                      = default;
        persistent_data(const persistent_data&) = delete;
        persistent_data(persistent_data&&)      = delete;
        persistent_data& operator=(const persistent_data&) = delete;
        persistent_data& operator=(persistent_data&&) = delete;

        persistent_data(pointer _master, std::thread::id _tid)
        : m_master_thread(_tid)
        , m_master_instance(_master)
        {}

        void reset()
        {
            m_master_instance = nullptr;
            m_children.clear();
        }
    };

    bool                m_is_master = false;
    static thread_id_t& f_master_thread();
    static mutex_t&     f_mutex();
    static pointer&     f_master_instance();
    static list_t&      f_children();

    static persistent_data& f_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }
};

//======================================================================================//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::thread_id_t&
singleton<Type, Pointer>::f_master_thread()
{
    return f_persistent_data().m_master_thread;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::pointer&
singleton<Type, Pointer>::f_master_instance()
{
    return f_persistent_data().m_master_instance;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::mutex_t&
singleton<Type, Pointer>::f_mutex()
{
    return f_persistent_data().m_mutex;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::list_t&
singleton<Type, Pointer>::f_children()
{
    return f_persistent_data().m_children;
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
singleton<Type, Pointer>::singleton()
{
    initialize();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
singleton<Type, Pointer>::singleton(pointer ptr)
{
    initialize(ptr);
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
singleton<Type, Pointer>::~singleton()
{
    auto& del = get_deleter();
    del(_master_instance());
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
void
singleton<Type, Pointer>::initialize()
{
    if(!f_master_instance())
    {
        f_master_thread()   = std::this_thread::get_id();
        f_master_instance() = new Type();
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
void
singleton<Type, Pointer>::initialize(pointer ptr)
{
    if(!f_master_instance())
    {
        f_master_thread()   = std::this_thread::get_id();
        f_master_instance() = ptr;
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
void
singleton<Type, Pointer>::destroy()
{
    if(std::this_thread::get_id() == f_master_thread() && f_master_instance())
    {
        delete f_master_instance();
        f_master_instance() = nullptr;
    }
    else
    {
        remove(_local_instance().get());
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::pointer
singleton<Type, Pointer>::instance()
{
    if(std::this_thread::get_id() == f_master_thread())
        return master_instance();
    else if(!_local_instance().get())
    {
        _local_instance().reset(new Type());
        insert(_local_instance().get());
    }
    return _local_instance().get();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::pointer
singleton<Type, Pointer>::master_instance()
{
    if(!f_master_instance())
    {
        f_master_thread()   = std::this_thread::get_id();
        f_master_instance() = new Type();
    }
    return f_master_instance();
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
