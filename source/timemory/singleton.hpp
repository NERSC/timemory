//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
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

/** \file singleton.hpp
 * \headerfile singleton.hpp "timemory/singleton.hpp"
 * This is the C++ class provides thread-local singleton functionality with
 * the ability to overload the destruction of the singleton (critical to automatic
 * output at the end of the application)
 *
 */

#pragma once

#include "timemory/macros.hpp"

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
    using mutex_t       = std::mutex;
    using auto_lock_t   = std::unique_lock<mutex_t>;
    using pointer       = Type*;
    using list_t        = std::set<pointer>;
    using smart_pointer = Pointer;
    template <bool B, typename T = int>
    using enable_if_t = typename std::enable_if<B, T>::type;

public:
    // Constructor and Destructors
    singleton();
    singleton(pointer);
    ~singleton();

public:
    // public member function
    void initialize();
    void initialize(pointer);
    void destroy();

    // instance functions that initialize if nullptr
    static pointer instance();
    static pointer master_instance();

    // instance functions that do not initialize
    static smart_pointer smart_instance() { return _local_instance(); }
    static smart_pointer smart_master_instance() { return _master_instance(); }

    // for checking but not allocating
    static pointer instance_ptr() { return _local_instance().get(); }
    static pointer master_instance_ptr() { return f_master_instance; }

    // the thread the master instance was created on
    static thread_id_t master_thread_id() { return f_master_thread; }

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
        if(std::this_thread::get_id() == f_master_thread)
            f_master_instance = nullptr;
    }

    static list_t children() { return f_children; }
    static bool   is_master(pointer ptr) { return ptr == master_instance_ptr(); }

    static void insert(pointer itr)
    {
        auto_lock_t l(f_mutex);
        f_children.insert(itr);
    }

    static void remove(pointer itr)
    {
        auto_lock_t l(f_mutex);
        for(auto litr = f_children.begin(); litr != f_children.end(); ++litr)
        {
            if(*litr == itr)
            {
                f_children.erase(litr);
                break;
            }
        }
    }

    static mutex_t& get_mutex() { return f_mutex; }

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

private:
    // Private variables
    static thread_id_t f_master_thread;
    static mutex_t     f_mutex;
    static pointer     f_master_instance;
    static list_t      f_children;
};

//======================================================================================//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::thread_id_t singleton<Type, Pointer>::f_master_thread =
    std::this_thread::get_id();

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::pointer singleton<Type, Pointer>::f_master_instance =
    singleton<Type, Pointer>::_master_instance().get();

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::mutex_t singleton<Type, Pointer>::f_mutex;

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
typename singleton<Type, Pointer>::list_t singleton<Type, Pointer>::f_children;

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
    // should be called at __cxa_finalize so don't bother deleting
    auto& del = _master_instance().get_deleter();
    del(_master_instance().get());
    //_master_instance().reset();
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
void
singleton<Type, Pointer>::initialize()
{
    if(!f_master_instance)
    {
        f_master_thread   = std::this_thread::get_id();
        f_master_instance = new Type();
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
void
singleton<Type, Pointer>::initialize(pointer ptr)
{
    if(!f_master_instance)
    {
        f_master_thread   = std::this_thread::get_id();
        f_master_instance = ptr;
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type, typename Pointer>
void
singleton<Type, Pointer>::destroy()
{
    //_local_instance().reset();
    if(std::this_thread::get_id() == f_master_thread)
    {
        delete f_master_instance;
        f_master_instance = nullptr;
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
    if(std::this_thread::get_id() == f_master_thread)
        return master_instance();
    else if(!_local_instance())
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
    if(!f_master_instance)
    {
        f_master_thread   = std::this_thread::get_id();
        f_master_instance = new Type();
    }
    return f_master_instance;
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
