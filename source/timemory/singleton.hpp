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

#pragma once

#include "timemory/macros.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <thread>

//======================================================================================//

namespace tim
{
//======================================================================================//

template <typename _Tp>
void
default_deleter(_Tp* ptr)
{
    delete ptr;
}

//======================================================================================//

template <typename _Tp, typename Deleter = std::default_delete<_Tp>>
class singleton
{
public:
    using Type           = _Tp;
    using this_type      = singleton<Type, Deleter>;
    using value_type     = Type;
    using pointer        = Type*;
    using reference      = Type&;
    using thread_id_t    = std::thread::id;
    using unique_pointer = std::unique_ptr<value_type, Deleter>;

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
    static unique_pointer raw_instance() { return _local_instance(); }

    // for checking but not allocating
    static pointer instance_ptr() { return _local_instance().get(); }
    static pointer master_instance_ptr() { return f_master_instance; }

    // the thread the master instance was created on
    static thread_id_t master_thread_id() { return f_master_thread; }

    // since we are overloading delete we overload new
    void* operator new(size_t)
    {
        void* ptr = ::new this_type();
        return ptr;
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

private:
    // Private functions
    static unique_pointer& _local_instance()
    {
        tim_static_thread_local unique_pointer _instance = unique_pointer();
        return _instance;
    }

    static unique_pointer& _master_instance()
    {
        static unique_pointer _instance = unique_pointer();
        return _instance;
    }

    void* operator new[](std::size_t) noexcept { return nullptr; }
    void  operator delete[](void*) noexcept {}

private:
    // Private variables
    static thread_id_t f_master_thread;
    static pointer     f_master_instance;
};

//======================================================================================//

template <typename _Tp, typename Deleter>
typename singleton<_Tp, Deleter>::thread_id_t singleton<_Tp, Deleter>::f_master_thread =
    std::this_thread::get_id();

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Deleter>
typename singleton<_Tp, Deleter>::pointer singleton<_Tp, Deleter>::f_master_instance =
    singleton<_Tp, Deleter>::_master_instance().get();

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Deleter>
singleton<_Tp, Deleter>::singleton()
{
    initialize();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Deleter>
singleton<_Tp, Deleter>::singleton(pointer ptr)
{
    initialize(ptr);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Deleter>
singleton<_Tp, Deleter>::~singleton()
{
    // should be called at __cxa_finalize so don't bother deleting
    delete f_master_instance;
    f_master_instance = nullptr;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Deleter>
void
singleton<_Tp, Deleter>::initialize()
{
    if(!f_master_instance)
    {
        f_master_thread   = std::this_thread::get_id();
        f_master_instance = new _Tp();
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Deleter>
void
singleton<_Tp, Deleter>::initialize(pointer ptr)
{
    if(!f_master_instance)
    {
        f_master_thread   = std::this_thread::get_id();
        f_master_instance = ptr;
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Deleter>
void
singleton<_Tp, Deleter>::destroy()
{
    //_local_instance().reset();
    if(std::this_thread::get_id() == f_master_thread)
        f_master_instance = nullptr;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Deleter>
typename singleton<_Tp, Deleter>::pointer
singleton<_Tp, Deleter>::instance()
{
    if(std::this_thread::get_id() == f_master_thread)
        return master_instance();
    else if(!_local_instance())
        _local_instance().reset(new _Tp());
    return _local_instance().get();
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename Deleter>
typename singleton<_Tp, Deleter>::pointer
singleton<_Tp, Deleter>::master_instance()
{
    if(!f_master_instance)
    {
        f_master_thread   = std::this_thread::get_id();
        f_master_instance = new _Tp();
    }
    return f_master_instance;
}

//======================================================================================//

}  // namespace tim

//======================================================================================//
