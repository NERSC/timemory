//  MIT License
//  
//  Copyright (c) 2018, The Regents of the University of California, 
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//  
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#ifndef singleton_hpp_
#define singleton_hpp_

#include "timemory/macros.hpp"
#include <thread>
#include <memory>

//============================================================================//

namespace tim
{

//============================================================================//

template <typename _Tp>
class singleton
{
public:
    typedef _Tp                             value_type;
    typedef _Tp*                            pointer;
    typedef _Tp&                            reference;
    typedef _Tp*&                           pointer_reference;
    typedef const _Tp*                      const_pointer;
    typedef const _Tp&                      const_reference;
    typedef const _Tp*&                     const_pointer_reference;
    typedef std::thread::id                 thread_id_t;
    typedef std::shared_ptr<value_type>     shared_pointer;
    typedef std::shared_ptr<value_type>&    shared_pointer_reference;

public:
    // Constructor and Destructors
    singleton() { }
    singleton(pointer);
    // Virtual destructors are required by abstract classes 
    // so add it by default, just in case
    virtual ~singleton() { }

public:
    // public member function
    void initialize();
    void initialize(pointer);
    void destroy();

    // Public functions
    static shared_pointer instance();
    static shared_pointer master_instance();

    // for checking but not allocating
    static pointer unsafe_instance()        { return _local_instance().get(); }
    static pointer unsafe_master_instance() { return f_master_instance.get(); }

private:
    // Private functions
    static shared_pointer_reference _local_instance()
    {
        tim_static_thread_local shared_pointer _instance = shared_pointer();
        return _instance;
    }

private:
    // Private variables
    static  thread_id_t         f_master_thread;
    static  shared_pointer      f_master_instance;
};

//============================================================================//

template <typename _Tp>
typename singleton<_Tp>::thread_id_t
singleton<_Tp>::f_master_thread = std::this_thread::get_id();

//----------------------------------------------------------------------------//

template <typename _Tp>
typename singleton<_Tp>::shared_pointer
singleton<_Tp>::f_master_instance = singleton<_Tp>::shared_pointer();

//----------------------------------------------------------------------------//

template <typename _Tp>
singleton<_Tp>::singleton(pointer ptr)
{
    initialize(ptr);
    if(!f_master_instance)
    {
        f_master_thread = std::this_thread::get_id();
        _local_instance().reset(new _Tp());
        f_master_instance = _local_instance();
    }
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void singleton<_Tp>::initialize()
{
    if(!f_master_instance)
    {
        f_master_thread = std::this_thread::get_id();
        _local_instance().reset(new _Tp());
        f_master_instance = _local_instance();
    }
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void singleton<_Tp>::initialize(pointer ptr)
{
    if(!f_master_instance)
    {
        f_master_thread = std::this_thread::get_id();
        _local_instance().reset(ptr);
        f_master_instance = _local_instance();
    }
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void singleton<_Tp>::destroy()
{
    _local_instance().reset();
    if(_local_instance().get() == f_master_instance.get())
        f_master_instance.reset();
}

//----------------------------------------------------------------------------//

template <typename _Tp>
typename singleton<_Tp>::shared_pointer
singleton<_Tp>::instance()
{
    if(!_local_instance())
    {
        _local_instance().reset(new _Tp());
        if(!f_master_instance)
            f_master_instance = _local_instance();
    }

    return _local_instance();
}

//----------------------------------------------------------------------------//

template <typename _Tp>
typename singleton<_Tp>::shared_pointer
singleton<_Tp>::master_instance()
{
    if(!f_master_instance)
    {
        if(!_local_instance())
            _local_instance().reset(new _Tp());
        f_master_instance = _local_instance();
    }

    return f_master_instance;
}

//============================================================================//

} // namespace tim

//============================================================================//

#endif

