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

namespace tim
{

//============================================================================//

template <typename _Tp>
class tim_api singleton
{
public:
    typedef _Tp                 value_type;
    typedef _Tp*                pointer;
    typedef _Tp&                reference;
    typedef _Tp*&               pointer_reference;
    typedef const _Tp*          const_pointer;
    typedef const _Tp&          const_reference;
    typedef const _Tp*&         const_pointer_reference;
    typedef std::thread::id     thread_id_t;

public:
    // Constructor and Destructors
    singleton() { }
    // Virtual destructors are required by abstract classes 
    // so add it by default, just in case
    virtual ~singleton() { }

public:
    // public member function
    void initialize();
    void destroy();

    // Public functions
    static pointer instance();
    static pointer local_instance();
    static pointer master_instance();

    // for checking but not allocating
    static pointer unsafe_instance()        { return f_local_instance;  }
    static pointer unsafe_local_instance()  { return f_local_instance;  }
    static pointer unsafe_master_instance() { return f_master_instance; }

    // for when destructor is explicitly called
    static void null_instance()             { f_local_instance = nullptr; }
    static void null_local_instance()       { f_local_instance = nullptr; }
    static void null_master_instance()      { f_local_instance = nullptr; }

private:
    // Private variables
    static                  thread_id_t f_master_thread;
    static                  pointer     f_master_instance;
    tim_static_thread_local pointer     f_local_instance;

};

//============================================================================//

template <typename _Tp>
typename singleton<_Tp>::thread_id_t
singleton<_Tp>::f_master_thread = std::this_thread::get_id();

//----------------------------------------------------------------------------//

template <typename _Tp>
typename singleton<_Tp>::pointer
singleton<_Tp>::f_master_instance = nullptr;

//----------------------------------------------------------------------------//

template <typename _Tp>
tim_thread_local typename singleton<_Tp>::pointer
singleton<_Tp>::f_local_instance = nullptr;

//----------------------------------------------------------------------------//

template <typename _Tp>
void singleton<_Tp>::initialize()
{
    if(!f_master_instance)
    {
        f_master_thread = std::this_thread::get_id();
        f_master_instance = new _Tp();
        f_local_instance = f_master_instance;
    }
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void singleton<_Tp>::destroy()
{
    if(f_local_instance != f_master_instance)
    {
        delete f_local_instance;
        f_local_instance = nullptr;
    }
    else
    {
        delete f_local_instance;
        f_local_instance = nullptr;
        f_master_instance = nullptr;
    }
}

//----------------------------------------------------------------------------//

template <typename _Tp>
typename singleton<_Tp>::pointer
singleton<_Tp>::instance()
{
    if(!f_local_instance)
    {
        f_local_instance = new _Tp();
        if(!f_master_instance)
            f_master_instance = f_local_instance;
    }

    return f_local_instance;
}

//----------------------------------------------------------------------------//

template <typename _Tp>
typename singleton<_Tp>::pointer
singleton<_Tp>::local_instance()
{
    return instance();
}

//----------------------------------------------------------------------------//

template <typename _Tp>
typename singleton<_Tp>::pointer
singleton<_Tp>::master_instance()
{
    if(!f_local_instance)
    {
        f_local_instance = new _Tp();
        if(!f_master_instance)
            f_master_instance = f_local_instance;
    }

    return f_master_instance;
}

//============================================================================//

} // namespace tim

//============================================================================//

#endif

