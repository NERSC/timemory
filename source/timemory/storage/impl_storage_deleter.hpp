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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "timemory/mpl/type_traits.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/singleton.hpp"

#include <memory>
#include <thread>
#include <type_traits>

namespace tim
{
namespace impl
{
template <typename StorageT>
struct storage_deleter : public std::default_delete<StorageT>
{
    using pointer_t   = std::unique_ptr<StorageT, storage_deleter<StorageT>>;
    using singleton_t = tim::singleton<StorageT, pointer_t>;

    storage_deleter()  = default;
    ~storage_deleter() = default;

    void operator()(StorageT* ptr)
    {
        // if(ptr == nullptr)
        //    return;

        StorageT*       master     = singleton_t::master_instance_ptr();
        std::thread::id master_tid = singleton_t::master_thread_id();
        std::thread::id this_tid   = std::this_thread::get_id();

        static_assert(!std::is_same<StorageT, tim::base::storage>::value,
                      "Error! Base class");
        // tim::dmp::barrier();

        if(ptr && master && ptr != master)
        {
            ptr->StorageT::stack_clear();
            master->StorageT::merge(ptr);
        }
        else
        {
            // sometimes the worker threads get deleted after the master thread
            // but the singleton class will ensure it is merged so we are
            // safe to leak here
            if(ptr && !master && this_tid != master_tid)
            {
                ptr->StorageT::free_shared_manager();
                ptr = nullptr;
                return;
            }

            if(ptr)
            {
                ptr->StorageT::print();
            }
            else if(master)
            {
                if(!_printed_master)
                {
                    master->StorageT::stack_clear();
                    master->StorageT::print();
                    master->StorageT::cleanup();
                    _printed_master = true;
                }
            }
        }

        if(this_tid == master_tid)
        {
            if(ptr)
            {
                // ptr->StorageT::disable();
                ptr->StorageT::free_shared_manager();
            }
            delete ptr;
            ptr = nullptr;
        }
        else
        {
            if(master && ptr != master)
                singleton_t::remove(ptr);

            if(ptr)
                ptr->StorageT::free_shared_manager();
            delete ptr;
            ptr = nullptr;
        }

        if(_printed_master && !_deleted_master)
        {
            if(master)
            {
                // master->StorageT::disable();
                master->StorageT::free_shared_manager();
            }
            delete master;
            master          = nullptr;
            _deleted_master = true;
        }

        using Type = typename StorageT::component_type;
        if(_deleted_master)
            trait::runtime_enabled<Type>::set(false);
    }

    bool _printed_master = false;
    bool _deleted_master = false;
};
}  // namespace impl
}  // namespace tim
