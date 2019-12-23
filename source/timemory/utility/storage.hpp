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

/** \file utility/storage.hpp
 * \headerfile utility/storage.hpp "timemory/utility/storage.hpp"
 * Storage of the call-graph for each component. Each component has a thread-local
 * singleton that holds the call-graph. When a worker thread is deleted, it merges
 * itself back into the master thread storage. When the master thread is deleted,
 * it handles I/O (i.e. text file output, JSON output, stdout output). This class
 * needs some refactoring for clarity and probably some non-templated inheritance
 * for common features because this is probably the most convoluted piece of
 * development in the entire repository.
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/mpl/type_traits.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

//--------------------------------------------------------------------------------------//

#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

//--------------------------------------------------------------------------------------//

#include "timemory/utility/impl/storage_false.hpp"
#include "timemory/utility/impl/storage_true.hpp"

//--------------------------------------------------------------------------------------//

namespace tim
{
//======================================================================================//
//
//      front facing storage class
//
//======================================================================================//

template <typename _Tp>
class storage : public impl::storage<_Tp, implements_storage<_Tp>::value>
{
    static constexpr bool implements_storage_v = implements_storage<_Tp>::value;
    using this_type                            = storage<_Tp>;
    using base_type                            = impl::storage<_Tp, implements_storage_v>;
    using deleter_t                            = impl::storage_deleter<base_type>;
    using smart_pointer                        = std::unique_ptr<base_type, deleter_t>;
    using singleton_t                          = singleton<base_type, smart_pointer>;
    using pointer                              = typename singleton_t::pointer;
    using auto_lock_t                          = typename singleton_t::auto_lock_t;
    using iterator                             = typename base_type::iterator;
    using const_iterator                       = typename base_type::const_iterator;

    friend struct impl::storage_deleter<this_type>;
    friend class manager;
};

//--------------------------------------------------------------------------------------//
/// args:
///     1) filename
///     2) reference an object
///
template <typename _Tp>
void
generic_serialization(const std::string&, const _Tp&);

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

template <typename StorageType>
struct tim::impl::storage_deleter : public std::default_delete<StorageType>
{
    using Pointer     = std::unique_ptr<StorageType, storage_deleter<StorageType>>;
    using singleton_t = tim::singleton<StorageType, Pointer>;

    storage_deleter()  = default;
    ~storage_deleter() = default;

    void operator()(StorageType* ptr)
    {
        StorageType*    master     = singleton_t::master_instance_ptr();
        std::thread::id master_tid = singleton_t::master_thread_id();
        std::thread::id this_tid   = std::this_thread::get_id();

        tim::dmp::barrier();

        if(ptr && master && ptr != master)
        {
            ptr->StorageType::stack_clear();
            master->StorageType::merge(ptr);
        }
        else
        {
            if(ptr)
            {
                ptr->StorageType::print();
            }
            else if(master)
            {
                if(!_printed_master)
                {
                    master->StorageType::stack_clear();
                    master->StorageType::print();
                    master->StorageType::cleanup();
                    _printed_master = true;
                }
            }
        }

        if(this_tid == master_tid)
        {
            if(ptr)
                ptr->StorageType::free_shared_manager();
            delete ptr;
        }
        else
        {
            if(master && ptr != master)
            {
                singleton_t::remove(ptr);
            }
            if(ptr)
                ptr->StorageType::free_shared_manager();
            delete ptr;
        }
        if(_printed_master && !_deleted_master)
        {
            if(master)
                master->StorageType::free_shared_manager();
            delete master;
            _deleted_master = true;
        }
    }

    bool _printed_master = false;
    bool _deleted_master = false;
};

//======================================================================================//
