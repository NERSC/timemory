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

/**
 * \file timemory/storage/declaration.hpp
 * \brief The declaration for the types for storage without definitions
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/gperf.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/call_stack.hpp"
#include "timemory/operations/types/cleanup.hpp"
#include "timemory/storage/base_storage.hpp"
#include "timemory/storage/graph.hpp"
#include "timemory/storage/graph_data.hpp"
#include "timemory/storage/macros.hpp"
#include "timemory/storage/node.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/storage/value_storage.hpp"
#include "timemory/storage/void_storage.hpp"
#include "timemory/tpls/cereal/cereal.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/singleton.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/utility/utility.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
TIMEMORY_NOINLINE TIMEMORY_NOCLONE storage_singleton<Tp>*
                                   get_storage_singleton();
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
storage_singleton<Tp>*
get_storage_singleton()
{
    using singleton_type  = tim::storage_singleton<Tp>;
    using component_type  = typename Tp::component_type;
    static auto _instance = std::unique_ptr<singleton_type>(
        (trait::runtime_enabled<component_type>::get()) ? new singleton_type{} : nullptr);
    return _instance.get();
}
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
template <typename Tp, bool>
struct storage_impl;
//
template <typename Tp>
struct storage_impl<Tp, true>
{
    using type = value_storage<Tp>;
};
//
template <typename Tp>
struct storage_impl<Tp, false>
{
    using type = void_storage<Tp>;
};
//
template <typename Tp>
using storage_impl_t = typename storage_impl<
    Tp,
    trait::uses_value_storage<Tp, typename trait::collects_data<Tp>::type>::value>::type;
//
}  // namespace impl
//
//--------------------------------------------------------------------------------------//
//
/// \class tim::storage
/// \tparam Tp Component type
///
/// \brief Responsible for maintaining the call-stack storage in timemory. This class
/// and the serialization library are responsible for most of the timemory compilation
/// time.
template <typename Tp>
class storage : public impl::storage_impl_t<Tp>
{
public:
    using Vp                                   = typename trait::collects_data<Tp>::type;
    static constexpr bool uses_value_storage_v = trait::uses_value_storage<Tp, Vp>::value;

    using this_type      = storage<Tp>;
    using base_type      = impl::storage_impl_t<Tp>;
    using deleter_type   = impl::storage_deleter<this_type>;
    using singleton_type = singleton<this_type, std::unique_ptr<this_type, deleter_type>>;
    using auto_lock_t    = typename singleton_type::auto_lock_t;
    using iterator       = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;
    using result_type    = typename base_type::result_vector_type;
    using distrib_type   = typename base_type::dmp_result_vector_type;
    using result_node    = typename base_type::result_node;
    using graph_type     = typename base_type::graph_type;
    using graph_node     = typename base_type::graph_node;

    friend class manager;
    friend class impl::void_storage<Tp>;
    friend class impl::value_storage<Tp>;
    friend struct impl::storage_deleter<this_type>;

    /// get the pointer to the storage on the current thread. Will initialize instance if
    /// one does not exist.
    static storage* instance();

    /// get the pointer to the storage on the primary thread. Will initialize instance if
    /// one does not exist.
    static storage* master_instance();

    /// get the pointer to the storage on the current thread w/o initializing if one does
    /// not exist
    static storage* noninit_instance();

    /// get the pointer to the storage on the primary thread w/o initializing if one does
    /// not exist
    static storage* noninit_master_instance();

    /// returns whether storage is finalizing on the primary thread
    static bool& master_is_finalizing() { return base_type::master_is_finalizing(); }

    /// returns whether storage is finalizing on the current thread
    static bool& worker_is_finalizing() { return base_type::worker_is_finalizing(); }

    /// returns whether storage is finalizing on any thread
    static bool is_finalizing() { return base_type::is_finalizing(); }

    /// reset the storage data
    using base_type::reset;

    /// returns whether any data has been stored
    using base_type::empty;

    /// get the current estimated number of nodes
    using base_type::size;

    /// inspect the graph and get the true number of nodes
    using base_type::true_size;

    /// get the depth of the last node which pushed to hierarchical storage. Nodes which
    /// used \ref tim::scope::flat or have \ref tim::trait::flat_storage type-trait
    /// set to true will not affect this value
    using base_type::depth;

    /// drop the current node depth and set the current node to it's parent
    using base_type::pop;

    /// insert a new node
    using base_type::insert;

    /// add a component to the stack which can be flushed if the merging or output is
    /// requested/required
    using base_type::stack_pop;

    /// remove component from the stack that will be flushed if the merging or output is
    /// requested/required
    using base_type::stack_push;

private:
    static singleton_type* get_singleton() TIMEMORY_VISIBILITY("default");
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type>::singleton_type*
storage<Type>::get_singleton()
{
    return get_storage_singleton<this_type>();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type>*
storage<Type>::instance()
{
    return get_singleton() ? get_singleton()->instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type>*
storage<Type>::master_instance()
{
    return get_singleton() ? get_singleton()->master_instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type>*
storage<Type>::noninit_instance()
{
    return get_singleton() ? get_singleton()->instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type>*
storage<Type>::noninit_master_instance()
{
    return get_singleton() ? get_singleton()->master_instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
//
//--------------------------------------------------------------------------------------//
//
template <typename StorageType>
struct storage_deleter : public std::default_delete<StorageType>
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

        // tim::dmp::barrier();

        if(ptr && master && ptr != master)
        {
            ptr->StorageType::stack_clear();
            master->StorageType::merge(ptr);
        }
        else
        {
            // sometimes the worker threads get deleted after the master thread
            // but the singleton class will ensure it is merged so we are
            // safe to leak here
            if(ptr && !master && this_tid != master_tid)
            {
                ptr->StorageType::free_shared_manager();
                ptr = nullptr;
                return;
            }

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
            {
                // ptr->StorageType::disable();
                ptr->StorageType::free_shared_manager();
            }
            delete ptr;
        }
        else
        {
            if(master && ptr != master)
                singleton_t::remove(ptr);

            if(ptr)
                ptr->StorageType::free_shared_manager();
            delete ptr;
        }

        if(_printed_master && !_deleted_master)
        {
            if(master)
            {
                // master->StorageType::disable();
                master->StorageType::free_shared_manager();
            }
            delete master;
            _deleted_master = true;
        }

        using Type = typename StorageType::component_type;
        if(_deleted_master)
            trait::runtime_enabled<Type>::set(false);
    }

    bool _printed_master = false;
    bool _deleted_master = false;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl
}  // namespace tim
