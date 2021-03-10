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
//
//--------------------------------------------------------------------------------------//
//
//                              impl::storage<Tp, true>
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
class storage<Type, true> : public base::storage
{
public:
    //----------------------------------------------------------------------------------//
    //
    static constexpr bool has_data_v = true;

    template <typename KeyT, typename MappedT>
    using uomap_t = std::unordered_map<KeyT, MappedT>;

    using result_node    = node::result<Type>;
    using graph_node     = node::graph<Type>;
    using strvector_t    = std::vector<string_t>;
    using uintvector_t   = std::vector<uint64_t>;
    using EmptyT         = std::tuple<>;
    using base_type      = base::storage;
    using component_type = Type;
    using this_type      = storage<Type, has_data_v>;
    using node_type      = typename node::data<Type>::node_type;
    using stats_type     = typename node::data<Type>::stats_type;
    using result_type    = typename node::data<Type>::result_type;
    using result_array_t = std::vector<result_node>;
    using dmp_result_t   = std::vector<result_array_t>;
    using printer_t      = operation::finalize::print<Type, has_data_v>;
    using sample_array_t = std::vector<Type>;
    using graph_node_t   = graph_node;
    using graph_data_t   = graph_data<graph_node_t>;
    using graph_t        = typename graph_data_t::graph_t;
    using graph_type     = graph_t;
    using iterator       = typename graph_type::iterator;
    using const_iterator = typename graph_type::const_iterator;
    using parent_type    = tim::storage<Type>;
    using auto_lock_t    = std::unique_lock<std::recursive_mutex>;

    template <typename Vp>
    using secondary_data_t       = std::tuple<iterator, const std::string&, Vp>;
    using iterator_hash_submap_t = uomap_t<int64_t, iterator>;
    using iterator_hash_map_t    = uomap_t<int64_t, iterator_hash_submap_t>;

    friend class tim::manager;
    friend struct node::result<Type>;
    friend struct node::graph<Type>;
    friend struct impl::storage_deleter<this_type>;
    friend struct operation::finalize::get<Type, has_data_v>;
    friend struct operation::finalize::mpi_get<Type, has_data_v>;
    friend struct operation::finalize::upc_get<Type, has_data_v>;
    friend struct operation::finalize::dmp_get<Type, has_data_v>;
    friend struct operation::finalize::print<Type, has_data_v>;
    friend struct operation::finalize::merge<Type, has_data_v>;

public:
    // static functions
    static bool& master_is_finalizing();
    static bool& worker_is_finalizing();
    static bool  is_finalizing();

private:
    static std::atomic<int64_t>& instance_count();

public:
    storage();
    ~storage() override;

    storage(const this_type&) = delete;
    storage(this_type&&)      = delete;

    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

public:
    void get_shared_manager();

    void print() final { internal_print(); }
    void cleanup() final { operation::cleanup<Type>{}; }
    void disable() final { trait::runtime_enabled<component_type>::set(false); }
    void initialize() final;
    void finalize() final;
    void stack_clear() final;
    bool global_init() final;
    bool thread_init() final;
    bool data_init() final;

    const graph_data_t& data() const { return *m_call_stack.data(); }
    const graph_t&      graph() const { return m_call_stack.data()->graph(); }
    int64_t             depth() const { return m_call_stack.depth(); }
    graph_data_t&       data() { return *m_call_stack.data(); }
    graph_t&            graph() { return m_call_stack.data()->graph(); }
    iterator&           current() { return m_call_stack.data()->current(); }

    void          reset() { m_call_stack.reset(); }
    inline bool   empty() const { return m_call_stack.empty(); }
    inline size_t size() const { return m_call_stack.size(); }
    inline size_t true_size() const { return m_call_stack.true_size(); }
    iterator      pop() { return m_call_stack.pop(); }

    result_array_t get();
    dmp_result_t   mpi_get();
    dmp_result_t   upc_get();
    dmp_result_t   dmp_get();

    template <typename Tp>
    Tp& get(Tp&);
    template <typename Tp>
    Tp& mpi_get(Tp&);
    template <typename Tp>
    Tp& upc_get(Tp&);
    template <typename Tp>
    Tp& dmp_get(Tp&);

    std::shared_ptr<printer_t> get_printer() const { return m_printer; }

    iterator_hash_map_t get_node_ids() const { return m_call_stack.get_node_ids(); }

    void stack_push(Type* _obj) { m_call_stack.stack_push(_obj); }
    void stack_pop(Type* _obj) { m_call_stack.stack_pop(_obj); }

    void insert_init();

    iterator insert(scope::config scope_data, const Type& obj, uint64_t hash_id)
    {
        return m_call_stack.insert(scope_data, obj, hash_id);
    }

    // append an instance to the graph
    template <typename Vp>
    iterator append(const secondary_data_t<Vp>& _secondary)
    {
        return m_call_stack.append(_secondary);
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned int version);

    void add_sample(Type&& _obj) { m_samples.emplace_back(std::forward<Type>(_obj)); }

    auto&       get_samples() { return m_samples; }
    const auto& get_samples() const { return m_samples; }

protected:
    void     merge();
    void     merge(this_type* itr);
    string_t get_prefix(const graph_node&);
    string_t get_prefix(iterator _node) { return get_prefix(*_node); }
    string_t get_prefix(const uint64_t& _id);

    parent_type&       get_upcast();
    const parent_type& get_upcast() const;

private:
    void check_consistency();

    template <typename Archive>
    void do_serialize(Archive& ar);

    void internal_print();

    void _data_init() const;

private:
    using call_stack_t = operation::call_stack<Type>;
    call_stack_t               m_call_stack{};
    std::shared_ptr<printer_t> m_printer;
    sample_array_t             m_samples;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
void
storage<Type, true>::serialize(Archive& ar, const unsigned int version)
{
    auto&& _results = dmp_get();
    operation::serialization<Type>{}(ar, _results);
    consume_parameters(version);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Archive>
void
storage<Type, true>::do_serialize(Archive& ar)
{
    if(m_is_master)
        merge();

    auto&& _results = dmp_get();
    operation::serialization<Type>{}(ar, _results);
}
//
//--------------------------------------------------------------------------------------//
//
//                      impl::storage<Type, false>
//                          impl::storage_false
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
class storage<Type, false> : public base::storage
{
public:
    //----------------------------------------------------------------------------------//
    //
    static constexpr bool has_data_v = false;

    using result_node    = std::tuple<>;
    using graph_node     = std::tuple<>;
    using graph_t        = std::tuple<>;
    using graph_type     = graph_t;
    using dmp_result_t   = std::vector<std::tuple<>>;
    using result_array_t = std::vector<std::tuple<>>;
    using uintvector_t   = std::vector<uint64_t>;
    using base_type      = base::storage;
    using component_type = Type;
    using this_type      = storage<Type, has_data_v>;
    using string_t       = std::string;
    using printer_t      = operation::finalize::print<Type, has_data_v>;
    using parent_type    = tim::storage<Type>;
    using auto_lock_t    = std::unique_lock<std::recursive_mutex>;

    using iterator       = void*;
    using const_iterator = const void*;

    friend class tim::manager;
    friend struct node::result<Type>;
    friend struct node::graph<Type>;
    friend struct impl::storage_deleter<this_type>;
    friend struct operation::finalize::get<Type, has_data_v>;
    friend struct operation::finalize::mpi_get<Type, has_data_v>;
    friend struct operation::finalize::upc_get<Type, has_data_v>;
    friend struct operation::finalize::dmp_get<Type, has_data_v>;
    friend struct operation::finalize::print<Type, has_data_v>;
    friend struct operation::finalize::merge<Type, has_data_v>;

public:
    static bool& master_is_finalizing();
    static bool& worker_is_finalizing();
    static bool  is_finalizing();

private:
    static std::atomic<int64_t>& instance_count();

public:
    storage();
    ~storage() override;

    explicit storage(const this_type&) = delete;
    explicit storage(this_type&&)      = delete;
    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&& rhs) = delete;

    void print() final { finalize(); }
    void cleanup() final { operation::cleanup<Type>{}; }
    void stack_clear() final;
    void disable() final { trait::runtime_enabled<component_type>::set(false); }

    void initialize() final;
    void finalize() final;

    void                             reset() {}
    TIMEMORY_NODISCARD bool          empty() const { return true; }
    TIMEMORY_NODISCARD inline size_t size() const { return 0; }
    TIMEMORY_NODISCARD inline size_t true_size() const { return 0; }
    TIMEMORY_NODISCARD inline size_t depth() const { return 0; }

    iterator pop() { return nullptr; }
    iterator insert(int64_t, const Type&, const string_t&) { return nullptr; }

    template <typename Archive>
    void serialize(Archive&, const unsigned int)
    {}

    void stack_push(Type* obj) { m_stack.insert(obj); }
    void stack_pop(Type* obj);

    TIMEMORY_NODISCARD std::shared_ptr<printer_t> get_printer() const
    {
        return m_printer;
    }

protected:
    void get_shared_manager();
    void merge();
    void merge(this_type* itr);

    parent_type&       get_upcast();
    const parent_type& get_upcast() const;

private:
    template <typename Archive>
    void do_serialize(Archive&)
    {}

private:
    std::unordered_set<Type*>  m_stack;
    std::shared_ptr<printer_t> m_printer;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl
//
//--------------------------------------------------------------------------------------//
//
/// \class tim::storage<Tp, Vp>
/// \tparam Tp Component type
/// \tparam Vp Component intermediate value type
///
/// \brief Responsible for maintaining the call-stack storage in timemory. This class
/// and the serialization library are responsible for most of the timemory compilation
/// time.
template <typename Tp>
class storage
: public impl::storage<
      Tp, trait::uses_value_storage<Tp, typename trait::collects_data<Tp>::type>::value>
{
public:
    using Vp                                   = typename trait::collects_data<Tp>::type;
    static constexpr bool uses_value_storage_v = trait::uses_value_storage<Tp, Vp>::value;
    using this_type                            = storage<Tp>;
    using base_type                            = impl::storage<Tp, uses_value_storage_v>;
    using deleter_t                            = impl::storage_deleter<this_type>;
    using smart_pointer                        = std::unique_ptr<this_type, deleter_t>;
    using singleton_t                          = singleton<this_type, smart_pointer>;
    using pointer                              = typename singleton_t::pointer;
    using auto_lock_t                          = typename singleton_t::auto_lock_t;
    using iterator                             = typename base_type::iterator;
    using const_iterator                       = typename base_type::const_iterator;
    using singleton_type                       = singleton_t;

    friend struct impl::storage_deleter<this_type>;
    friend class impl::storage<Tp, uses_value_storage_v>;
    friend class manager;

    /// get the pointer to the storage on the current thread. Will initialize instance if
    /// one does not exist.
    static pointer instance();

    /// get the pointer to the storage on the primary thread. Will initialize instance if
    /// one does not exist.
    static pointer master_instance();

    /// get the pointer to the storage on the current thread w/o initializing if one does
    /// not exist
    static pointer noninit_instance();

    /// get the pointer to the storage on the primary thread w/o initializing if one does
    /// not exist
    static pointer noninit_master_instance();

    /// returns whether storage is finalizing on the primary thread
    using base_type::master_is_finalizing;
    /// returns whether storage is finalizing on the current thread
    using base_type::worker_is_finalizing;
    /// returns whether storage is finalizing on any thread
    using base_type::is_finalizing;
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
    static singleton_t* get_singleton() { return get_storage_singleton<this_type>(); }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type>::pointer
storage<Type>::instance()
{
    return get_singleton() ? get_singleton()->instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type>::pointer
storage<Type>::master_instance()
{
    return get_singleton() ? get_singleton()->master_instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type>::pointer
storage<Type>::noninit_instance()
{
    return get_singleton() ? get_singleton()->instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type>::pointer
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
