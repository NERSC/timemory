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
#include "timemory/operations/types/cleanup.hpp"
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
    static auto _dtor = scope::destructor{ []() { _instance.reset(); } };
    return _instance.get();
    consume_parameters(_dtor);
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
    using smart_pointer  = std::unique_ptr<this_type, impl::storage_deleter<this_type>>;
    using singleton_t    = singleton<this_type, smart_pointer>;
    using singleton_type = singleton_t;
    using pointer        = typename singleton_t::pointer;
    using auto_lock_t    = typename singleton_t::auto_lock_t;
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
    static pointer instance();
    static pointer master_instance();
    static pointer noninit_instance();
    static pointer noninit_master_instance();

    static bool& master_is_finalizing();
    static bool& worker_is_finalizing();
    static bool  is_finalizing();

private:
    static singleton_t* get_singleton() { return get_storage_singleton<this_type>(); }
    static std::atomic<int64_t>& instance_count();

public:
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

    const graph_data_t& data() const;
    const graph_t&      graph() const;
    int64_t             depth() const;
    graph_data_t&       data();
    graph_t&            graph();
    iterator&           current();

    void        reset();
    inline bool empty() const
    {
        return (m_graph_data_instance) ? (_data().graph().size() <= 1) : true;
    }
    inline size_t size() const
    {
        return (m_graph_data_instance) ? (_data().graph().size() - 1) : 0;
    }
    inline size_t true_size() const
    {
        if(!m_graph_data_instance)
            return 0;
        size_t _sz = _data().graph().size();
        size_t _dc = _data().dummy_count();
        return (_dc < _sz) ? (_sz - _dc) : 0;
    }
    iterator       pop();
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

    iterator_hash_map_t get_node_ids() const { return m_node_ids; }

    void stack_push(Type* obj) { m_stack.insert(obj); }
    void stack_pop(Type* obj);

    void insert_init();

    iterator insert(scope::config scope_data, const Type& obj, uint64_t hash_id);

    // append a value to the the graph
    template <typename Vp, enable_if_t<!std::is_same<decay_t<Vp>, Type>::value, int> = 0>
    iterator append(const secondary_data_t<Vp>& _secondary);

    // append an instance to the graph
    template <typename Vp, enable_if_t<std::is_same<decay_t<Vp>, Type>::value, int> = 0>
    iterator append(const secondary_data_t<Vp>& _secondary);

    template <typename Archive>
    void serialize(Archive& ar, unsigned int version);

    void add_sample(Type&& _obj) { m_samples.emplace_back(std::forward<Type>(_obj)); }

    auto&       get_samples() { return m_samples; }
    const auto& get_samples() const { return m_samples; }

protected:
    iterator insert_tree(uint64_t hash_id, const Type& obj, uint64_t hash_depth);
    iterator insert_timeline(uint64_t hash_id, const Type& obj, uint64_t hash_depth);
    iterator insert_flat(uint64_t hash_id, const Type& obj, uint64_t hash_depth);
    iterator insert_hierarchy(uint64_t hash_id, const Type& obj, uint64_t hash_depth,
                              bool has_head);

    void     merge();
    void     merge(this_type* itr);
    string_t get_prefix(const graph_node&);
    string_t get_prefix(iterator _node) { return get_prefix(*_node); }
    string_t get_prefix(const uint64_t& _id);

private:
    void check_consistency();

    template <typename Archive>
    void do_serialize(Archive& ar);

    void internal_print();

    graph_data_t&       _data();
    const graph_data_t& _data() const
    {
        using type_t = decay_t<remove_pointer_t<decltype(this)>>;
        return const_cast<type_t*>(this)->_data();
    }

private:
    uint64_t                   m_timeline_counter    = 1;
    mutable graph_data_t*      m_graph_data_instance = nullptr;
    iterator_hash_map_t        m_node_ids;
    std::unordered_set<Type*>  m_stack;
    std::shared_ptr<printer_t> m_printer;
    sample_array_t             m_samples;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::reset()
{
    // have the data graph erase all children of the head node
    if(m_graph_data_instance)
        m_graph_data_instance->reset();
    // erase all the cached iterators except for m_node_ids[0][0]
    for(auto& ditr : m_node_ids)
    {
        auto _depth = ditr.first;
        if(_depth != 0)
        {
            ditr.second.clear();
        }
        else
        {
            for(auto itr = ditr.second.begin(); itr != ditr.second.end(); ++itr)
            {
                if(itr->first != 0)
                    ditr.second.erase(itr);
            }
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert(scope::config scope_data, const Type& obj, uint64_t hash_id)
{
    insert_init();

    using force_tree_t = trait::tree_storage<Type>;
    using force_flat_t = trait::flat_storage<Type>;
    using force_time_t = trait::timeline_storage<Type>;

    // if data is all the way up to the zeroth (relative) depth then worker
    // threads should insert a new dummy at the current master thread id and depth.
    // Be aware, this changes 'm_current' inside the data graph
    //
    if(!m_is_master && _data().at_sea_level() &&
       _data().dummy_count() < m_settings->get_max_thread_bookmarks())
        _data().add_dummy();

    // compute the insertion depth
    auto hash_depth = scope_data.compute_depth<force_tree_t, force_flat_t, force_time_t>(
        _data().depth());

    // compute the insertion key
    auto hash_value = scope_data.compute_hash<force_tree_t, force_flat_t, force_time_t>(
        hash_id, hash_depth, m_timeline_counter);

    // alias the true id with the insertion key
    add_hash_id(hash_id, hash_value);

    // even when flat is combined with timeline, it still inserts at depth of 1
    // so this is easiest check
    if(scope_data.is_flat() || force_flat_t::value)
        return insert_flat(hash_value, obj, hash_depth);

    // in the case of tree + timeline, timeline will have appropriately modified the
    // depth and hash so it doesn't really matter which check happens first here
    // however, the query for is_timeline() is cheaper so we will check that
    // and fallback to inserting into tree without a check
    // if(scope_data.is_timeline())
    //    return insert_timeline(hash_value, obj, hash_depth);

    // default fall-through if neither flat nor timeline
    return insert_tree(hash_value, obj, hash_depth);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Vp, enable_if_t<!std::is_same<decay_t<Vp>, Type>::value, int>>
typename storage<Type, true>::iterator
storage<Type, true>::append(const secondary_data_t<Vp>& _secondary)
{
    insert_init();

    // get the iterator and check if valid
    auto&& _itr = std::get<0>(_secondary);
    if(!_data().graph().is_valid(_itr))
        return nullptr;

    // compute hash of prefix
    auto _hash_id = add_hash_id(std::get<1>(_secondary));
    // compute hash w.r.t. parent iterator (so identical kernels from different
    // call-graph parents do not locate same iterator)
    auto _hash = _hash_id ^ _itr->id();
    // add the hash alias
    add_hash_id(_hash_id, _hash);
    // compute depth
    auto _depth = _itr->depth() + 1;

    // see if depth + hash entry exists already
    auto _nitr = m_node_ids[_depth].find(_hash);
    if(_nitr != m_node_ids[_depth].end())
    {
        // if so, then update
        auto& _obj = _nitr->second->obj();
        _obj += std::get<2>(_secondary);
        _obj.set_laps(_nitr->second->obj().get_laps() + 1);
        auto& _stats = _nitr->second->stats();
        operation::add_statistics<Type>(_nitr->second->obj(), _stats);
        return _nitr->second;
    }

    // else, create a new entry
    auto&& _tmp = Type{};
    _tmp += std::get<2>(_secondary);
    _tmp.set_laps(_tmp.get_laps() + 1);
    graph_node_t _node{ _hash, _tmp, _depth, m_thread_idx };
    _node.stats() += _tmp.get();
    auto& _stats = _node.stats();
    operation::add_statistics<Type>(_tmp, _stats);
    auto itr = _data().emplace_child(_itr, std::move(_node));
    itr->obj().set_iterator(itr);
    m_node_ids[_depth][_hash] = itr;
    return itr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Vp, enable_if_t<std::is_same<decay_t<Vp>, Type>::value, int>>
typename storage<Type, true>::iterator
storage<Type, true>::append(const secondary_data_t<Vp>& _secondary)
{
    insert_init();

    // get the iterator and check if valid
    auto&& _itr = std::get<0>(_secondary);
    if(!_data().graph().is_valid(_itr))
        return nullptr;

    // compute hash of prefix
    auto _hash_id = add_hash_id(std::get<1>(_secondary));
    // compute hash w.r.t. parent iterator (so identical kernels from different
    // call-graph parents do not locate same iterator)
    auto _hash = _hash_id ^ _itr->id();
    // add the hash alias
    add_hash_id(_hash_id, _hash);
    // compute depth
    auto _depth = _itr->depth() + 1;

    // see if depth + hash entry exists already
    auto _nitr = m_node_ids[_depth].find(_hash);
    if(_nitr != m_node_ids[_depth].end())
    {
        _nitr->second->obj() += std::get<2>(_secondary);
        return _nitr->second;
    }

    // else, create a new entry
    auto&& _tmp = std::get<2>(_secondary);
    auto   itr  = _data().emplace_child(
        _itr, graph_node_t{ _hash, _tmp, static_cast<int64_t>(_depth), m_thread_idx });
    itr->obj().set_iterator(itr);
    m_node_ids[_depth][_hash] = itr;
    return itr;
}
//
//----------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_tree(uint64_t hash_id, const Type& obj, uint64_t hash_depth)
{
    // PRINT_HERE("%s", "");
    bool has_head = _data().has_head();
    return insert_hierarchy(hash_id, obj, hash_depth, has_head);
}

//----------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_timeline(uint64_t hash_id, const Type& obj,
                                     uint64_t hash_depth)
{
    // PRINT_HERE("%s", "");
    auto _current = _data().current();
    return _data().emplace_child(
        _current,
        graph_node_t{ hash_id, obj, static_cast<int64_t>(hash_depth), m_thread_idx });
}

//----------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_flat(uint64_t hash_id, const Type& obj, uint64_t hash_depth)
{
    // PRINT_HERE("%s", "");
    static thread_local auto _current = _data().head();
    static thread_local bool _first   = true;
    if(_first)
    {
        _first = false;
        if(_current.begin())
        {
            _current = _current.begin();
        }
        else
        {
            auto itr = _data().emplace_child(
                _current, graph_node_t{ hash_id, obj, static_cast<int64_t>(hash_depth),
                                        m_thread_idx });
            m_node_ids[hash_depth][hash_id] = itr;
            _current                        = itr;
            return itr;
        }
    }

    auto _existing = m_node_ids[hash_depth].find(hash_id);
    if(_existing != m_node_ids[hash_depth].end())
        return m_node_ids[hash_depth].find(hash_id)->second;

    auto itr = _data().emplace_child(
        _current,
        graph_node_t{ hash_id, obj, static_cast<int64_t>(hash_depth), m_thread_idx });
    m_node_ids[hash_depth][hash_id] = itr;
    return itr;
}
//
//----------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::insert_hierarchy(uint64_t hash_id, const Type& obj,
                                      uint64_t hash_depth, bool has_head)
{
    using id_hash_map_t = typename iterator_hash_map_t::mapped_type;
    // PRINT_HERE("%s", "");

    auto& m_data = m_graph_data_instance;
    auto  tid    = m_thread_idx;

    // if first instance
    if(!has_head || (m_is_master && m_node_ids.empty()))
    {
        auto itr = m_data->append_child(
            graph_node_t{ hash_id, obj, static_cast<int64_t>(hash_depth), tid });
        m_node_ids[hash_depth][hash_id] = itr;
        return itr;
    }

    // lambda for updating settings
    auto _update = [&](iterator itr) {
        m_data->depth() = itr->depth();
        return (m_data->current() = itr);
    };

    if(m_node_ids[hash_depth].find(hash_id) != m_node_ids[hash_depth].end() &&
       m_node_ids[hash_depth].find(hash_id)->second->depth() == m_data->depth())
    {
        return _update(m_node_ids[hash_depth].find(hash_id)->second);
    }

    using sibling_itr = typename graph_t::sibling_iterator;
    graph_node_t node{ hash_id, obj, m_data->depth(), tid };

    // lambda for inserting child
    auto _insert_child = [&]() {
        node.depth() = hash_depth;
        auto itr     = m_data->append_child(std::move(node));
        auto ditr    = m_node_ids.find(hash_depth);
        if(ditr == m_node_ids.end())
            m_node_ids.insert({ hash_depth, id_hash_map_t{} });
        auto hitr = m_node_ids.at(hash_depth).find(hash_id);
        if(hitr == m_node_ids.at(hash_depth).end())
            m_node_ids.at(hash_depth).insert({ hash_id, iterator{} });
        m_node_ids.at(hash_depth).at(hash_id) = itr;
        return itr;
    };

    auto current = m_data->current();
    if(!m_data->graph().is_valid(current))
        _insert_child();

    // check children first because in general, child match is ideal
    auto fchild = graph_t::child(current, 0);
    if(m_data->graph().is_valid(fchild))
    {
        for(sibling_itr itr = fchild.begin(); itr != fchild.end(); ++itr)
        {
            if((hash_id) == itr->id())
                return _update(itr);
        }
    }

    // occasionally, we end up here because of some of the threading stuff that
    // has to do with the head node. Protected against mis-matches in hierarchy
    // because the actual hash includes the depth so "example" at depth 2
    // has a different hash than "example" at depth 3.
    if((hash_id) == current->id())
        return current;

    // check siblings
    for(sibling_itr itr = current.begin(); itr != current.end(); ++itr)
    {
        // skip if current
        if(itr == current)
            continue;
        // check hash id's
        if((hash_id) == itr->id())
            return _update(itr);
    }

    return _insert_child();
}

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
template <typename Type>
typename storage<Type, true>::pointer
storage<Type, true>::instance()
{
    return get_singleton() ? get_singleton()->instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::pointer
storage<Type, true>::master_instance()
{
    return get_singleton() ? get_singleton()->master_instance() : nullptr;
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
    using smart_pointer  = std::unique_ptr<this_type, impl::storage_deleter<this_type>>;
    using singleton_t    = singleton<this_type, smart_pointer>;
    using singleton_type = singleton_t;
    using pointer        = typename singleton_t::pointer;
    using auto_lock_t    = typename singleton_t::auto_lock_t;
    using printer_t      = operation::finalize::print<Type, has_data_v>;

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
    static pointer instance();
    static pointer master_instance();
    static pointer noninit_instance();
    static pointer noninit_master_instance();

    static bool& master_is_finalizing();
    static bool& worker_is_finalizing();
    static bool  is_finalizing();

private:
    static singleton_t* get_singleton() { return get_storage_singleton<this_type>(); }
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
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::instance()
{
    return get_singleton() ? get_singleton()->instance() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::master_instance()
{
    return get_singleton() ? get_singleton()->master_instance() : nullptr;
}
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
template <typename Tp, typename Vp>
class storage : public impl::storage<Tp, trait::uses_value_storage<Tp, Vp>::value>
{
public:
    static constexpr bool uses_value_storage_v = trait::uses_value_storage<Tp, Vp>::value;
    using this_type                            = storage<Tp, Vp>;
    using base_type                            = impl::storage<Tp, uses_value_storage_v>;
    using deleter_t                            = impl::storage_deleter<base_type>;
    using smart_pointer                        = std::unique_ptr<base_type, deleter_t>;
    using singleton_t                          = singleton<base_type, smart_pointer>;
    using pointer                              = typename singleton_t::pointer;
    using auto_lock_t                          = typename singleton_t::auto_lock_t;
    using iterator                             = typename base_type::iterator;
    using const_iterator                       = typename base_type::const_iterator;

    friend struct impl::storage_deleter<this_type>;
    friend class manager;

    /// get the pointer to the storage on the current thread. Will initialize instance if
    /// one does not exist.
    using base_type::instance;
    /// get the pointer to the storage on the primary thread. Will initialize instance if
    /// one does not exist.
    using base_type::master_instance;
    /// get the pointer to the storage on the current thread w/o initializing if one does
    /// not exist
    using base_type::noninit_instance;
    /// get the pointer to the storage on the primary thread w/o initializing if one does
    /// not exist
    using base_type::noninit_master_instance;
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
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
class storage<Tp, type_list<>>
: public storage<
      Tp, conditional_t<trait::is_available<Tp>::value, typename Tp::value_type, void>>
{
public:
    using Vp =
        conditional_t<trait::is_available<Tp>::value, typename Tp::value_type, void>;
    static constexpr bool uses_value_storage_v = trait::uses_value_storage<Tp, Vp>::value;
    using this_type                            = storage<Tp, Vp>;
    using base_type                            = impl::storage<Tp, uses_value_storage_v>;
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
        // if(ptr == nullptr)
        //    return;

        StorageType*    master     = singleton_t::master_instance_ptr();
        std::thread::id master_tid = singleton_t::master_thread_id();
        std::thread::id this_tid   = std::this_thread::get_id();

        static_assert(!std::is_same<StorageType, tim::base::storage>::value,
                      "Error! Base class");
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
            ptr = nullptr;
        }
        else
        {
            if(master && ptr != master)
                singleton_t::remove(ptr);

            if(ptr)
                ptr->StorageType::free_shared_manager();
            delete ptr;
            ptr = nullptr;
        }

        if(_printed_master && !_deleted_master)
        {
            if(master)
            {
                // master->StorageType::disable();
                master->StorageType::free_shared_manager();
            }
            delete master;
            master          = nullptr;
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
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Vp>
inline base::storage*
base::storage::base_instance()
{
    using storage_type = tim::storage<Tp, Vp>;

    // thread-local variable
    static thread_local base::storage* _ret = nullptr;

    // return nullptr is disabled
    if(!trait::runtime_enabled<Tp>::get())
        return nullptr;

    // if nullptr, try to get instance
    if(_ret == nullptr)
    {
        // thread will copy the hash-table so use a lock here
        auto_lock_t lk(type_mutex<base::storage>());
        _ret = static_cast<base::storage*>(storage_type::instance());
    }

    // return pointer
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
