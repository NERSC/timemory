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

#include "timemory/backends/dmp.hpp"
#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/operations/types/add_statistics.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/storage/types.hpp"

#include <unordered_set>

namespace tim
{
namespace operation
{
/// \struct tim::operation::call_stack
/// \tparam Tp Component type
///
/// \brief Dynamic call-stack tracking. Supports insertion in hierarchy, flat,
/// hierarchy-timeline, and flat-timeline modes.
template <typename Tp>
struct call_stack
{
    template <typename KeyT, typename MappedT>
    using uomap_t = std::unordered_map<KeyT, MappedT>;

    using this_type              = call_stack<Tp>;
    using uintvector_t           = std::vector<uint64_t>;
    using graph_node             = node::graph<Tp>;
    using graph_type             = tim::graph<graph_node>;
    using graph_data_type        = graph_data<graph_node>;
    using iterator               = typename graph_type::iterator;
    using const_iterator         = typename graph_type::const_iterator;
    using result_node            = node::result<Tp>;
    using node_type              = typename node::data<Tp>::node_type;
    using stats_type             = typename node::data<Tp>::stats_type;
    using result_type            = typename node::data<Tp>::result_type;
    using result_vector_type     = std::vector<result_node>;
    using dmp_result_vector_type = std::vector<result_vector_type>;
    using iterator_hash_submap_t = uomap_t<int64_t, iterator>;
    using iterator_hash_map_t    = uomap_t<int64_t, iterator_hash_submap_t>;
    using settings_type          = std::shared_ptr<tim::settings>;
    using graph_data_ptr_type    = std::shared_ptr<graph_data_type>;

    template <typename Up>
    using secondary_data_t = std::tuple<iterator, const std::string&, Up>;

public:
    call_stack()  = default;
    ~call_stack() = default;
    explicit call_stack(this_type*);

    TIMEMORY_DEFAULT_MOVE_ONLY_OBJECT(call_stack)

public:
    /// initialize the call-graph data. This should be done only when data will be
    /// inserted.
    graph_data_ptr_type initialize_data();

    /// pointer to the underlying \ref tim::graph
    graph_type* graph() const;

    /// pointer to the manager of the \ref tim::graph
    graph_data_ptr_type data() const;

    /// returns the current depth of the call-stack
    int64_t depth() const;

    /// decrements the current depth of the call-stack
    iterator pop();

    /// returns the latest hierarchical updated node
    iterator current();

    /// iterators keyed by hash and by depth
    iterator_hash_map_t get_node_ids() const;

    /// reset the call-stack
    void reset();

    /// store a pointer to an object that is currently on the call-stack
    void stack_push(Tp*);

    /// remove a pointer to an object after it has updated it's call-stack entry
    void stack_pop(Tp*);

    /// remove any of the pointers currently on the stack
    void stack_clear();

    /// return whether the call-stack has data
    bool empty() const;

    /// get the estimated number of entries
    size_t size() const;

    /// get the true number of entries
    size_t true_size() const;

public:
    /// generic insert based on the value of the scope configuration
    iterator insert(scope::config, const Tp&, uint64_t);

    /// insert the object into the hierarchy
    iterator insert_tree(uint64_t, const Tp&, uint64_t);

    /// insert the object into the call-graph but make it unique
    iterator insert_timeline(uint64_t, const Tp&, uint64_t);

    /// insert the object into the call-graph as a child of the head node
    iterator insert_flat(uint64_t, const Tp&, uint64_t);

    /// append a map of objects (constructed from values) as children of the current node
    template <typename Up>
    iterator append(const secondary_data_t<Up>&,
                    enable_if_t<!std::is_same<decay_t<Up>, Tp>::value, int> = 0);

    /// append a map of objects as children of the current node
    template <typename Up>
    iterator append(const secondary_data_t<Up>&,
                    enable_if_t<std::is_same<decay_t<Up>, Tp>::value, long> = 0);

    /// denote that this call-graph is a child of another call-graph or is the primary
    /// call-graph (if nullptr)
    void set_parent(this_type* _parent) { m_parent = _parent; }

    /// denote that this call-graph is the primary call-graph (redundant check)
    void set_primary(bool _v) { m_is_primary = _v; }

private:
    iterator               insert_hierarchy(uint64_t, const Tp&, uint64_t, bool);
    graph_data_type&       _data();
    const graph_data_type& _data() const;

private:
    bool                     m_is_primary       = false;
    uint32_t                 m_thread_idx       = threading::get_id();
    uint64_t                 m_timeline_counter = 1;
    this_type*               m_parent           = nullptr;
    iterator_hash_map_t      m_node_ids         = {};
    std::unordered_set<Tp*>  m_stack            = {};
    std::shared_ptr<manager> m_manager          = {};
    graph_hash_map_ptr_t     m_hash_ids         = ::tim::get_hash_ids();
    graph_hash_alias_ptr_t   m_hash_aliases     = ::tim::get_hash_aliases();
    settings_type            m_settings = tim::settings::shared_instance<TIMEMORY_API>();
    mutable graph_data_ptr_type m_graph_data_instance = nullptr;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up>
typename call_stack<Tp>::iterator
call_stack<Tp>::append(const secondary_data_t<Up>& _secondary,
                       enable_if_t<!std::is_same<decay_t<Up>, Tp>::value, int>)
{
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
        operation::add_statistics<Tp>{ _nitr->second->obj(), _stats };
        return _nitr->second;
    }

    // else, create a new entry
    auto&& _tmp = Tp{};
    _tmp += std::get<2>(_secondary);
    _tmp.set_laps(_tmp.get_laps() + 1);
    graph_node _node{ _hash, _tmp, _depth, m_thread_idx };
    _node.stats() += _tmp.get();
    auto& _stats = _node.stats();
    operation::add_statistics<Tp>{ _tmp, _stats };
    auto itr = _data().emplace_child(_itr, _node);
    itr->obj().set_iterator(itr);
    m_node_ids[_depth][_hash] = itr;
    return itr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename Up>
typename call_stack<Tp>::iterator
call_stack<Tp>::append(const secondary_data_t<Up>& _secondary,
                       enable_if_t<std::is_same<decay_t<Up>, Tp>::value, long>)
{
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
    auto&&     _tmp = std::get<2>(_secondary);
    graph_node _node{ _hash, _tmp, _depth, m_thread_idx };
    auto       itr = _data().emplace_child(_itr, _node);
    itr->obj().set_iterator(itr);
    m_node_ids[_depth][_hash] = itr;
    return itr;
}
//
}  // namespace operation
}  // namespace tim
