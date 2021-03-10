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

#ifndef TIMEMORY_OPERATIONS_TYPES_CALL_STACK_CPP_
#define TIMEMORY_OPERATIONS_TYPES_CALL_STACK_CPP_ 1

#include "timemory/operations/types/call_stack.hpp"
#include "timemory/operations/types.hpp"

#include <cassert>

namespace tim
{
namespace operation
{
//
template <typename Tp>
typename call_stack<Tp>::graph_data_ptr_type
call_stack<Tp>::initialize_data()
{
    // call initializing function, return shared_ptr
    return (_data(), m_graph_data_instance);
}
//
template <typename Tp>
typename call_stack<Tp>::graph_type*
call_stack<Tp>::graph() const
{
    return (m_graph_data_instance) ? &m_graph_data_instance->graph() : nullptr;
}
//
template <typename Tp>
typename call_stack<Tp>::graph_data_ptr_type
call_stack<Tp>::data() const
{
    return m_graph_data_instance;
}
//
template <typename Tp>
int64_t
call_stack<Tp>::depth() const
{
    return (m_graph_data_instance) ? m_graph_data_instance->depth() : 0;
}
//
template <typename Tp>
typename call_stack<Tp>::iterator
call_stack<Tp>::pop()
{
    return (m_graph_data_instance) ? m_graph_data_instance->pop_graph() : iterator{};
}
//
template <typename Tp>
typename call_stack<Tp>::iterator
call_stack<Tp>::current()
{
    return _data().current();
}
//
template <typename Tp>
typename call_stack<Tp>::iterator_hash_map_t
call_stack<Tp>::get_node_ids() const
{
    return m_node_ids;
}
//
template <typename Tp>
void
call_stack<Tp>::reset()
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
template <typename Tp>
void
call_stack<Tp>::stack_push(Tp* obj)
{
    m_stack.insert(obj);
}
//
template <typename Tp>
void
call_stack<Tp>::stack_pop(Tp* obj)
{
    auto itr = m_stack.find(obj);
    if(itr != m_stack.end())
    {
        m_stack.erase(itr);
    }
}
//
template <typename Tp>
void
call_stack<Tp>::stack_clear()
{
    if(!m_stack.empty())
    {
        std::unordered_set<Tp*> _stack = m_stack;
        for(auto& itr : _stack)
            operation::stop<Tp>{ *itr };
    }
    m_stack.clear();
}
//
template <typename Tp>
bool
call_stack<Tp>::empty() const
{
    return (graph()) ? (graph()->size() <= 1) : true;
}
//
template <typename Tp>
size_t
call_stack<Tp>::size() const
{
    return (graph()) ? (graph()->size() - 1) : 0;
}
//
template <typename Tp>
size_t
call_stack<Tp>::true_size() const
{
    if(!m_graph_data_instance)
        return 0;
    size_t _sz = _data().graph().size();
    size_t _dc = _data().dummy_count();
    return (_dc < _sz) ? (_sz - _dc) : 0;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
typename call_stack<Tp>::iterator
call_stack<Tp>::insert(scope::config scope_data, const Tp& obj, uint64_t hash_id)
{
    using force_tree_t = trait::tree_storage<Tp>;
    using force_flat_t = trait::flat_storage<Tp>;
    using force_time_t = trait::timeline_storage<Tp>;

    // if data is all the way up to the zeroth (relative) depth then worker
    // threads should insert a new dummy at the current master thread id and depth.
    // Be aware, this changes 'm_current' inside the data graph
    //
    if(!m_is_primary && _data().at_sea_level() &&
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
//----------------------------------------------------------------------------------//
//
template <typename Tp>
typename call_stack<Tp>::iterator
call_stack<Tp>::insert_tree(uint64_t hash_id, const Tp& obj, uint64_t hash_depth)
{
    // PRINT_HERE("%s", "");
    bool has_head = _data().has_head();
    return insert_hierarchy(hash_id, obj, hash_depth, has_head);
}

//----------------------------------------------------------------------------------//
//
template <typename Tp>
typename call_stack<Tp>::iterator
call_stack<Tp>::insert_timeline(uint64_t hash_id, const Tp& obj, uint64_t hash_depth)
{
    // PRINT_HERE("%s", "");
    auto       _current = _data().current();
    graph_node _node(hash_id, obj, hash_depth, m_thread_idx);
    return _data().emplace_child(_current, _node);
}

//----------------------------------------------------------------------------------//
//
template <typename Tp>
typename call_stack<Tp>::iterator
call_stack<Tp>::insert_flat(uint64_t hash_id, const Tp& obj, uint64_t hash_depth)
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
            graph_node _node{ hash_id, obj, static_cast<int64_t>(hash_depth),
                              m_thread_idx };
            auto       itr                  = _data().emplace_child(_current, _node);
            m_node_ids[hash_depth][hash_id] = itr;
            _current                        = itr;
            return itr;
        }
    }

    auto _existing = m_node_ids[hash_depth].find(hash_id);
    if(_existing != m_node_ids[hash_depth].end())
        return m_node_ids[hash_depth].find(hash_id)->second;

    graph_node _node{ hash_id, obj, static_cast<int64_t>(hash_depth), m_thread_idx };
    auto       itr                  = _data().emplace_child(_current, _node);
    m_node_ids[hash_depth][hash_id] = itr;
    return itr;
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tp>
typename call_stack<Tp>::iterator
call_stack<Tp>::insert_hierarchy(uint64_t hash_id, const Tp& obj, uint64_t hash_depth,
                                 bool has_head)
{
    using id_hash_map_t = typename iterator_hash_map_t::mapped_type;
    // PRINT_HERE("%s", "");

    auto& m_data = m_graph_data_instance;
    auto  tid    = m_thread_idx;

    // if first instance
    if(!has_head || (m_is_primary && m_node_ids.empty()))
    {
        graph_node _node{ hash_id, obj, static_cast<int64_t>(hash_depth), tid };
        auto       itr                  = m_data->append_child(_node);
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

    using sibling_itr = typename graph_type::sibling_iterator;
    graph_node _node{ hash_id, obj, m_data->depth(), tid };

    // lambda for inserting child
    auto _insert_child = [&]() {
        _node.depth() = hash_depth;
        auto itr      = m_data->append_child(_node);
        auto ditr     = m_node_ids.find(hash_depth);
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
    auto fchild = graph_type::child(current, 0);
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
template <typename Tp>
typename call_stack<Tp>::graph_data_type&
call_stack<Tp>::_data()
{
    if(m_graph_data_instance == nullptr)
    {
        if(m_parent)
        {
            auto _m = m_parent->data();
            DEBUG_PRINT_HERE("[%s]> child thread: %i, parent pointer: %p",
                             demangle<Tp>().c_str(), (int) m_thread_idx, (void*) _m);
            if(_m && _m->current())
            {
                auto       _current = _m->current();
                auto       _id      = _current->id();
                auto       _depth   = _current->depth();
                graph_node _node{ _id, operation::dummy<Tp>{}(), _depth, m_thread_idx };
                m_graph_data_instance =
                    std::make_shared<graph_data_type>(_node, _depth, _m);
                m_graph_data_instance->depth()     = _depth;
                m_graph_data_instance->sea_level() = _depth;
            }
            else
            {
                graph_node _node{ 0, operation::dummy<Tp>{}(), 1, m_thread_idx };
                m_graph_data_instance = std::make_shared<graph_data_type>(_node, 1, _m);
                m_graph_data_instance->depth()     = 1;
                m_graph_data_instance->sea_level() = 1;
            }
            m_graph_data_instance->set_parent(_m);
        }
        else
        {
            std::string _prefix = "> [tot] total";
            add_hash_id(_prefix);
            graph_node _node{ 0, operation::dummy<Tp>{}(), 0, m_thread_idx };
            m_graph_data_instance = std::make_shared<graph_data_type>(_node, 0, nullptr);
            m_graph_data_instance->depth()     = 0;
            m_graph_data_instance->sea_level() = 0;
            DEBUG_PRINT_HERE("[%s]> primary thread: %i, pointer: %p",
                             demangle<Tp>().c_str(), (int) m_thread_idx,
                             (void*) m_graph_data_instance.get());
        }

        if(m_node_ids.empty())
            m_node_ids[0][0] = m_graph_data_instance->current();
    }

    return *m_graph_data_instance;
}
//
template <typename Tp>
const typename call_stack<Tp>::graph_data_type&
call_stack<Tp>::_data() const
{
    return const_cast<this_type*>(this)->_data();
}
//
}  // namespace operation
}  // namespace tim

#endif
