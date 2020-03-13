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
 * \file timemory/storage/definition.hpp
 * \brief The definitions for the types in storage
 */

#pragma once

#include "timemory/hash/declaration.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/plotting/declaration.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/utility/stream.hpp"

#include <fstream>
#include <memory>

namespace tim
{
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
        (trait::runtime_enabled<component_type>::get()) ? new singleton_type() : nullptr);
    return _instance.get();
}
//
//--------------------------------------------------------------------------------------//
//
//                              base::storage
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
//
//======================================================================================//
//
//                              impl::storage<Type, bool>
//                                  impl::storage
//
//======================================================================================//
//
//                                      TRUE
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
template <typename Type>
typename storage<Type, true>::pointer
storage<Type, true>::noninit_instance()
{
    return get_singleton() ? get_singleton()->instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::pointer
storage<Type, true>::noninit_master_instance()
{
    return get_singleton() ? get_singleton()->master_instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
storage<Type, true>::master_is_finalizing()
{
    static bool _instance = false;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
storage<Type, true>::worker_is_finalizing()
{
    static thread_local bool _instance = master_is_finalizing();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::is_finalizing()
{
    return worker_is_finalizing() || master_is_finalizing();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::atomic<int64_t>&
storage<Type, true>::instance_count()
{
    static std::atomic<int64_t> _counter{ 0 };
    return _counter;
}
//
//--------------------------------------------------------------------------------------//
//
//                                      FALSE
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
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::noninit_instance()
{
    return get_singleton() ? get_singleton()->instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, false>::pointer
storage<Type, false>::noninit_master_instance()
{
    return get_singleton() ? get_singleton()->master_instance_ptr() : nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
storage<Type, false>::master_is_finalizing()
{
    static bool _instance = false;
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool&
storage<Type, false>::worker_is_finalizing()
{
    static thread_local bool _instance = master_is_finalizing();
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, false>::is_finalizing()
{
    return worker_is_finalizing() || master_is_finalizing();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::atomic<int64_t>&
storage<Type, false>::instance_count()
{
    static std::atomic<int64_t> _counter{ 0 };
    return _counter;
}
//
//======================================================================================//
//
//                              impl::storage<Type, true>
//                                  impl::storage_true
//
//======================================================================================//
//
template <typename Type>
storage<Type, true>::storage()
: base_type(singleton_t::is_master_thread(), instance_count()++, Type::get_label())
{
    if(settings::debug())
        printf("[%s]> constructing @ %i...\n", m_label.c_str(), __LINE__);

    component::state<Type>::has_storage() = true;

    static std::atomic<int32_t> _skip_once(0);
    if(_skip_once++ > 0)
    {
        // make sure all worker instances have a copy of the hash id and aliases
        auto               _master       = singleton_t::master_instance();
        graph_hash_map_t   _hash_ids     = *_master->get_hash_ids();
        graph_hash_alias_t _hash_aliases = *_master->get_hash_aliases();
        for(const auto& itr : _hash_ids)
        {
            if(m_hash_ids->find(itr.first) == m_hash_ids->end())
                m_hash_ids->insert({ itr.first, itr.second });
        }
        for(const auto& itr : _hash_aliases)
        {
            if(m_hash_aliases->find(itr.first) == m_hash_aliases->end())
                m_hash_aliases->insert({ itr.first, itr.second });
        }
    }

    get_shared_manager();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, true>::~storage()
{
    if(settings::debug())
        printf("[%s]> destructing @ %i...\n", m_label.c_str(), __LINE__);

    if(!m_is_master)
        singleton_t::master_instance()->merge(this);

    delete m_graph_data_instance;
    m_graph_data_instance = nullptr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::initialize()
{
    if(m_initialized)
        return;
    if(settings::debug())
        printf("[%s]> initializing...\n", m_label.c_str());
    m_initialized = true;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::finalize()
{
    if(m_finalized)
        return;

    if(!m_initialized)
        return;

    if(settings::debug())
        PRINT_HERE("[%s]> finalizing...", m_label.c_str());

    m_finalized            = true;
    worker_is_finalizing() = true;
    if(m_is_master)
        master_is_finalizing() = true;

    if(m_thread_init)
        Type::thread_finalize(this);

    if(m_is_master && m_global_init)
        Type::global_finalize(this);

    if(settings::debug())
        PRINT_HERE("[%s]> finalizing...", m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::stack_clear()
{
    using Base                       = typename Type::base_type;
    std::unordered_set<Type*> _stack = m_stack;
    if(settings::stack_clearing())
        for(auto& itr : _stack)
        {
            static_cast<Base*>(itr)->stop();
            static_cast<Base*>(itr)->pop_node();
        }
    m_stack.clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::global_init()
{
    if(!m_global_init)
        return [&]() {
            m_global_init = true;
            if(!m_is_master && master_instance())
                master_instance()->global_init();
            if(m_is_master)
                Type::global_init(this);
            return m_global_init;
        }();
    return m_global_init;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::thread_init()
{
    if(!m_thread_init)
        return [&]() {
            m_thread_init = true;
            if(!m_is_master && master_instance())
                master_instance()->thread_init();
            bool _global_init = global_init();
            consume_parameters(_global_init);
            Type::thread_init(this);
            return m_thread_init;
        }();
    return m_thread_init;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::data_init()
{
    if(!m_data_init)
        return [&]() {
            m_data_init = true;
            if(!m_is_master && master_instance())
                master_instance()->data_init();
            bool _global_init = global_init();
            bool _thread_init = thread_init();
            consume_parameters(_global_init, _thread_init);
            check_consistency();
            return m_data_init;
        }();
    return m_data_init;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
const typename storage<Type, true>::graph_data_t&
storage<Type, true>::data() const
{
    if(!is_finalizing())
    {
        static thread_local auto _init = const_cast<this_type*>(this)->data_init();
        consume_parameters(_init);
    }
    return _data();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
const typename storage<Type, true>::graph_t&
storage<Type, true>::graph() const
{
    if(!is_finalizing())
    {
        static thread_local auto _init = const_cast<this_type*>(this)->data_init();
        consume_parameters(_init);
    }
    return _data().graph();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
int64_t
storage<Type, true>::depth() const
{
    if(!is_finalizing())
    {
        static thread_local auto _init = const_cast<this_type*>(this)->data_init();
        consume_parameters(_init);
    }
    return (is_finalizing()) ? 0 : _data().depth();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::graph_data_t&
storage<Type, true>::data()
{
    if(!is_finalizing())
    {
        static thread_local auto _init = data_init();
        consume_parameters(_init);
    }
    return _data();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::graph_t&
storage<Type, true>::graph()
{
    if(!is_finalizing())
    {
        static thread_local auto _init = data_init();
        consume_parameters(_init);
    }
    return _data().graph();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator&
storage<Type, true>::current()
{
    if(!is_finalizing())
    {
        static thread_local auto _init = data_init();
        consume_parameters(_init);
    }
    return _data().current();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::iterator
storage<Type, true>::pop()
{
    auto itr = _data().pop_graph();
    // if data has popped all the way up to the zeroth (relative) depth
    // then worker threads should insert a new dummy at the current
    // master thread id and depth. Be aware, this changes 'm_current' inside
    // the data graph
    //
    if(_data().at_sea_level())
        _data().add_dummy();
    return itr;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::dmp_get()
{
    dmp_result_t _ret;
    operation::finalize::dmp_get<Type, true>(*this, _ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::stack_pop(Type* obj)
{
    auto itr = m_stack.find(obj);
    if(itr != m_stack.end())
        m_stack.erase(itr);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::check_consistency()
{
    auto* ptr = &_data();
    if(ptr != m_graph_data_instance)
    {
        fprintf(stderr, "[%s]> mismatched graph data on master thread: %p vs. %p\n",
                m_label.c_str(), (void*) ptr, static_cast<void*>(m_graph_data_instance));
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::insert_init()
{
    bool _global_init = global_init();
    bool _thread_init = thread_init();
    bool _data_init   = data_init();
    consume_parameters(_global_init, _thread_init, _data_init);
    // check this now to ensure everything is initialized
    if(m_node_ids.size() == 0 || m_graph_data_instance == nullptr)
        initialize();
}
//
//--------------------------------------------------------------------------------------//
//
//                     impl::storage<Type, true>::result_node
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, true>::result_node::result_node(uint64_t _hash, const Type& _data,
                                              const string_t& _prefix, int64_t _depth,
                                              uint64_t            _rolling,
                                              const uintvector_t& _hierarchy,
                                              const stats_type&   _stats)
: base_type(_hash, _data, _prefix, _depth, _rolling, _hierarchy, _stats)
{}
//
//--------------------------------------------------------------------------------------//
//
//                     impl::storage<Type, true>::graph_node
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, true>::graph_node::graph_node()
: base_type(0, Type{}, 0, stats_type{})
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, true>::graph_node::graph_node(base_type&& _base)
: base_type(std::forward<base_type>(_base))
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, true>::graph_node::graph_node(const uint64_t& _id, const Type& _obj,
                                            int64_t _depth)
: base_type(_id, _obj, _depth, stats_type{})
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::graph_node::operator==(const graph_node& rhs) const
{
    return (id() == rhs.id() && depth() == rhs.depth());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
bool
storage<Type, true>::graph_node::operator!=(const graph_node& rhs) const
{
    return !(*this == rhs);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
Type
storage<Type, true>::graph_node::get_dummy()
{
    using object_base_t = typename Type::base_type;
    return object_base_t::dummy();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::string
storage<Type, true>::get_prefix(const graph_node& node)
{
    auto _ret = get_hash_identifier(m_hash_ids, m_hash_aliases, node.id());
    if(_ret.find("unknown-hash=") == 0)
    {
        if(!m_is_master)
        {
            auto _master = singleton_t::master_instance();
            return _master->get_prefix(node);
        }
        else
        {
            return get_hash_identifier(node.id());
        }
    }
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::string
storage<Type, true>::get_prefix(const uint64_t& id)
{
    auto _ret = get_hash_identifier(m_hash_ids, m_hash_aliases, id);
    if(_ret.find("unknown-hash=") == 0)
    {
        if(!m_is_master)
        {
            auto _master = singleton_t::master_instance();
            return _master->get_prefix(id);
        }
        else
        {
            return get_hash_identifier(id);
        }
    }
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::graph_data_t&
storage<Type, true>::_data()
{
    using object_base_t = typename Type::base_type;

    if(m_graph_data_instance == nullptr)
    {
        auto_lock_t lk(singleton_t::get_mutex(), std::defer_lock);

        if(!m_is_master && master_instance())
        {
            static bool _data_init = master_instance()->data_init();
            auto&       m          = master_instance()->data();
            consume_parameters(_data_init);

            if(!lk.owns_lock())
                lk.lock();

            if(m.current())
            {
                auto         _current = m.current();
                auto         _id      = _current->id();
                auto         _depth   = _current->depth();
                graph_node_t node(_id, object_base_t::dummy(), _depth);
                if(!m_graph_data_instance)
                    m_graph_data_instance = new graph_data_t(node, _depth, &m);
                m_graph_data_instance->depth()     = _depth;
                m_graph_data_instance->sea_level() = _depth;
            }
            else
            {
                graph_node_t node(0, object_base_t::dummy(), 0);
                if(!m_graph_data_instance)
                    m_graph_data_instance = new graph_data_t(node, 0, nullptr);
                m_graph_data_instance->depth()     = 0;
                m_graph_data_instance->sea_level() = 0;
            }
        }
        else
        {
            if(!lk.owns_lock())
                lk.lock();

            std::string _prefix = "> [tot] total";
            add_hash_id(_prefix);
            graph_node_t node(0, object_base_t::dummy(), 0);
            if(!m_graph_data_instance)
                m_graph_data_instance = new graph_data_t(node, 0, nullptr);
            m_graph_data_instance->depth()     = 0;
            m_graph_data_instance->sea_level() = 0;
        }

        if(m_node_ids.size() == 0)
            m_node_ids[0][0] = m_graph_data_instance->current();
    }

    m_initialized = true;
    return *m_graph_data_instance;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::merge()
{
    if(!m_is_master || !m_initialized)
        return;

    auto m_children = singleton_t::children();
    if(m_children.size() == 0)
        return;

    for(auto& itr : m_children)
        merge(itr);

    // create lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
    if(!l.owns_lock())
        l.lock();

    for(auto& itr : m_children)
        if(itr != this)
            itr->data().clear();

    stack_clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::merge(this_type* itr)
{
    using pre_order_iterator = typename graph_t::pre_order_iterator;
    using sibling_iterator   = typename graph_t::sibling_iterator;

    // don't merge self
    if(itr == this)
        return;

    // if merge was not initialized return
    if(itr && !itr->is_initialized())
        return;

    itr->stack_clear();

    // create lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
    if(!l.owns_lock())
        l.lock();

    auto _copy_hash_ids = [&]() {
        for(const auto& _itr : (*itr->get_hash_ids()))
            if(m_hash_ids->find(_itr.first) == m_hash_ids->end())
                (*m_hash_ids)[_itr.first] = _itr.second;
        for(const auto& _itr : (*itr->get_hash_aliases()))
            if(m_hash_aliases->find(_itr.first) == m_hash_aliases->end())
                (*m_hash_aliases)[_itr.first] = _itr.second;
    };

    // if self is not initialized but itr is, copy data
    if(itr && itr->is_initialized() && !this->is_initialized())
    {
        PRINT_HERE("[%s]> Warning! master is not initialized! Segmentation fault likely",
                   Type::get_label().c_str());
        graph().insert_subgraph_after(_data().head(), itr->data().head());
        m_initialized = itr->m_initialized;
        m_finalized   = itr->m_finalized;
        _copy_hash_ids();
        return;
    }
    else
    {
        _copy_hash_ids();
    }

    if(itr->size() == 0 || !itr->data().has_head())
        return;

    int64_t num_merged     = 0;
    auto    inverse_insert = itr->data().get_inverse_insert();

    for(auto entry : inverse_insert)
    {
        auto master_entry = data().find(entry.second);
        if(master_entry != data().end())
        {
            pre_order_iterator pitr(entry.second);

            if(itr->graph().is_valid(pitr) && pitr)
            {
                if(settings::debug() || settings::verbose() > 2)
                    PRINT_HERE("[%s]> worker is merging %i records into %i records",
                               Type::get_label().c_str(), (int) itr->size(),
                               (int) this->size());

                pre_order_iterator pos = master_entry;

                if(*pos == *pitr)
                {
                    ++num_merged;
                    sibling_iterator other = pitr;
                    for(auto sitr = other.begin(); sitr != other.end(); ++sitr)
                    {
                        pre_order_iterator pchild = sitr;
                        if(pchild->obj().nlaps() == 0)
                            continue;
                        graph().append_child(pos, pchild);
                    }
                }

                if(settings::debug() || settings::verbose() > 2)
                    PRINT_HERE("[%s]> master has %i records", Type::get_label().c_str(),
                               (int) this->size());

                // remove the entry from this graph since it has been added
                itr->graph().erase_children(entry.second);
                itr->graph().erase(entry.second);
            }
        }
    }

    int64_t merge_size = static_cast<int64_t>(inverse_insert.size());
    if(num_merged != merge_size)
    {
        int64_t           diff = merge_size - num_merged;
        std::stringstream ss;
        ss << "Testing error! Missing " << diff << " merge points. The worker thread "
           << "contained " << merge_size << " bookmarks but only merged " << num_merged
           << " nodes!";

        PRINT_HERE("%s", ss.str().c_str());

#if defined(TIMEMORY_TESTING)
        throw std::runtime_error(ss.str());
#endif
    }

    if(num_merged == 0)
    {
        if(settings::debug() || settings::verbose() > 2)
            PRINT_HERE("[%s]> worker is not merged!", Type::get_label().c_str());
        pre_order_iterator _nitr(itr->data().head());
        ++_nitr;
        if(!graph().is_valid(_nitr))
            _nitr = pre_order_iterator(itr->data().head());
        graph().append_child(_data().head(), _nitr);
    }

    itr->data().clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::result_array_t
storage<Type, true>::get()
{
    result_array_t _ret;
    operation::finalize::get<Type, true>(*this, _ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::mpi_get()
{
    dmp_result_t _ret;
    operation::finalize::mpi_get<Type, true>(*this, _ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::upc_get()
{
    dmp_result_t _ret;
    operation::finalize::upc_get<Type, true>(*this, _ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::internal_print()
{
    base::storage::stop_profiler();

    if(!m_initialized && !m_finalized)
        return;

    auto requires_json = trait::requires_json<Type>::value;
    auto label         = Type::get_label();

    if(!singleton_t::is_master(this))
    {
        singleton_t::master_instance()->merge(this);
        finalize();
    }
    else if(settings::auto_output())
    {
        merge();
        finalize();

        if(!trait::runtime_enabled<Type>::get())
        {
            instance_count().store(0);
            return;
        }

        bool _json_forced = requires_json;
        bool _file_output = settings::file_output();
        bool _cout_output = settings::cout_output();
        bool _json_output = (settings::json_output() || _json_forced) && _file_output;
        bool _text_output = settings::text_output() && _file_output;
        bool _plot_output = settings::plot_output() && _json_output;

        // if the graph wasn't ever initialized, exit
        if(!m_graph_data_instance)
        {
            instance_count().store(0);
            return;
        }

        // no entries
        if(_data().graph().size() <= 1)
        {
            instance_count().store(0);
            return;
        }

        if(!_file_output && !_cout_output && !_json_forced)
        {
            instance_count().store(0);
            return;
        }

        dmp::barrier();
        m_node_init       = dmp::is_initialized();
        m_node_rank       = dmp::rank();
        m_node_size       = dmp::size();
        auto _results     = this->get();
        auto _dmp_results = this->dmp_get();
        dmp::barrier();

        if(settings::debug())
            printf("[%s]|%i> dmp results size: %i\n", label.c_str(), m_node_rank,
                   (int) _dmp_results.size());

        if(_dmp_results.size() > 0)
        {
            if(m_node_rank != 0)
            {
                instance_count().store(0);
                return;
            }
            else
            {
                _results.clear();
                for(const auto& sitr : _dmp_results)
                {
                    for(const auto& ritr : sitr)
                        _results.push_back(ritr);
                }
            }
        }
        else if(m_node_init && m_node_rank > 0)
        {
            instance_count().store(0);
            return;
        }

#if defined(DEBUG)
        if(tim::settings::debug() && tim::settings::verbose() > 3)
        {
            printf("\n");
            size_t w = 0;
            for(const auto& itr : _results)
                w = std::max<size_t>(w, itr.prefix().length());
            for(const auto& itr : _results)
            {
                std::cout << std::setw(w) << std::left << itr.prefix() << " : "
                          << itr.data();
                auto _hierarchy = itr.hierarchy();
                for(size_t i = 0; i < _hierarchy.size(); ++i)
                {
                    if(i == 0)
                        std::cout << " :: ";
                    std::cout << get_prefix(_hierarchy[i]);
                    if(i + 1 < _hierarchy.size())
                        std::cout << "/";
                }
                std::cout << std::endl;
            }
            printf("\n");
        }
#endif

        settings::indent_width<Type, 0>(Type::get_width());
        settings::indent_width<Type, 1>(4);
        settings::indent_width<Type, 2>(4);

        int64_t _max_depth = 0;
        // find the max width
        for(const auto mitr : _dmp_results)
        {
            for(const auto& itr : mitr)
            {
                const auto& itr_obj    = itr.data();
                const auto& itr_prefix = itr.prefix();
                const auto& itr_depth  = itr.depth();
                if(itr_depth < 0 || itr_depth > settings::max_depth())
                    continue;
                _max_depth = std::max<int64_t>(_max_depth, itr_depth);
                // find global max
                settings::indent_width<Type, 0>(itr_prefix.length());
                settings::indent_width<Type, 1>(std::log10(itr_obj.nlaps()) + 1);
                settings::indent_width<Type, 2>(std::log10(itr_depth) + 1);
            }
        }

        // return type of get() function
        using get_return_type = decltype(std::declval<const Type>().get());
        using compute_type    = math::compute<get_return_type>;

        auto_lock_t slk(type_mutex<decltype(std::cout)>(), std::defer_lock);
        if(!slk.owns_lock())
            slk.lock();

        std::ofstream*       fout = nullptr;
        decltype(std::cout)* cout = nullptr;

        //--------------------------------------------------------------------------//
        // output to json file
        //
        if(_json_output)
        {
            printf("\n");
            auto is_json = (std::is_same<trait::output_archive_t<Type>,
                                         cereal::MinimalJSONOutputArchive>::value ||
                            std::is_same<trait::output_archive_t<Type>,
                                         cereal::PrettyJSONOutputArchive>::value);
            auto fext    = (is_json) ? ".json" : ".xml";
            auto jname   = settings::compose_output_filename(label, fext);
            if(jname.length() > 0)
            {
                printf("[%s]|%i> Outputting '%s'...\n", label.c_str(), m_node_rank,
                       jname.c_str());
                add_json_output(label, jname);
                {
                    using serial_write_t        = write_serialization<this_type>;
                    auto          num_instances = instance_count().load();
                    std::ofstream ofs(jname.c_str());
                    if(ofs)
                    {
                        // ensure write final block during destruction
                        // before the file is closed
                        using policy_type = policy::output_archive_t<Type>;
                        auto oa           = policy_type::get(ofs);
                        oa->setNextName("timemory");
                        oa->startNode();

                        oa->setNextName("ranks");
                        oa->startNode();
                        oa->makeArray();
                        for(uint64_t i = 0; i < _dmp_results.size(); ++i)
                        {
                            oa->startNode();
                            (*oa)(cereal::make_nvp("rank", i));
                            (*oa)(cereal::make_nvp("concurrency", num_instances));
                            serial_write_t::serialize(*this, *oa, 1, _dmp_results.at(i));
                            oa->finishNode();
                        }
                        oa->finishNode();
                        oa->finishNode();
                    }
                    if(ofs)
                        ofs << std::endl;
                    ofs.close();
                }
            }

            if(_plot_output)
            {
                // PRINT_HERE("rank = %i", m_node_rank);
                if(m_node_rank == 0)
                    plotting::plot<Type>(Type::get_label(), settings::output_path(),
                                         settings::dart_output(), jname);
            }
        }
        else if(_file_output && _text_output)
        {
            printf("\n");
        }

        //--------------------------------------------------------------------------//
        // output to text file
        //
        if(_file_output && _text_output)
        {
            auto fname = settings::compose_output_filename(label, ".txt");
            if(fname.length() > 0)
            {
                fout = new std::ofstream(fname.c_str());
                if(fout && *fout)
                {
                    printf("[%s]|%i> Outputting '%s'...\n", label.c_str(), m_node_rank,
                           fname.c_str());
                    add_text_output(label, fname);
                }
                else
                {
                    delete fout;
                    fout = nullptr;
                    fprintf(stderr, "[storage<%s>::%s @ %i]|%i> Error opening '%s'...\n",
                            label.c_str(), __FUNCTION__, __LINE__, m_node_rank,
                            fname.c_str());
                }
            }
        }

        //--------------------------------------------------------------------------//
        // output to cout
        //
        if(_cout_output)
        {
            cout = &std::cout;
            printf("\n");
        }

        auto stream_fmt   = Type::get_format_flags();
        auto stream_width = Type::get_width();
        auto stream_prec  = Type::get_precision();

        utility::stream _stream('|', '-', stream_fmt, stream_width, stream_prec);
        for(auto itr = _results.begin(); itr != _results.end(); ++itr)
        {
            auto& itr_obj    = itr->data();
            auto& itr_prefix = itr->prefix();
            auto& itr_depth  = itr->depth();
            auto  itr_laps   = itr_obj.nlaps();

            if(itr_depth < 0 || itr_depth > settings::max_depth())
                continue;

            // counts the number of non-exclusive values
            int64_t nexclusive = 0;
            // the sum of the exclusive values
            get_return_type exclusive_values{};

            // if we are not at the bottom of the call stack (i.e. completely
            // inclusive)
            if(itr_depth < _max_depth)
            {
                // get the next iteration
                auto eitr = itr;
                std::advance(eitr, 1);
                // counts the number of non-exclusive values
                nexclusive = 0;
                // the sum of the exclusive values
                exclusive_values = get_return_type{};
                // continue while not at end of graph until first sibling is
                // encountered
                if(eitr != _results.end())
                {
                    auto eitr_depth = eitr->depth();
                    while(eitr_depth != itr_depth)
                    {
                        auto& eitr_obj = eitr->data();

                        // if one level down, this is an exclusive value
                        if(eitr_depth == itr_depth + 1)
                        {
                            // if first exclusive value encountered: assign; else:
                            // combine
                            if(nexclusive == 0)
                                exclusive_values = eitr_obj.get();
                            else
                                compute_type::plus(exclusive_values, eitr_obj.get());
                            // increment. beyond 0 vs. 1, this value plays no role
                            ++nexclusive;
                        }
                        // increment iterator for next while check
                        ++eitr;
                        if(eitr == _results.end())
                            break;
                        eitr_depth = eitr->depth();
                    }
                }
            }

            auto itr_self  = compute_type::percent_diff(exclusive_values, itr_obj.get());
            auto itr_stats = itr->stats();

            bool _first = std::distance(_results.begin(), itr) == 0;
            if(_first)
                operation::print_header<Type>(itr_obj, _stream, itr_stats);

            operation::print<Type>(itr_obj, _stream, itr_prefix, itr_laps, itr_depth,
                                   itr_self, itr_stats);

            _stream.add_row();
        }

        if(cout != nullptr)
            *cout << _stream << std::flush;
        if(fout != nullptr)
            *fout << _stream << std::flush;

        if(fout)
        {
            fout->close();
            delete fout;
            fout = nullptr;
        }

        bool _dart_output = settings::dart_output();

        // if only a specific type should be echoed
        if(settings::dart_type().length() > 0)
        {
            auto dtype = settings::dart_type();
            if(operation::echo_measurement<Type>::lowercase(dtype) !=
               operation::echo_measurement<Type>::lowercase(label))
                _dart_output = false;
        }

        if(_dart_output)
        {
            printf("\n");
            uint64_t _nitr = 0;
            for(auto& itr : _results)
            {
                auto& itr_depth = itr.depth();

                if(itr_depth < 0 || itr_depth > settings::max_depth())
                    continue;

                // if only a specific number of measurements should be echoed
                if(settings::dart_count() > 0 && _nitr >= settings::dart_count())
                    continue;

                auto&       itr_obj       = itr.data();
                auto&       itr_hierarchy = itr.hierarchy();
                strvector_t str_hierarchy{};
                for(const auto& hitr : itr_hierarchy)
                    str_hierarchy.push_back(get_prefix(hitr));
                operation::echo_measurement<Type>(itr_obj, str_hierarchy);
                ++_nitr;
            }
        }
        instance_count().store(0);
    }
    else
    {
        if(singleton_t::is_master(this))
        {
            instance_count().store(0);
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::get_shared_manager()
{
    using func_t = std::function<void()>;

    // only perform this operation when not finalizing
    if(!this_type::is_finalizing())
    {
        m_manager         = tim::manager::instance();
        bool   _is_master = singleton_t::is_master(this);
        func_t _finalize  = [&]() {
            auto _instance = this_type::get_singleton();
            if(_instance)
            {
                this->stack_clear();
                _instance->reset(this);
                _instance->smart_instance().reset();
                if(_is_master)
                    _instance->smart_master_instance().reset();
            }
            trait::runtime_enabled<Type>::set(false);
        };
        m_manager->add_finalizer(Type::get_label(), std::move(_finalize), _is_master);
    }
}
//
//======================================================================================//
//
//                              impl::storage<Type, false>
//                                  impl::storage_false
//
//======================================================================================//
//
template <typename Type>
storage<Type, false>::storage()
: base_type(singleton_t::is_master_thread(), instance_count()++, Type::get_label())
{
    if(settings::debug())
        printf("[%s]> constructing @ %i...\n", m_label.c_str(), __LINE__);
    get_shared_manager();
    component::state<Type>::has_storage() = true;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, false>::~storage()
{
    if(settings::debug())
        printf("[%s]> destructing @ %i...\n", m_label.c_str(), __LINE__);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::stack_clear()
{
    using Base                       = typename Type::base_type;
    std::unordered_set<Type*> _stack = m_stack;
    for(auto& itr : _stack)
    {
        static_cast<Base*>(itr)->stop();
        static_cast<Base*>(itr)->pop_node();
    }
    m_stack.clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::initialize()
{
    if(m_initialized)
        return;

    if(settings::debug())
        printf("[%s]> initializing...\n", m_label.c_str());

    m_initialized = true;

    if(!m_is_master)
    {
        Type::thread_init(this);
    }
    else
    {
        Type::global_init(this);
        Type::thread_init(this);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::finalize()
{
    if(m_finalized)
        return;

    if(!m_initialized)
        return;

    if(settings::debug())
        printf("[%s]> finalizing...\n", m_label.c_str());

    m_finalized = true;
    if(!m_is_master)
    {
        worker_is_finalizing() = true;
        Type::thread_finalize(this);
    }
    else
    {
        master_is_finalizing() = true;
        worker_is_finalizing() = true;
        Type::thread_finalize(this);
        Type::global_finalize(this);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::stack_pop(Type* obj)
{
    auto itr = m_stack.find(obj);
    if(itr != m_stack.end())
    {
        m_stack.erase(itr);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::merge()
{
    auto m_children = singleton_t::children();
    if(m_children.size() == 0)
        return;

    if(settings::stack_clearing())
        for(auto& itr : m_children)
            merge(itr);

    stack_clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::merge(this_type* itr)
{
    itr->stack_clear();

    // create lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
    if(!l.owns_lock())
        l.lock();

    for(const auto& _itr : (*itr->get_hash_ids()))
        if(m_hash_ids->find(_itr.first) == m_hash_ids->end())
            (*m_hash_ids)[_itr.first] = _itr.second;
    for(const auto& _itr : (*itr->get_hash_aliases()))
        if(m_hash_aliases->find(_itr.first) == m_hash_aliases->end())
            (*m_hash_aliases)[_itr.first] = _itr.second;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::get_shared_manager()
{
    using func_t = std::function<void()>;

    // only perform this operation when not finalizing
    if(!this_type::is_finalizing())
    {
        m_manager         = tim::manager::instance();
        bool   _is_master = singleton_t::is_master(this);
        func_t _finalize  = [&]() {
            auto _instance = this_type::get_singleton();
            if(_instance)
            {
                this->stack_clear();
                _instance->reset(this);
            }
            trait::runtime_enabled<Type>::set(false);
        };
        m_manager->add_finalizer(Type::get_label(), std::move(_finalize), _is_master);
    }
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
