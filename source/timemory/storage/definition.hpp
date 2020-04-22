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

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
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
    m_printer = std::make_shared<printer_t>(Type::get_label(), this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, true>::~storage()
{
    component::state<Type>::has_storage() = false;

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
    manager::instance()->is_finalizing(true);

    auto upcast = static_cast<tim::storage<Type, typename Type::value_type>*>(this);

    if(m_thread_init)
        Type::thread_finalize(upcast);

    if(m_is_master && m_global_init)
        Type::global_finalize(upcast);

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
            auto upcast =
                static_cast<tim::storage<Type, typename Type::value_type>*>(this);
            if(m_is_master)
                Type::global_init(upcast);
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
            auto upcast =
                static_cast<tim::storage<Type, typename Type::value_type>*>(this);
            Type::thread_init(upcast);
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
                graph_node_t node(_id, object_base_t::dummy(), _depth, m_thread_idx);
                if(!m_graph_data_instance)
                    m_graph_data_instance = new graph_data_t(node, _depth, &m);
                m_graph_data_instance->depth()     = _depth;
                m_graph_data_instance->sea_level() = _depth;
            }
            else
            {
                graph_node_t node(0, object_base_t::dummy(), 0, m_thread_idx);
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
            graph_node_t node(0, object_base_t::dummy(), 0, m_thread_idx);
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
    if(itr)
        operation::finalize::merge<Type, true>(*this, *itr);
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

        if(!m_printer)
            m_printer.reset(new printer_t(Type::get_label(), this));

        m_printer->execute();

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
        auto _label = Type::label();
        for(auto& itr : _label)
            itr = toupper(itr);
        std::stringstream env_var;
        env_var << "TIMEMORY_" << _label << "_ENABLED";
        auto _enabled = tim::get_env<bool>(env_var.str(), true);
        trait::runtime_enabled<Type>::set(_enabled);

        m_manager         = tim::manager::instance();
        bool   _is_master = singleton_t::is_master(this);
        auto   _cleanup   = [&]() {};
        func_t _finalize  = [&]() {
            auto _instance = this_type::get_singleton();
            if(_instance)
            {
                if(settings::debug() || settings::verbose() > 1)
                    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                               "calling _instance->reset(this)");
                _instance->reset(this);
                if(settings::debug() || settings::verbose() > 1)
                    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                               "calling _instance->smart_instance().reset()");
                _instance->smart_instance().reset();
                if(_is_master)
                {
                    if(settings::debug() || settings::verbose() > 1)
                        PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                                   "calling _instance->smart_master_instance().reset()");
                    _instance->smart_master_instance().reset();
                }
            }
            trait::runtime_enabled<Type>::set(false);
        };

        m_manager->add_finalizer(Type::get_label(), std::move(_cleanup),
                                 std::move(_finalize), _is_master);
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
    // m_printer = std::make_shared<printer_t>(Type::get_label(), this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, false>::~storage()
{
    component::state<Type>::has_storage() = false;
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

    auto upcast = static_cast<tim::storage<Type, typename Type::value_type>*>(this);

    if(!m_is_master)
    {
        Type::thread_init(upcast);
    }
    else
    {
        Type::global_init(upcast);
        Type::thread_init(upcast);
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

    auto upcast = static_cast<tim::storage<Type, typename Type::value_type>*>(this);

    m_finalized = true;
    manager::instance()->is_finalizing(true);
    if(!m_is_master)
    {
        worker_is_finalizing() = true;
        Type::thread_finalize(upcast);
    }
    else
    {
        master_is_finalizing() = true;
        worker_is_finalizing() = true;
        Type::thread_finalize(upcast);
        Type::global_finalize(upcast);
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
    if(itr)
        operation::finalize::merge<Type, false>(*this, *itr);
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
        auto _label = Type::label();
        for(auto& itr : _label)
            itr = toupper(itr);
        std::stringstream env_var;
        env_var << "TIMEMORY_" << _label << "_ENABLED";
        auto _enabled = tim::get_env<bool>(env_var.str(), true);
        trait::runtime_enabled<Type>::set(_enabled);

        m_manager         = tim::manager::instance();
        bool   _is_master = singleton_t::is_master(this);
        auto   _cleanup   = [&]() {};
        func_t _finalize  = [&]() {
            auto _instance = this_type::get_singleton();
            if(_instance)
            {
                if(settings::debug() || settings::verbose() > 1)
                    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                               "calling _instance->reset(this)");
                _instance->reset(this);
                if(settings::debug() || settings::verbose() > 1)
                    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                               "calling _instance->smart_instance().reset()");
                _instance->smart_instance().reset();
                if(_is_master)
                {
                    if(settings::debug() || settings::verbose() > 1)
                        PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                                   "calling _instance->smart_master_instance().reset()");
                    _instance->smart_master_instance().reset();
                }
            }
            trait::runtime_enabled<Type>::set(false);
        };

        m_manager->add_finalizer(Type::get_label(), std::move(_cleanup),
                                 std::move(_finalize), _is_master);
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
