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

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/data/stream.hpp"
#include "timemory/hash/declaration.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/operations/types/decode.hpp"
#include "timemory/operations/types/fini.hpp"
#include "timemory/operations/types/init.hpp"
#include "timemory/operations/types/node.hpp"
#include "timemory/operations/types/start.hpp"
#include "timemory/operations/types/stop.hpp"
#include "timemory/plotting/declaration.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/storage/macros.hpp"
#include "timemory/storage/types.hpp"

#include <fstream>
#include <memory>
#include <utility>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              base::storage
//
//--------------------------------------------------------------------------------------//
//
namespace base
{
//
#if !defined(TIMEMORY_STORAGE_HIDE_DEFINITION)
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_LINKAGE
storage::storage(bool _is_master, int64_t _instance_id, std::string _label)
: m_is_master(_is_master)
, m_instance_id(_instance_id)
, m_label(std::move(_label))
, m_manager(::tim::manager::instance())
, m_settings(::tim::settings::shared_instance())
{
    if(m_is_master && m_instance_id > 0)
    {
        int _id = m_instance_id;
        PRINT_HERE("%s: %i (%s)",
                   "Error! base::storage is master but is not zero instance", _id,
                   m_label.c_str());
        if(m_instance_id > 10)
        {
            // at this point we have a recursive loop
            TIMEMORY_EXCEPTION("Duplication!")
        }
    }

    if(!m_is_master && m_instance_id == 0)
    {
        int _id = m_instance_id;
        PRINT_HERE("%s: %i (%s)",
                   "Warning! base::storage is not master but is zero instance", _id,
                   m_label.c_str());
    }

    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "%s: %i (%s)",
                           "base::storage instance created", (int) m_instance_id,
                           m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_LINKAGE storage::~storage()
{
    CONDITIONAL_PRINT_HERE(m_settings->get_debug(),
                           "base::storage instance %i deleted for %s",
                           (int) m_instance_id, m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_LINKAGE std::atomic<int>&
                         storage::storage_once_flag()
{
    static std::atomic<int> _instance(0);
    return _instance;
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_LINKAGE void
storage::stop_profiler()
{
    // disable gperf if profiling
#    if defined(TIMEMORY_USE_GPERFTOOLS) || defined(TIMEMORY_USE_GPERFTOOLS_PROFILER) || \
        defined(TIMEMORY_USE_GPERFTOOLS_TCMALLOC)
    try
    {
        if(storage_once_flag()++ == 0)
            gperftools::profiler_stop();
    } catch(std::exception& e)
    {
        std::cerr << "Error calling gperftools::profiler_stop(): " << e.what()
                  << ". Continuing..." << std::endl;
    }
#    endif
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_LINKAGE void
storage::add_hash_id(uint64_t _lhs, uint64_t _rhs)
{
    ::tim::add_hash_id(m_hash_ids, m_hash_aliases, _lhs, _rhs);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_LINKAGE hash_value_t
                         storage::add_hash_id(const std::string& _prefix)
{
    return ::tim::add_hash_id(m_hash_ids, _prefix);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_LINKAGE void
storage::add_file_output(const std::string& _category, const std::string& _label,
                         const std::string& _file)
{
    if(m_manager)
        m_manager->add_file_output(_category, _label, _file);
}
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STORAGE_LINKAGE void
storage::free_shared_manager()
{
    if(m_manager)
        m_manager->remove_finalizer(m_label);
}
//
#endif
//--------------------------------------------------------------------------------------//
//
}  // namespace base
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
: base_type(singleton_t::is_master_thread(), instance_count()++, demangle<Type>())
{
    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "constructing %s", m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() > 1, 16);

    component::state<Type>::has_storage() = true;

    static std::atomic<int32_t> _skip_once(0);
    if(_skip_once++ > 0)
    {
        // make sure all worker instances have a copy of the hash id and aliases
        auto _master = singleton_t::master_instance();
        if(_master)
        {
            hash_map_t       _hash_ids     = *_master->get_hash_ids();
            hash_alias_map_t _hash_aliases = *_master->get_hash_aliases();
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
    }

    get_shared_manager();
    // m_printer = std::make_shared<printer_t>(Type::get_label(), this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
storage<Type, true>::~storage()
{
    component::state<Type>::has_storage() = false;

    auto _debug = m_settings->get_debug();

    CONDITIONAL_PRINT_HERE(_debug, "[%s|%li]> destroying storage", m_label.c_str(),
                           (long) m_instance_id);

    auto _main_instance = singleton_t::master_instance();

    if(!m_is_master)
    {
        if(_main_instance)
        {
            CONDITIONAL_PRINT_HERE(_debug, "[%s|%li]> merging into primary instance",
                                   m_label.c_str(), (long) m_instance_id);
            _main_instance->merge(this);
        }
        else
        {
            CONDITIONAL_PRINT_HERE(
                _debug, "[%s|%li]> skipping merge into non-existent primary instance",
                m_label.c_str(), (long) m_instance_id);
        }
    }

    if(m_graph_data_instance)
    {
        CONDITIONAL_PRINT_HERE(_debug, "[%s|%li]> deleting graph data", m_label.c_str(),
                               (long) m_instance_id);
        delete m_graph_data_instance;
    }

    m_graph_data_instance = nullptr;

    CONDITIONAL_PRINT_HERE(_debug, "[%s|%li]> storage destroyed", m_label.c_str(),
                           (long) m_instance_id);
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
    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "initializing %s", m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() > 1, 16);
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

    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "finalizing %s", m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() > 1, 16);

    m_finalized            = true;
    worker_is_finalizing() = true;
    if(m_is_master)
        master_is_finalizing() = true;
    manager::instance()->is_finalizing(true);

    using fini_t = operation::fini<Type>;
    using ValueT = typename trait::collects_data<Type>::type;
    auto upcast  = static_cast<tim::storage<Type, ValueT>*>(this);

    if(m_thread_init)
        fini_t(upcast, operation::mode_constant<operation::fini_mode::thread>{});
    if(m_is_master && m_global_init)
        fini_t(upcast, operation::mode_constant<operation::fini_mode::global>{});

    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "finalized %s", m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, true>::stack_clear()
{
    if(!m_stack.empty() && m_settings->get_stack_clearing())
    {
        std::unordered_set<Type*> _stack = m_stack;
        for(auto& itr : _stack)
        {
            operation::generic_operator<Type, operation::start<Type>, TIMEMORY_API>{
                *itr
            };
            operation::generic_operator<Type, operation::pop_node<Type>, TIMEMORY_API>{
                *itr
            };
        }
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
    {
        return [&]() {
            m_global_init = true;
            if(!m_is_master && master_instance())
                master_instance()->global_init();
            if(m_is_master)
            {
                using init_t = operation::init<Type>;
                using ValueT = typename trait::collects_data<Type>::type;
                auto upcast  = static_cast<tim::storage<Type, ValueT>*>(this);
                init_t(upcast, operation::mode_constant<operation::init_mode::global>{});
            }
            return m_global_init;
        }();
    }
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
    {
        return [&]() {
            m_thread_init = true;
            if(!m_is_master && master_instance())
                master_instance()->thread_init();
            bool _global_init = global_init();
            consume_parameters(_global_init);
            using init_t = operation::init<Type>;
            using ValueT = typename trait::collects_data<Type>::type;
            auto upcast  = static_cast<tim::storage<Type, ValueT>*>(this);
            init_t(upcast, operation::mode_constant<operation::init_mode::thread>{});
            return m_thread_init;
        }();
    }
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
    {
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
    }
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
        using type_t                   = decay_t<remove_pointer_t<decltype(this)>>;
        static thread_local auto _init = const_cast<type_t*>(this)->data_init();
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
        using type_t                   = decay_t<remove_pointer_t<decltype(this)>>;
        static thread_local auto _init = const_cast<type_t*>(this)->data_init();
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
        using type_t                   = decay_t<remove_pointer_t<decltype(this)>>;
        static thread_local auto _init = const_cast<type_t*>(this)->data_init();
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
    return _data().pop_graph();
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
    if(m_node_ids.empty() || m_graph_data_instance == nullptr)
        initialize();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::string
storage<Type, true>::get_prefix(const graph_node& node)
{
    auto _ret = operation::decode<TIMEMORY_API>{}(m_hash_ids, m_hash_aliases, node.id());
    if(_ret.find("unknown-hash=") == 0)
    {
        if(!m_is_master && singleton_t::master_instance())
        {
            auto _master = singleton_t::master_instance();
            return _master->get_prefix(node);
        }

        return operation::decode<TIMEMORY_API>{}(node.id());
    }

#if defined(TIMEMORY_TESTING) || defined(TIMEMORY_INTERNAL_TESTING)
    if(_ret.empty() || _ret.find("unknown-hash=") == 0)
    {
        TIMEMORY_EXCEPTION("Hash-lookup error!")
    }
#endif

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
        if(!m_is_master && singleton_t::master_instance())
        {
            auto _master = singleton_t::master_instance();
            return _master->get_prefix(id);
        }

        return get_hash_identifier(id);
    }

#if defined(TIMEMORY_TESTING) || defined(TIMEMORY_INTERNAL_TESTING)
    if(_ret.empty() || _ret.find("unknown-hash=") == 0)
    {
        TIMEMORY_EXCEPTION("Hash-lookup error!")
    }
#endif

    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::graph_data_t&
storage<Type, true>::_data()
{
    if(m_graph_data_instance == nullptr)
    {
        auto_lock_t lk(singleton_t::get_mutex(), std::defer_lock);

        if(!m_is_master && master_instance())
        {
            static thread_local bool _data_init = master_instance()->data_init();
            auto&                    m          = master_instance()->data();
            consume_parameters(_data_init);

            if(!lk.owns_lock())
                lk.lock();

            CONDITIONAL_PRINT_HERE(
                m_settings->get_debug(), "[%s]> Worker: %i, master ptr: %p",
                demangle<Type>().c_str(), (int) m_thread_idx, (void*) &m);
            TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
                m_settings->get_debug() && m_settings->get_verbose() > 1, 16);
            if(m.current())
            {
                auto         _current = m.current();
                auto         _id      = _current->id();
                auto         _depth   = _current->depth();
                graph_node_t node(_id, operation::dummy<Type>{}(), _depth, m_thread_idx);
                if(!m_graph_data_instance)
                    m_graph_data_instance = new graph_data_t(node, _depth, &m);
                m_graph_data_instance->depth()     = _depth;
                m_graph_data_instance->sea_level() = _depth;
            }
            else
            {
                if(!m_graph_data_instance)
                {
                    graph_node_t node(0, operation::dummy<Type>{}(), 1, m_thread_idx);
                    m_graph_data_instance = new graph_data_t(node, 1, &m);
                }
                m_graph_data_instance->depth()     = 1;
                m_graph_data_instance->sea_level() = 1;
            }
            m_graph_data_instance->set_master(&m);
        }
        else
        {
            if(!lk.owns_lock())
                lk.lock();

            std::string _prefix = "> [tot] total";
            add_hash_id(_prefix);
            graph_node_t node(0, operation::dummy<Type>{}(), 0, m_thread_idx);
            if(!m_graph_data_instance)
                m_graph_data_instance = new graph_data_t(node, 0, nullptr);
            m_graph_data_instance->depth()     = 0;
            m_graph_data_instance->sea_level() = 0;
            CONDITIONAL_PRINT_HERE(m_settings->get_debug(),
                                   "[%s]> Master: %i, master ptr: %p",
                                   demangle<Type>().c_str(), (int) m_thread_idx,
                                   (void*) m_graph_data_instance);
            TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
                m_settings->get_debug() && m_settings->get_verbose() > 1, 16);
        }

        if(m_node_ids.empty())
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
    if(m_children.empty())
        return;

    for(auto& itr : m_children)
        merge(itr);

    // create lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
    if(!l.owns_lock())
        l.lock();

    for(auto& itr : m_children)
    {
        if(itr != this)
            itr->data().clear();
    }

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
typename storage<Type, true>::result_array_t
storage<Type, true>::get()
{
    result_array_t _ret;
    operation::finalize::get<Type, true>{ *this }(_ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
storage<Type, true>::get(Tp& _ret)
{
    return operation::finalize::get<Type, true>{ *this }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::mpi_get()
{
    dmp_result_t _ret;
    operation::finalize::mpi_get<Type, true>{ *this }(_ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
storage<Type, true>::mpi_get(Tp& _ret)
{
    return operation::finalize::mpi_get<Type, true>{ *this }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::upc_get()
{
    dmp_result_t _ret;
    operation::finalize::upc_get<Type, true>{ *this }(_ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
storage<Type, true>::upc_get(Tp& _ret)
{
    return operation::finalize::upc_get<Type, true>{ *this }(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename storage<Type, true>::dmp_result_t
storage<Type, true>::dmp_get()
{
    dmp_result_t _ret;
    operation::finalize::dmp_get<Type, true>{ *this }(_ret);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
Tp&
storage<Type, true>::dmp_get(Tp& _ret)
{
    return operation::finalize::dmp_get<Type, true>{ *this }(_ret);
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

    auto _is_primary       = singleton_t::is_master(this);
    auto _primary_instance = singleton_t::master_instance();

    if(!_is_primary && !_primary_instance && common_singleton::is_main_thread())
    {
        PRINT_HERE("[%s]> storage instance (%p) on main thread is not designated as the "
                   "primary but there is a nullptr to primary. Designating as primary",
                   m_label.c_str(), (void*) this);
        _is_primary = true;
    }

    if(!_is_primary)
    {
        if(_primary_instance)
            _primary_instance->merge(this);
        finalize();
    }
    else
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

        // generate output
        if(m_settings->get_auto_output())
        {
            m_printer.reset(new printer_t(Type::get_label(), this, m_settings));

            if(m_manager)
                m_manager->add_entries(this->size());

            m_printer->execute();
        }

        instance_count().store(0);
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
        if(!m_manager)
            return;

        auto   _label = Type::label();
        size_t pos    = std::string::npos;
        // remove the namespaces
        for(const auto* itr : { "tim::component::", "tim::project::", "tim::tpls::",
                                "tim::api::", "tim::" })
        {
            while((pos = _label.find(itr)) != std::string::npos)
                _label = _label.replace(pos, std::string(itr).length(), "");
        }
        // replace spaces with underscores
        while((pos = _label.find(' ')) != std::string::npos)
            _label = _label.replace(pos, 1, "_");
        // convert to upper-case
        for(auto& itr : _label)
            itr = toupper(itr);
        for(auto itr : { ':', '<', '>' })
        {
            auto _pos = _label.find(itr);
            while(_pos != std::string::npos)
                _pos = _label.erase(_pos, 1).find(itr);
        }
        std::stringstream env_var;
        env_var << "TIMEMORY_" << _label << "_ENABLED";
        auto _enabled = tim::get_env<bool>(env_var.str(), true);
        trait::runtime_enabled<Type>::set(_enabled);

        auto _instance_id = m_instance_id;
        bool _is_master   = m_is_master;
        auto _sync        = [&]() {
            if(m_graph_data_instance)
                this->data().sync_sea_level();
        };
        auto _cleanup = [_is_master, _instance_id]() {
            if(_is_master)
                return;
            if(manager::master_instance())
                manager::master_instance()->remove_synchronization(demangle<Type>(),
                                                                   _instance_id);
            if(manager::instance())
                manager::instance()->remove_synchronization(demangle<Type>(),
                                                            _instance_id);
        };
        func_t _finalize = [&]() {
            auto _instance = this_type::get_singleton();
            if(_instance)
            {
                auto _debug_v = m_settings->get_debug();
                auto _verb_v  = m_settings->get_verbose();
                if(_debug_v || _verb_v > 1)
                {
                    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                               "calling singleton::reset(this)");
                }
                _instance->reset(this);
                if((m_is_master || common_singleton::is_main_thread()) && _instance)
                {
                    if(_debug_v || _verb_v > 1)
                    {
                        PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                                   "calling singleton::reset()");
                    }
                    _instance->reset();
                }
            }
            else
            {
                DEBUG_PRINT_HERE("[%s]> %p", demangle<Type>().c_str(), (void*) _instance);
            }
            if(m_is_master)
                trait::runtime_enabled<Type>::set(false);
        };

        if(!m_is_master)
        {
            manager::master_instance()->add_synchronization(
                demangle<Type>(), m_instance_id, std::move(_sync));
            m_manager->add_synchronization(demangle<Type>(), m_instance_id,
                                           std::move(_sync));
        }

        m_manager->add_finalizer(demangle<Type>(), std::move(_cleanup),
                                 std::move(_finalize), m_is_master,
                                 trait::fini_priority<Type>::value);
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
: base_type(singleton_t::is_master_thread(), instance_count()++, demangle<Type>())
{
    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "constructing %s", m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() > 1, 16);
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
    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "destroying %s", m_label.c_str());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
storage<Type, false>::stack_clear()
{
    if(!m_stack.empty() && m_settings->get_stack_clearing())
    {
        std::unordered_set<Type*> _stack = m_stack;
        for(auto& itr : _stack)
            operation::stop<Type>{ *itr };
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

    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "initializing %s", m_label.c_str());
    TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(
        m_settings->get_debug() && m_settings->get_verbose() > 1, 16);

    m_initialized = true;

    using init_t = operation::init<Type>;
    using ValueT = typename trait::collects_data<Type>::type;
    auto upcast  = static_cast<tim::storage<Type, ValueT>*>(this);

    if(!m_is_master)
    {
        init_t(upcast, operation::mode_constant<operation::init_mode::thread>{});
    }
    else
    {
        init_t(upcast, operation::mode_constant<operation::init_mode::global>{});
        init_t(upcast, operation::mode_constant<operation::init_mode::thread>{});
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

    CONDITIONAL_PRINT_HERE(m_settings->get_debug(), "finalizing %s", m_label.c_str());

    using fini_t = operation::fini<Type>;
    using ValueT = typename trait::collects_data<Type>::type;
    auto upcast  = static_cast<tim::storage<Type, ValueT>*>(this);

    m_finalized = true;
    manager::instance()->is_finalizing(true);
    if(!m_is_master)
    {
        worker_is_finalizing() = true;
        fini_t(upcast, operation::mode_constant<operation::fini_mode::thread>{});
    }
    else
    {
        master_is_finalizing() = true;
        worker_is_finalizing() = true;
        fini_t(upcast, operation::mode_constant<operation::fini_mode::thread>{});
        fini_t(upcast, operation::mode_constant<operation::fini_mode::global>{});
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

    if(m_settings->get_stack_clearing())
    {
        for(auto& itr : m_children)
            merge(itr);
    }

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
        if(!m_manager)
            return;
        if(m_manager->is_finalizing())
            return;

        auto _label = Type::label();
        for(auto& itr : _label)
            itr = toupper(itr);
        std::stringstream env_var;
        env_var << "TIMEMORY_" << _label << "_ENABLED";
        auto _enabled = tim::get_env<bool>(env_var.str(), true);
        trait::runtime_enabled<Type>::set(_enabled);

        auto   _cleanup  = [&]() {};
        func_t _finalize = [&]() {
            auto _instance = this_type::get_singleton();
            if(_instance)
            {
                auto _debug_v = m_settings->get_debug();
                auto _verb_v  = m_settings->get_verbose();
                if(_debug_v || _verb_v > 1)
                {
                    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                               "calling singleton::reset(this)");
                }
                _instance->reset(this);
                if((m_is_master || common_singleton::is_main_thread()) && _instance)
                {
                    if(_debug_v || _verb_v > 1)
                    {
                        PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                                   "calling singleton::reset()");
                    }
                    _instance->reset();
                }
            }
            else
            {
                DEBUG_PRINT_HERE("[%s]> %p", demangle<Type>().c_str(), (void*) _instance);
            }
            if(m_is_master)
                trait::runtime_enabled<Type>::set(false);
        };

        m_manager->add_finalizer(demangle<Type>(), std::move(_cleanup),
                                 std::move(_finalize), m_is_master,
                                 trait::fini_priority<Type>::value);
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
