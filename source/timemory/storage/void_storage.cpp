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

#ifndef TIMEMORY_STORAGE_VOID_STORAGE_CPP_
#define TIMEMORY_STORAGE_VOID_STORAGE_CPP_ 1

#include "timemory/storage/void_storage.hpp"
#include "timemory/manager.hpp"
#include "timemory/operations/types/fini.hpp"
#include "timemory/operations/types/init.hpp"
#include "timemory/operations/types/node.hpp"
#include "timemory/operations/types/start.hpp"
#include "timemory/operations/types/stop.hpp"

namespace tim
{
namespace impl
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
std::atomic<int64_t>&
void_storage<Type>::instance_count()
{
    static std::atomic<int64_t> _counter{ 0 };
    return _counter;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
typename void_storage<Type>::parent_type&
void_storage<Type>::get_upcast()
{
    return static_cast<parent_type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
const typename void_storage<Type>::parent_type&
void_storage<Type>::get_upcast() const
{
    return static_cast<const parent_type&>(*this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void_storage<Type>::void_storage()
: base_type(parent_type::singleton_type::is_master_thread(), instance_count()++,
            demangle<Type>())
{
    if(m_settings->get_debug())
        printf("[%s]> constructing @ %i...\n", m_label.c_str(), __LINE__);
    get_shared_manager();
    component::state<Type>::has_storage() = true;
    // m_printer = std::make_shared<printer_t>(Type::get_label(), this);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void_storage<Type>::~void_storage()
{
    component::state<Type>::has_storage() = false;
    if(m_settings->get_debug())
        printf("[%s]> destructing @ %i...\n", m_label.c_str(), __LINE__);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
void_storage<Type>::stack_clear()
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
void_storage<Type>::initialize()
{
    if(m_initialized)
        return;

    if(m_settings->get_debug())
        printf("[%s]> initializing...\n", m_label.c_str());

    m_initialized = true;

    using init_t = operation::init<Type>;
    auto upcast  = &get_upcast();

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
void_storage<Type>::finalize()
{
    if(m_finalized)
        return;

    if(!m_initialized)
        return;

    if(m_settings->get_debug())
        printf("[%s]> finalizing...\n", m_label.c_str());

    using fini_t = operation::fini<Type>;
    auto upcast  = &get_upcast();

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
void_storage<Type>::stack_pop(Type* obj)
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
void_storage<Type>::merge()
{
    auto m_children = parent_type::singleton_type::children();
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
void_storage<Type>::merge(this_type* itr)
{
    if(itr)
        operation::finalize::merge<Type, false>{ get_upcast(),
                                                 static_cast<parent_type&>(*itr) };
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
void
void_storage<Type>::get_shared_manager()
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

        bool   _is_master = parent_type::singleton_type::is_master(&get_upcast());
        auto   _cleanup   = [&]() {};
        func_t _finalize  = [&]() {
            auto _instance = parent_type::get_singleton();
            if(_instance)
            {
                auto _debug_v = m_settings->get_debug();
                auto _verb_v  = m_settings->get_verbose();
                if(_debug_v || _verb_v > 1)
                {
                    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                               "calling _instance->reset(this)");
                }
                _instance->reset(&get_upcast());
                // if(_debug_v || _verb_v > 1)
                //    PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                //               "calling _instance->smart_instance().reset()");
                // _instance->smart_instance().reset();
                if(_is_master && _instance)
                {
                    if(_debug_v || _verb_v > 1)
                    {
                        PRINT_HERE("[%s] %s", demangle<Type>().c_str(),
                                   "calling _instance->reset()");
                    }
                    _instance->reset();
                    // _instance->smart_master_instance().reset();
                }
            }
            else
            {
                DEBUG_PRINT_HERE("[%s]> %p", demangle<Type>().c_str(), (void*) _instance);
            }
            if(_is_master)
                trait::runtime_enabled<Type>::set(false);
        };

        m_manager->add_finalizer(demangle<Type>(), std::move(_cleanup),
                                 std::move(_finalize), _is_master);
    }
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace impl
}  // namespace tim

#endif
