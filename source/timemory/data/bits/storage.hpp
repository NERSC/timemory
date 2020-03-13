//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file timemory/utility/bits/storage.hpp
 * \headerfile timemory/utility/bits/storage.hpp "timemory/utility/bits/storage.hpp"
 * Provides implementation of some storage related functions
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/data/base_storage.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/types.hpp"

#include <string>

//======================================================================================//
/*
template <typename Type>
void
tim::impl::storage<Type, true>::get_shared_manager()
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

//--------------------------------------------------------------------------------------//

template <typename Type>
void
tim::impl::storage<Type, false>::get_shared_manager()
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

//--------------------------------------------------------------------------------------//

template <typename Tp>
tim::storage_singleton<Tp>*
tim::get_storage_singleton()
{
    using singleton_type  = tim::storage_singleton<Tp>;
    using component_type  = typename Tp::component_type;
    static auto _instance = std::unique_ptr<singleton_type>(
        (trait::runtime_enabled<component_type>::get()) ? new singleton_type() : nullptr);
    return _instance.get();
}
*/
//======================================================================================//
