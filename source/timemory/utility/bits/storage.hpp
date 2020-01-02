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
#include "timemory/manager.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/types.hpp"

#include <string>

//======================================================================================//

template <typename _Tp>
void
tim::generic_serialization(const std::string& fname, const _Tp& obj)
{
    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
    std::ofstream         ofs(fname.c_str());
    if(ofs)
    {
        // ensure json write final block during destruction before the file is closed
        //                                  args: precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 2);
        cereal::JSONOutputArchive          oa(ofs, opts);
        oa.setNextName("timemory");
        oa.startNode();
        oa(cereal::make_nvp("data", obj));
        oa.finishNode();
    }
    if(ofs)
        ofs << std::endl;
    ofs.close();
}

//--------------------------------------------------------------------------------------//

inline void
tim::base::storage::add_text_output(const string_t& _label, const string_t& _file)
{
    m_manager = ::tim::manager::instance();
    if(m_manager)
        m_manager->add_text_output(_label, _file);
}

//--------------------------------------------------------------------------------------//

inline void
tim::base::storage::add_json_output(const string_t& _label, const string_t& _file)
{
    m_manager = ::tim::manager::instance();
    if(m_manager)
        m_manager->add_json_output(_label, _file);
}

//--------------------------------------------------------------------------------------//

template <typename Type>
void
tim::impl::storage<Type, true>::get_shared_manager()
{
    // only perform this operation when not finalizing
    if(!this_type::is_finalizing())
    {
        if(settings::debug())
            PRINT_HERE("%s", "getting shared manager");
        m_manager         = ::tim::manager::instance();
        using func_t      = ::tim::manager::finalizer_func_t;
        bool   _is_master = singleton_t::is_master(this);
        func_t _finalize  = [&]() {
            auto _instance = this_type::get_singleton();
            if(_instance)
            {
                this->stack_clear();
                _instance->reset(this);
            }
        };
        m_manager->add_finalizer(Type::label(), std::move(_finalize), _is_master);
    }
}

//--------------------------------------------------------------------------------------//

template <typename Type>
void
tim::impl::storage<Type, false>::get_shared_manager()
{
    // only perform this operation when not finalizing
    if(!this_type::is_finalizing())
    {
        if(settings::debug())
            PRINT_HERE("%s", "getting shared manager");
        m_manager         = ::tim::manager::instance();
        using func_t      = ::tim::manager::finalizer_func_t;
        bool   _is_master = singleton_t::is_master(this);
        func_t _finalize  = [&]() {
            auto _instance = this_type::get_singleton();
            if(_instance)
            {
                this->stack_clear();
                _instance->reset(this);
            }
        };
        m_manager->add_finalizer(Type::label(), std::move(_finalize), _is_master);
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
tim::impl::storage_singleton_t<_Tp>*
tim::get_storage_singleton()
{
    using singleton_type  = ::tim::impl::storage_singleton_t<_Tp>;
    static auto _instance = std::unique_ptr<singleton_type>(new singleton_type());
    return _instance.get();
}

//======================================================================================//
