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
 * \file timemory/operations/types/finalize_get.hpp
 * \brief Definition for various functions for finalize_get in operations
 */

#pragma once

//======================================================================================//
//
#include "timemory/operations/macros.hpp"
//
#include "timemory/operations/types.hpp"
//
#include "timemory/operations/declaration.hpp"
//
//======================================================================================//

namespace tim
{
namespace operation
{
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct get<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using result_type              = typename storage_type::result_array_t;
    using distrib_type             = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;

    get(storage_type&, result_type&);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct get<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    get(storage_type&) {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
get<Type, true>::get(storage_type& data, result_type& ret)
{
    bool _thread_scope_only = trait::thread_scope_only<Type>::value;
    bool _use_tid_prefix    = (!settings::collapse_threads() || _thread_scope_only);
    auto _num_thr_count     = manager::get_thread_count();

    data.m_node_init = dmp::is_initialized();
    data.m_node_rank = dmp::rank();
    data.m_node_size = dmp::size();

    //------------------------------------------------------------------------------//
    //
    //  Compute the thread prefix
    //
    //------------------------------------------------------------------------------//
    auto _get_thread_prefix = [&](const graph_node& itr) {
        if(!_use_tid_prefix || itr.tid() == std::numeric_limits<uint16_t>::max())
            return std::string(">>> ");

        // prefix spacing
        static uint16_t width = 1;
        if(_num_thr_count > 9)
            width = std::max(width, (uint16_t)(log10(_num_thr_count) + 1));
        std::stringstream ss;
        ss.fill('0');
        ss << "|" << std::setw(width) << itr.tid() << ">>> ";
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //
    //  Compute the node prefix
    //
    //------------------------------------------------------------------------------//
    auto _get_node_prefix = [&](const graph_node& itr) {
        if(!data.m_node_init)
            return _get_thread_prefix(itr);

        // prefix spacing
        static uint16_t width = 1;
        if(data.m_node_size > 9)
            width = std::max(width, (uint16_t)(log10(data.m_node_size) + 1));
        std::stringstream ss;
        ss.fill('0');
        ss << "|" << std::setw(width) << data.m_node_rank << _get_thread_prefix(itr);
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //
    //  Compute the indentation
    //
    //------------------------------------------------------------------------------//
    // fix up the prefix based on the actual depth
    auto _compute_modified_prefix = [&](const graph_node& itr) {
        std::string _prefix      = data.get_prefix(itr);
        std::string _indent      = "";
        std::string _node_prefix = _get_node_prefix(itr);

        int64_t _depth = itr.depth() - 1;
        if(_depth > 0)
        {
            for(int64_t ii = 0; ii < _depth - 1; ++ii)
                _indent += "  ";
            _indent += "|_";
        }

        return _node_prefix + _indent + _prefix;
    };

    // convert graph to a vector
    auto convert_graph = [&]() {
        result_type _list;
        {
            // the head node should always be ignored
            int64_t _min = std::numeric_limits<int64_t>::max();
            for(const auto& itr : data.graph())
                _min = std::min<int64_t>(_min, itr.depth());

            for(auto itr = data.graph().begin(); itr != data.graph().end(); ++itr)
            {
                if(itr->depth() > _min)
                {
                    auto _depth     = itr->depth() - (_min + 1);
                    auto _prefix    = _compute_modified_prefix(*itr);
                    auto _rolling   = itr->id();
                    auto _stats     = itr->stats();
                    auto _parent    = graph_type::parent(itr);
                    auto _hierarchy = hierarchy_type{};
                    auto _tid       = itr->tid();
                    auto _pid       = itr->pid();
                    if(_parent && _parent->depth() > _min)
                    {
                        while(_parent)
                        {
                            _hierarchy.push_back(_parent->id());
                            _rolling += _parent->id();
                            _parent = graph_type::parent(_parent);
                            if(!_parent || !(_parent->depth() > _min))
                                break;
                        }
                    }
                    if(_hierarchy.size() > 1)
                        std::reverse(_hierarchy.begin(), _hierarchy.end());
                    _hierarchy.push_back(itr->id());
                    auto&& _entry = result_node(itr->id(), itr->obj(), _prefix, _depth,
                                                _rolling, _hierarchy, _stats, _tid, _pid);
                    _list.push_back(_entry);
                }
            }
        }

        if(!settings::collapse_threads() || _thread_scope_only)
            return _list;

        result_type _combined;

        //--------------------------------------------------------------------------//
        //
        auto _equiv = [&](const result_node& _lhs, const result_node& _rhs) {
            return (_lhs.hash() == _rhs.hash() && _lhs.prefix() == _rhs.prefix() &&
                    _lhs.depth() == _rhs.depth() &&
                    _lhs.rolling_hash() == _rhs.rolling_hash());
        };

        //--------------------------------------------------------------------------//
        //
        auto _exists = [&](const result_node& _lhs) {
            for(auto itr = _combined.begin(); itr != _combined.end(); ++itr)
            {
                if(_equiv(_lhs, *itr))
                    return itr;
            }
            return _combined.end();
        };

        //--------------------------------------------------------------------------//
        //  collapse duplicates
        //
        for(auto& itr : _list)
        {
            auto citr = _exists(itr);
            if(citr == _combined.end())
            {
                itr.tid() = std::numeric_limits<uint16_t>::max();
                _combined.emplace_back(itr);
            }
            else
            {
                citr->data() += itr.data();
                citr->data().plus(itr.data());
                citr->stats() += itr.stats();
            }
        }
        return _combined;
    };

    ret = convert_graph();
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
