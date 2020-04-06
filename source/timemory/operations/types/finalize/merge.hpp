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
struct merge<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using singleton_t              = typename storage_type::singleton_t;
    using graph_t                  = typename storage_type::graph_t;

    merge(storage_type& lhs, storage_type& rhs)
    {
        using pre_order_iterator = typename graph_t::pre_order_iterator;
        using sibling_iterator   = typename graph_t::sibling_iterator;

        // don't merge self
        if(&lhs == &rhs)
            return;

        // if merge was not initialized return
        if(!rhs.is_initialized())
            return;

        rhs.stack_clear();

        // create lock
        auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
        if(!l.owns_lock())
            l.lock();

        auto _copy_hash_ids = [&]() {
            for(const auto& itr : (*rhs.get_hash_ids()))
                if(lhs.m_hash_ids->find(itr.first) == lhs.m_hash_ids->end())
                    (*lhs.m_hash_ids)[itr.first] = itr.second;
            for(const auto& itr : (*rhs.get_hash_aliases()))
                if(lhs.m_hash_aliases->find(itr.first) == lhs.m_hash_aliases->end())
                    (*lhs.m_hash_aliases)[itr.first] = itr.second;
        };

        // if self is not initialized but itr is, copy data
        if(rhs.is_initialized() && !lhs.is_initialized())
        {
            PRINT_HERE(
                "[%s]> Warning! master is not initialized! Segmentation fault likely",
                Type::get_label().c_str());
            lhs.graph().insert_subgraph_after(lhs._data().head(), rhs.data().head());
            lhs.m_initialized = rhs.m_initialized;
            lhs.m_finalized   = rhs.m_finalized;
            _copy_hash_ids();
            return;
        }
        else
        {
            _copy_hash_ids();
        }

        if(rhs.size() == 0 || !rhs.data().has_head())
            return;

        int64_t num_merged     = 0;
        auto    inverse_insert = rhs.data().get_inverse_insert();

        for(auto entry : inverse_insert)
        {
            auto master_entry = lhs.data().find(entry.second);
            if(master_entry != lhs.data().end())
            {
                pre_order_iterator pitr(entry.second);

                if(rhs.graph().is_valid(pitr) && pitr)
                {
                    if(settings::debug() || settings::verbose() > 2)
                        PRINT_HERE("[%s]> worker is merging %i records into %i records",
                                   Type::get_label().c_str(), (int) rhs.size(),
                                   (int) lhs.size());

                    pre_order_iterator pos = master_entry;

                    if(*pos == *pitr)
                    {
                        ++num_merged;
                        sibling_iterator other = pitr;
                        for(auto sitr = other.begin(); sitr != other.end(); ++sitr)
                        {
                            pre_order_iterator pchild = sitr;
                            if(pchild->obj().get_laps() == 0)
                                continue;
                            lhs.graph().append_child(pos, pchild);
                        }
                    }

                    if(settings::debug() || settings::verbose() > 2)
                        PRINT_HERE("[%s]> master has %i records",
                                   Type::get_label().c_str(), (int) lhs.size());

                    // remove the entry from this graph since it has been added
                    rhs.graph().erase_children(entry.second);
                    rhs.graph().erase(entry.second);
                }
            }
        }

        int64_t merge_size = static_cast<int64_t>(inverse_insert.size());
        if(num_merged != merge_size)
        {
            int64_t           diff = merge_size - num_merged;
            std::stringstream ss;
            ss << "Testing error! Missing " << diff << " merge points. The worker thread "
               << "contained " << merge_size << " bookmarks but only merged "
               << num_merged << " nodes!";

            PRINT_HERE("%s", ss.str().c_str());

#if defined(TIMEMORY_TESTING)
            throw std::runtime_error(ss.str());
#endif
        }

        if(num_merged == 0)
        {
            if(settings::debug() || settings::verbose() > 2)
                PRINT_HERE("[%s]> worker is not merged!", Type::get_label().c_str());
            pre_order_iterator _nitr(rhs.data().head());
            ++_nitr;
            if(!lhs.graph().is_valid(_nitr))
                _nitr = pre_order_iterator(rhs.data().head());
            lhs.graph().append_child(lhs._data().head(), _nitr);
        }

        rhs.data().clear();
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct merge<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    using singleton_t              = typename storage_type::singleton_t;

    merge(storage_type& lhs, storage_type& rhs)
    {
        rhs.stack_clear();

        // create lock
        auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
        if(!l.owns_lock())
            l.lock();

        for(const auto& itr : *rhs.get_hash_ids())
            if(lhs.m_hash_ids->find(itr.first) == lhs.m_hash_ids->end())
                (*lhs.m_hash_ids)[itr.first] = itr.second;
        for(const auto& itr : (*rhs.get_hash_aliases()))
            if(lhs.m_hash_aliases->find(itr.first) == lhs.m_hash_aliases->end())
                (*lhs.m_hash_aliases)[itr.first] = itr.second;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
