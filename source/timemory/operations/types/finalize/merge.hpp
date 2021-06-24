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

#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/basic_tree.hpp"
#include "timemory/storage/graph.hpp"

#include <unordered_map>

namespace tim
{
namespace operation
{
namespace finalize
{
//
template <typename KeyT, typename MappedT>
using uomap_t = std::unordered_map<KeyT, MappedT>;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
merge<Type, true>::merge(storage_type& lhs, storage_type& rhs)
{
    using pre_order_iterator = typename graph_t::pre_order_iterator;
    using sibling_iterator   = typename graph_t::sibling_iterator;

    // don't merge self
    if(&lhs == &rhs)
        return;

    // create lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
    if(!l.owns_lock())
        l.lock();

    // if merge was not initialized return
    if(!rhs.is_initialized())
        return;

    auto* _settings = tim::settings::instance();
    if(!_settings)
        PRINT_HERE("[%s]> nullptr to settings!", Type::get_label().c_str());
    auto _debug =
        _settings != nullptr && (_settings->get_debug() || _settings->get_verbose() > 2);

    rhs.stack_clear();

    auto _copy_hash_ids = [&lhs, &rhs, _debug]() {
        // copy over mapping of hashes to strings
        if(rhs.get_hash_ids() && lhs.get_hash_ids())
        {
            CONDITIONAL_PRINT_HERE(
                _debug, "[%s]> merging %lu hash-ids into existing set of %lu hash-ids!",
                Type::get_label().c_str(), (unsigned long) rhs.get_hash_ids()->size(),
                (unsigned long) lhs.get_hash_ids()->size());

            auto _hash_ids = *rhs.get_hash_ids();
            for(const auto& itr : _hash_ids)
            {
                if(lhs.m_hash_ids->find(itr.first) == lhs.m_hash_ids->end())
                    lhs.m_hash_ids->emplace(itr.first, itr.second);
            }
        }
        // copy over aliases
        if(rhs.get_hash_aliases() && lhs.get_hash_aliases())
        {
            CONDITIONAL_PRINT_HERE(
                _debug,
                "[%s]> merging %lu hash-aliases into existing set of %lu hash-aliases!",
                Type::get_label().c_str(), (unsigned long) rhs.get_hash_aliases()->size(),
                (unsigned long) lhs.get_hash_aliases()->size());

            auto _hash_aliases = *rhs.get_hash_aliases();
            for(const auto& itr : _hash_aliases)
            {
                if(lhs.m_hash_aliases->find(itr.first) == lhs.m_hash_aliases->end())
                    lhs.m_hash_aliases->emplace(itr.first, itr.second);
            }
        }
    };

    // if self is not initialized but itr is, copy data
    if(rhs.is_initialized() && !lhs.is_initialized())
    {
        PRINT_HERE("[%s]> Warning! master is not initialized! Segmentation fault likely",
                   Type::get_label().c_str());
        lhs.graph().insert_subgraph_after(lhs._data().head(), rhs.data().head());
        lhs.m_initialized = rhs.m_initialized;
        lhs.m_finalized   = rhs.m_finalized;
        _copy_hash_ids();
        return;
    }

    _copy_hash_ids();

    if(rhs.empty() || !rhs.data().has_head())
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
                CONDITIONAL_PRINT_HERE(
                    _debug, "[%s]> worker is merging %i records into %i records",
                    Type::get_label().c_str(), (int) rhs.size(), (int) lhs.size());

                pre_order_iterator pos = master_entry;

                if(*pos == *pitr)
                {
                    ++num_merged;
                    sibling_iterator other = pitr;
                    for(auto sitr = other.begin(); sitr != other.end(); ++sitr)
                    {
                        pre_order_iterator pchild = sitr;
                        if(!pchild || pchild->obj().get_laps() == 0)
                            continue;
                        lhs.graph().append_child(pos, pchild);
                    }
                }

                CONDITIONAL_PRINT_HERE(_debug, "[%s]> master has %i records",
                                       Type::get_label().c_str(), (int) lhs.size());

                // remove the entry from this graph since it has been added
                // rhs.graph().erase(pitr);
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

        CONDITIONAL_PRINT_HERE(_debug, "%s", ss.str().c_str());

#if defined(TIMEMORY_TESTING) || defined(TIMEMORY_INTERNAL_TESTING)
        TIMEMORY_EXCEPTION(ss.str());
#endif
    }

    if(num_merged == 0)
    {
        CONDITIONAL_PRINT_HERE(_debug, "[%s]> worker is not merged!",
                               Type::get_label().c_str());
        pre_order_iterator _nitr(rhs.data().head());
        ++_nitr;
        if(!lhs.graph().is_valid(_nitr))
            _nitr = pre_order_iterator(rhs.data().head());
        lhs.graph().append_child(lhs._data().head(), _nitr);
    }

    CONDITIONAL_PRINT_HERE(_debug, "[%s]> clearing merged storage!",
                           Type::get_label().c_str());

    rhs.data().clear();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
merge<Type, true>::merge(result_type& dst, result_type& src)
{
    using result_node = typename result_type::value_type;

    try
    {
        dst.reserve(dst.size() + src.size());
    } catch(std::bad_alloc& e)
    {
        std::cerr << "Warning! Running out of memory is probable! " << e.what()
                  << std::endl;
    }

    //----------------------------------------------------------------------------------//
    //
    using merge_hash_map_t =
        uomap_t<int64_t,
                uomap_t<uint64_t, uomap_t<uint64_t, uomap_t<std::string, int64_t>>>>;

    // this is a look-up table for the index in dst of existing records with same depth +
    // hash + rolling-hash + prefix
    merge_hash_map_t _hash_table{};

    //----------------------------------------------------------------------------------//
    // add hash-table entries for a new record
    //
    auto _add_entry = [&_hash_table](const result_node& _obj, int64_t _idx) {
        _hash_table[_obj.depth()][_obj.hash()][_obj.rolling_hash()][_obj.prefix()] = _idx;
    };

    //----------------------------------------------------------------------------------//
    // get the index in dst of an existing match if it exists, otherwise return -1
    //
    auto _get_index = [&](const result_node& _lhs) -> int64_t {
        auto& _tbl = _hash_table[_lhs.depth()][_lhs.hash()][_lhs.rolling_hash()];
        auto  itr  = _tbl.find(_lhs.prefix());
        if(itr != _tbl.end())
            return itr->second;
        return -1;
    };

    //----------------------------------------------------------------------------------//
    // add hash-table entries for all the existing records
    //
    for(size_t i = 0; i < dst.size(); ++i)
        _add_entry(dst.at(i), static_cast<int64_t>(i));

    //----------------------------------------------------------------------------------//
    //  collapse duplicates
    //
    size_t cnt       = 0;
    size_t ndup      = 0;
    auto*  _settings = tim::settings::instance();
    auto   _debug    = (_settings) ? _settings->get_debug() : true;
    auto   _verbose  = (_settings) ? _settings->get_verbose() : 1;

    if(_debug || _verbose > 0)
    {
        fprintf(stderr, "Merging %lu new records into %lu existing records\n",
                (unsigned long) src.size(), (unsigned long) dst.size());
    }

    for(auto& itr : src)
    {
        if(_debug || _verbose > 3)
        {
            fprintf(stderr, "Checking %lu of %lu...", (unsigned long) cnt++,
                    (unsigned long) src.size());
        }

        auto idx = _get_index(itr);
        if(idx < 0)
        {
            if(_debug || _verbose > 3)
                fprintf(stderr, "new entry\n");
            _add_entry(itr, static_cast<int64_t>(dst.size()));
            dst.emplace_back(std::move(itr));
        }
        else
        {
            ++ndup;
            if(_debug || _verbose > 3)
                fprintf(stderr, "duplicate\n");
            auto& citr = dst.at(idx);
            citr.data() += itr.data();
            citr.data().plus(itr.data());
            citr.stats() += itr.stats();
        }
    }

    if(_debug || _verbose > 0)
    {
        fprintf(stderr, "Merged %lu duplicates into %lu records\n", (unsigned long) ndup,
                (unsigned long) dst.size());
    }

    // shrink the reserved memory
    dst.shrink_to_fit();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
basic_tree<Tp>
merge<Type, true>::operator()(const basic_tree<Tp>& _bt)
{
    using children_type = typename basic_tree<Tp>::children_type;

    // do nothing if no children
    if(_bt.get_children().empty())
        return _bt;

    // make copy
    auto _ret = _bt;
    // recursively apply
    for(auto& itr : _ret.get_children())
        *itr = (*this)(*itr);

    // aggregate children
    children_type _children{};
    for(auto& itr : _ret.get_children())
    {
        bool found = false;
        for(auto& citr : _children)
        {
            if(*citr == *itr)
            {
                found = true;
                *citr += *itr;
            }
        }
        if(!found)
            _children.emplace_back(itr);
        // _children.emplace_back(std::move(itr));
    }

    // update new children
    _ret.get_children() = _children;
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
basic_tree<Tp>
merge<Type, true>::operator()(const basic_tree<Tp>& _lhs, const basic_tree<Tp>& _rhs)
{
    return basic_tree<Tp>{ _lhs } += _rhs;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
std::vector<basic_tree<Tp>>
merge<Type, true>::operator()(const std::vector<basic_tree<Tp>>& _bt)
{
    auto _ret = _bt;
    for(auto& itr : _ret)
        itr = (*this)(itr);
    return _ret;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
std::vector<basic_tree<Tp>>
merge<Type, true>::operator()(const std::vector<basic_tree<Tp>>& _lhs,
                              const std::vector<basic_tree<Tp>>& _rhs)
{
    using basic_t      = basic_tree<Tp>;
    using basic_vec_t  = std::vector<basic_t>;
    using basic_map_t  = std::map<size_t, basic_t>;
    using basic_bool_t = std::vector<bool>;

    auto _p  = basic_map_t{};  // map of paired instances in lhs and rhs
    auto _ul = basic_map_t{};  // map of unpaired instances in lhs
    auto _ur = basic_map_t{};  // map of unpaired instances in rhs
    auto _bl = basic_bool_t(_lhs.size(), false);  // track if unpaired
    auto _br = basic_bool_t(_rhs.size(), false);
    for(size_t i = 0; i < _lhs.size(); ++i)
    {
        const auto& _l = _lhs.at(i);
        // look for any matches b/t lhs and rhs
        for(size_t j = 0; j < _rhs.size(); ++j)
        {
            const auto& _r = _rhs.at(j);
            if(_l == _r)
            {
                _bl.at(i) = true;
                _br.at(j) = true;
                if(_p.find(i) == _p.end())
                    _p[i] = (*this)(_l);
                _p[i] += (*this)(_r);
            }
        }
        // insert any unmatched instances into unused-lhs map
        if(!_bl.at(i))
            _ul[i] = (*this)(_l);
    }

    for(size_t j = 0; j < _rhs.size(); ++j)
    {
        // insert any unmatched instances into unused-rhs map
        const auto& _r = _rhs.at(j);
        if(!_br.at(j))
            _ur[j] = (*this)(_r);
    }

    // create the final product
    auto _ret = basic_vec_t{};
    auto n    = std::max<size_t>(_lhs.size(), _rhs.size());
    _ret.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        auto _append = [&](auto& _obj) {
            auto itr = _obj.find(i);
            if(itr != _obj.end())
                _ret.emplace_back(itr->second);
            //_ret.emplace_back(std::move(itr->second));
        };
        _append(_p);
        _append(_ul);
        _append(_ur);
    }

    return (*this)(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
template <typename Tp>
std::vector<basic_tree<Tp>>
merge<Type, true>::operator()(const std::vector<std::vector<basic_tree<Tp>>>& _bt,
                              size_t                                          _root)
{
    if(_bt.empty())
        return _bt;

    if(_root >= _bt.size())
        _root = 0;
    auto _ret = _bt.at(_root);
    for(size_t i = 0; i < _ret.size(); ++i)
    {
        if(i == _root)
            continue;
        _ret = (*this)(_ret, _ret.at(i));
    }

    return (*this)(_ret);
}
//
//--------------------------------------------------------------------------------------//
//
/*template <typename Type>
template <typename Tp>
void
merge<Type, true>::operator()(GraphT& _g, ItrT _root, ItrT _rhs)
{
    using pre_order_iterator = typename GraphT::pre_order_iterator;
    using sibling_iterator   = typename GraphT::sibling_iterator;

    auto _equiv = [](ItrT _lhs, ItrT _rhs) {
        if(_lhs->depth() != _rhs->depth())
            return false;
        auto _lhs_id = get_hash_id(get_hash_aliases(), _lhs->id());
        auto _rhs_id = get_hash_id(get_hash_aliases(), _rhs->id());
        return (_lhs_id == _rhs_id);
    };

    for(sibling_iterator ritr = _rhs.begin(); ritr != _rhs.end(); ++ritr)
    {
        bool found = false;
        for(sibling_iterator itr = _root.begin(); itr != _root.end(); ++itr)
        {
            if(_equiv(ritr, itr))
            {
                found = true;
                if(itr != ritr)
                {
                    itr->data() += ritr->data();
                    itr->data().plus(ritr->data());
                    itr->stats() += ritr->stats();
                }
            }
        }
        if(!found)
        {
            pre_order_iterator citr = ritr;
            _g.append_child(_root, citr);
        }
    }
}*/
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
merge<Type, false>::merge(storage_type& lhs, storage_type& rhs)
{
    rhs.stack_clear();

    // create lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);
    if(!l.owns_lock())
        l.lock();

    for(const auto& itr : *rhs.get_hash_ids())
    {
        if(lhs.m_hash_ids->find(itr.first) == lhs.m_hash_ids->end())
            (*lhs.m_hash_ids)[itr.first] = itr.second;
    }
    for(const auto& itr : (*rhs.get_hash_aliases()))
    {
        if(lhs.m_hash_aliases->find(itr.first) == lhs.m_hash_aliases->end())
            (*lhs.m_hash_aliases)[itr.first] = itr.second;
    }
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
}  // namespace operation
}  // namespace tim
