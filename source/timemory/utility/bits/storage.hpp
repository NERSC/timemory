//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
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

/** \file storage.hpp
 * \headerfile storage.hpp "timemory/details/storage.hpp"
 * Provides inline implementation of storage member functions
 *
 */

#pragma once

#include "timemory/components.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/operations.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/settings.hpp"
#include "timemory/utility/macros.hpp"

#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <thread>

//======================================================================================//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace math
{
//--------------------------------------------------------------------------------------//
//
//      Combining daughter data
//
//--------------------------------------------------------------------------------------//

template <typename... _Types>
void
combine(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    apply<void>::plus(lhs, rhs);
}

//--------------------------------------------------------------------------------------//

template <
    typename _Tp, typename... _ExtraArgs,
    template <typename, typename...> class _Container,
    typename _TupleA = _Container<_Tp, _ExtraArgs...>,
    typename _TupleB = std::tuple<_Tp, _ExtraArgs...>,
    typename std::enable_if<!(std::is_same<_TupleA, _TupleB>::value), int>::type = 0>
void
combine(_Container<_Tp, _ExtraArgs...>& lhs, const _Container<_Tp, _ExtraArgs...>& rhs)
{
    auto len = std::min(lhs.size(), rhs.size());
    for(decltype(len) i = 0; i < len; ++i)
        lhs[i] += rhs[i];
}

//--------------------------------------------------------------------------------------//

template <typename _Key, typename _Mapped, typename... _ExtraArgs>
void
combine(std::map<_Key, _Mapped, _ExtraArgs...>&       lhs,
        const std::map<_Key, _Mapped, _ExtraArgs...>& rhs)
{
    for(auto itr : rhs)
    {
        if(lhs.find(itr.first) != lhs.end())
            lhs.find(itr.first)->second += itr.second;
        else
            lhs[itr.first] = itr.second;
    }
}

//--------------------------------------------------------------------------------------//

template <typename _Tp,
          typename std::enable_if<(!std::is_class<_Tp>::value), int>::type = 0>
void
combine(_Tp& lhs, const _Tp& rhs)
{
    lhs += rhs;
}

//--------------------------------------------------------------------------------------//
//
//      Computing percentage that excludes daughters
//
//--------------------------------------------------------------------------------------//

template <typename... _Types>
std::tuple<_Types...>
compute_percentage(const std::tuple<_Types...>& _lhs, const std::tuple<_Types...>& _rhs)
{
    std::tuple<_Types...> _ret;
    apply<void>::percent_diff(_ret, _lhs, _rhs);
    return _ret;
}

//--------------------------------------------------------------------------------------//

template <
    typename _Tp, typename... _ExtraArgs,
    template <typename, typename...> class _Container, typename _Ret = _Tp,
    typename _TupleA = _Container<_Tp, _ExtraArgs...>,
    typename _TupleB = std::tuple<_Tp, _ExtraArgs...>,
    typename std::enable_if<!(std::is_same<_TupleA, _TupleB>::value), int>::type = 0>
_Container<_Ret>
compute_percentage(const _Container<_Tp, _ExtraArgs...>& lhs,
                   const _Container<_Tp, _ExtraArgs...>& rhs)
{
    auto             len = std::min(lhs.size(), rhs.size());
    _Container<_Ret> perc(len, 0.0);
    for(decltype(len) i = 0; i < len; ++i)
    {
        if(rhs[i] > 0)
            perc[i] = (1.0 - (lhs[i] / rhs[i])) * 100.0;
    }
    return perc;
}

//--------------------------------------------------------------------------------------//

template <typename _Key, typename _Mapped, typename... _ExtraArgs>
std::map<_Key, _Mapped, _ExtraArgs...>
compute_percentage(const std::map<_Key, _Mapped, _ExtraArgs...>& lhs,
                   const std::map<_Key, _Mapped, _ExtraArgs...>& rhs)
{
    std::map<_Key, _Mapped, _ExtraArgs...> perc;
    for(auto itr : lhs)
    {
        if(rhs.find(itr.first) != rhs.end())
        {
            auto ritr = rhs.find(itr.first)->second;
            if(itr.second > 0)
                perc[itr.first] = (1.0 - (itr.second / ritr)) * 100.0;
        }
    }
    return perc;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename _Ret = _Tp,
          typename std::enable_if<(!std::is_class<_Tp>::value), int>::type = 0>
_Ret
compute_percentage(_Tp& lhs, const _Tp& rhs)
{
    return (rhs > 0) ? ((1.0 - (lhs / rhs)) * 100.0) : 0.0;
}

//--------------------------------------------------------------------------------------//
//
//      Printing percentage that excludes daughters
//
//--------------------------------------------------------------------------------------//

template <
    typename _Tp, typename... _ExtraArgs,
    template <typename, typename...> class _Container,
    typename _TupleA = _Container<_Tp, _ExtraArgs...>,
    typename _TupleB = std::tuple<_Tp, _ExtraArgs...>,
    typename std::enable_if<!(std::is_same<_TupleA, _TupleB>::value), int>::type = 0>
void
print_percentage(std::ostream& os, const _Container<_Tp, _ExtraArgs...>& obj)
{
    // negative values appear when multiple threads are involved.
    // This needs to be addressed
    for(size_t i = 0; i < obj.size(); ++i)
        if(obj[i] < 0.0 || !is_finite(obj[i]))
            return;

    std::stringstream ss;
    ss << "(exclusive: ";
    for(size_t i = 0; i < obj.size(); ++i)
    {
        ss << std::setprecision(1) << std::fixed << std::setw(5) << obj[i] << "%";
        if(i + 1 < obj.size())
            ss << ", ";
    }
    ss << ")";
    os << ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename _Key, typename _Mapped, typename... _ExtraArgs>
void
print_percentage(std::ostream& os, const std::map<_Key, _Mapped, _ExtraArgs...>& obj)
{
    // negative values appear when multiple threads are involved.
    // This needs to be addressed
    for(auto itr = obj.begin(); itr != obj.end(); ++itr)
        if(itr->second < 0.0 || !is_finite(itr->second))
            return;

    std::stringstream ss;
    ss << "(exclusive: ";
    for(auto itr = obj.begin(); itr != obj.end(); ++itr)
    {
        size_t i = std::distance(obj.begin(), itr);
        ss << std::setprecision(1) << std::fixed << std::setw(5) << itr->second << "%";
        if(i + 1 < obj.size())
            ss << ", ";
    }
    ss << ")";
    os << ss.str();
}

//--------------------------------------------------------------------------------------//

template <typename... _Types>
void
print_percentage(std::ostream&, const std::tuple<_Types...>&)
{}

//--------------------------------------------------------------------------------------//

template <typename _Tp,
          typename std::enable_if<(!std::is_class<_Tp>::value), int>::type = 0>
void
print_percentage(std::ostream& os, const _Tp& obj)
{
    // negative values appear when multiple threads are involved.
    // This needs to be addressed
    if(obj < 0.0 || !is_finite(obj))
        return;

    std::stringstream ss;
    ss << "(exclusive: ";
    ss << std::setprecision(1) << std::fixed << std::setw(5) << obj;
    ss << "%)";
    os << ss.str();
}

//--------------------------------------------------------------------------------------//

}  // namespace math
//--------------------------------------------------------------------------------------//

namespace impl
{
//--------------------------------------------------------------------------------------//
//
//              Storage functions for implemented types
//
//--------------------------------------------------------------------------------------//

template <typename ObjectType>
std::string
storage<ObjectType, true>::get_prefix(const graph_node& node)
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

//======================================================================================//

template <typename ObjectType>
typename storage<ObjectType, true>::graph_data_t&
storage<ObjectType, true>::_data()
{
    using object_base_t = typename ObjectType::base_type;

    if(m_graph_data_instance == nullptr && !m_is_master)
    {
        static bool _data_init = master_instance()->data_init();
        consume_parameters(_data_init);

        auto         m = *master_instance()->current();
        graph_node_t node(m.id(), object_base_t::dummy(), m.depth());
        m_graph_data_instance          = new graph_data_t(node);
        m_graph_data_instance->depth() = m.depth();
        if(m_node_ids.size() == 0)
            m_node_ids[0][0] = m_graph_data_instance->current();
    }
    else if(m_graph_data_instance == nullptr)
    {
        auto_lock_t lk(singleton_t::get_mutex(), std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();

        std::string _prefix = "> [tot] total";
        add_hash_id(_prefix);
        graph_node_t node(0, object_base_t::dummy(), 0);
        m_graph_data_instance          = new graph_data_t(node);
        m_graph_data_instance->depth() = 0;
        if(m_node_ids.size() == 0)
            m_node_ids[0][0] = m_graph_data_instance->current();
    }

    m_initialized = true;
    return *m_graph_data_instance;
}

//======================================================================================//

template <typename ObjectType>
void
storage<ObjectType, true>::merge()
{
    if(!m_is_master || !m_initialized)
        return;

    auto m_children = singleton_t::children();
    if(m_children.size() == 0)
        return;

    for(auto& itr : m_children)
        merge(itr);

    // create lock but don't immediately lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);

    // lock if not already owned
    if(!l.owns_lock())
        l.lock();

    for(auto& itr : m_children)
        if(itr != this)
            itr->data().clear();

    stack_clear();
}

//======================================================================================//

template <typename ObjectType>
void
storage<ObjectType, true>::merge(this_type* itr)
{
    using pre_order_iterator = typename graph_t::pre_order_iterator;

    // don't merge self
    if(itr == this)
        return;

    // if merge was not initialized return
    if(itr && !itr->is_initialized())
        return;

    itr->stack_clear();

    // create lock but don't immediately lock
    // auto_lock_t l(type_mutex<this_type>(), std::defer_lock);
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);

    // lock if not already owned
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
                   ObjectType::label().c_str());
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

    bool _merged = false;
    for(auto _titr = graph().begin(); _titr != graph().end(); ++_titr)
    {
        if(_titr && itr->data().has_head() && *_titr == *itr->data().head())
        {
            typename graph_t::pre_order_iterator _nitr(itr->data().head());
            if(graph().is_valid(_nitr.begin()) && _nitr.begin())
            {
                if(settings::debug())
                    PRINT_HERE("[%s]> worker is merging", ObjectType::label().c_str());
                pre_order_iterator _pos   = _titr;
                pre_order_iterator _other = _nitr.begin();
                graph().append_child(_pos, _other);
                _merged = true;
                break;
            }

            if(!_merged)
            {
                if(settings::debug())
                    PRINT_HERE("[%s]> worker is not merged!",
                               ObjectType::label().c_str());
                ++_nitr;
                if(graph().is_valid(_nitr) && _nitr)
                {
                    graph().append_child(_titr, _nitr);
                    _merged = true;
                    break;
                }
            }
        }
    }

    if(!_merged)
    {
        if(settings::debug())
            PRINT_HERE("[%s]> worker is not merged!", ObjectType::label().c_str());
        pre_order_iterator _nitr(itr->data().head());
        ++_nitr;
        if(!graph().is_valid(_nitr))
            _nitr = pre_order_iterator(itr->data().head());
        graph().append_child(_data().head(), _nitr);
    }

    itr->data().clear();
}

//======================================================================================//

template <typename ObjectType>
typename storage<ObjectType, true>::result_array_t
storage<ObjectType, true>::get()
{
    //------------------------------------------------------------------------------//
    //
    //  Compute the node prefix
    //
    //------------------------------------------------------------------------------//
    auto _get_node_prefix = [&]() {
        if(!m_node_init)
            return std::string(">>> ");

        // prefix spacing
        static uint16_t width = 1;
        if(m_node_size > 9)
            width = std::max(width, (uint16_t)(log10(m_node_size) + 1));
        std::stringstream ss;
        ss.fill('0');
        ss << "|" << std::setw(width) << m_node_rank << ">>> ";
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //
    //  Compute the indentation
    //
    //------------------------------------------------------------------------------//
    // fix up the prefix based on the actual depth
    auto _compute_modified_prefix = [&](const graph_node& itr) {
        std::string _prefix      = get_prefix(itr);
        std::string _indent      = "";
        std::string _node_prefix = _get_node_prefix();

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
        result_array_t _list;
        {
            // the head node should always be ignored
            int64_t _min = std::numeric_limits<int64_t>::max();
            for(const auto& itr : graph())
                _min = std::min<int64_t>(_min, itr.depth());

            for(auto itr = graph().begin(); itr != graph().end(); ++itr)
            {
                if(itr->depth() > _min)
                {
                    auto        _depth   = itr->depth() - (_min + 1);
                    auto        _prefix  = _compute_modified_prefix(*itr);
                    auto        _rolling = itr->id();
                    auto        _parent  = graph_t::parent(itr);
                    strvector_t _hierarchy;
                    if(_parent && _parent->depth() > _min)
                    {
                        while(_parent)
                        {
                            _hierarchy.push_back(get_prefix(*_parent));
                            _rolling += _parent->id();
                            _parent = graph_t::parent(_parent);
                            if(!_parent || !(_parent->depth() > _min))
                                break;
                        }
                    }
                    if(_hierarchy.size() > 1)
                        std::reverse(_hierarchy.begin(), _hierarchy.end());
                    _hierarchy.push_back(get_prefix(*itr));
                    result_node _entry(result_tuple_t{ itr->id(), itr->obj(), _prefix,
                                                       _depth, _rolling, _hierarchy });
                    _list.push_back(_entry);
                }
            }
        }

        bool _thread_scope_only = trait::thread_scope_only<ObjectType>::value;
        if(!settings::collapse_threads() || _thread_scope_only)
            return _list;

        result_array_t _combined;

        //--------------------------------------------------------------------------//
        //
        auto _equiv = [&](const result_node& _lhs, const result_node& _rhs) {
            return (std::get<0>(_lhs) == std::get<0>(_rhs) &&
                    std::get<2>(_lhs) == std::get<2>(_rhs) &&
                    std::get<3>(_lhs) == std::get<3>(_rhs) &&
                    std::get<4>(_lhs) == std::get<4>(_rhs));
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
        for(const auto& itr : _list)
        {
            auto citr = _exists(itr);
            if(citr == _combined.end())
            {
                _combined.push_back(itr);
            }
            else
            {
                std::get<1>(*citr) += std::get<1>(itr);
                std::get<1>(*citr).plus(std::get<1>(itr));
            }
        }
        return _combined;
    };

    return convert_graph();
}

//======================================================================================//

template <typename ObjectType>
typename storage<ObjectType, true>::dmp_result_t
storage<ObjectType, true>::mpi_get()
{
#if !defined(TIMEMORY_USE_MPI)
    if(settings::debug())
        PRINT_HERE("%s", "timemory not using MPI");

    return dmp_result_t(1, get());
#else
    if(settings::debug())
        PRINT_HERE("%s", "timemory using MPI");

    // not yet implemented
    // auto comm =
    //    (settings::mpi_output_per_node()) ? mpi::get_node_comm() : mpi::comm_world_v;
    auto comm = mpi::comm_world_v;
    mpi::barrier(comm);

    int mpi_rank = mpi::rank(comm);
    int mpi_size = mpi::size(comm);
    // int mpi_rank = m_node_rank;
    // int mpi_size = m_node_size;

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [&](const result_array_t& src) {
        std::stringstream ss;
        {
            auto space = cereal::JSONOutputArchive::Options::IndentChar::space;
            cereal::JSONOutputArchive::Options opt(16, space, 0);
            cereal::JSONOutputArchive          oa(ss);
            oa(cereal::make_nvp("data", src));
        }
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //  Used to convert the serialization to a result
    //
    auto recv_serialize = [&](const std::string& src) {
        result_array_t    ret;
        std::stringstream ss;
        ss << src;
        {
            cereal::JSONInputArchive ia(ss);
            ia(cereal::make_nvp("data", ret));
            if(settings::debug())
                printf("[RECV: %i]> data size: %lli\n", mpi_rank,
                       (long long int) ret.size());
        }
        return ret;
    };

    dmp_result_t results(mpi_size);

    auto ret     = get();
    auto str_ret = send_serialize(ret);

    if(mpi_rank == 0)
    {
        for(int i = 1; i < mpi_size; ++i)
        {
            std::string str;
            if(settings::debug())
                printf("[RECV: %i]> starting %i\n", mpi_rank, i);
            mpi::recv(str, i, 0, comm);
            if(settings::debug())
                printf("[RECV: %i]> completed %i\n", mpi_rank, i);
            results[i] = recv_serialize(str);
        }
        results[mpi_rank] = ret;
    }
    else
    {
        if(settings::debug())
            printf("[SEND: %i]> starting\n", mpi_rank);
        mpi::send(str_ret, 0, 0, comm);
        if(settings::debug())
            printf("[SEND: %i]> completed\n", mpi_rank);
        return dmp_result_t(1, ret);
    }

    return results;
#endif
}

//======================================================================================//

template <typename ObjectType>
typename storage<ObjectType, true>::dmp_result_t
storage<ObjectType, true>::upc_get()
{
#if !defined(TIMEMORY_USE_UPCXX)
    if(settings::debug())
        PRINT_HERE("%s", "timemory not using UPC++");

    return dmp_result_t(1, get());
#else
    if(settings::debug())
        PRINT_HERE("%s", "timemory using UPC++");

    upc::barrier();

    int upc_rank = upc::rank();
    int upc_size = upc::size();

    //------------------------------------------------------------------------------//
    //  Used to convert a result to a serialization
    //
    auto send_serialize = [=](const result_array_t& src) {
        std::stringstream ss;
        {
            auto space = cereal::JSONOutputArchive::Options::IndentChar::space;
            cereal::JSONOutputArchive::Options opt(16, space, 0);
            cereal::JSONOutputArchive          oa(ss);
            oa(cereal::make_nvp("data", src));
        }
        return ss.str();
    };

    //------------------------------------------------------------------------------//
    //  Used to convert the serialization to a result
    //
    auto recv_serialize = [=](const std::string& src) {
        result_array_t    ret;
        std::stringstream ss;
        ss << src;
        {
            cereal::JSONInputArchive ia(ss);
            ia(cereal::make_nvp("data", ret));
        }
        return ret;
    };

    //------------------------------------------------------------------------------//
    //  Function executed on remote node
    //
    auto remote_serialize = [=]() {
        return send_serialize(this_type::master_instance()->get());
    };

    dmp_result_t results(upc_size);

    //------------------------------------------------------------------------------//
    //  Combine on master rank
    //
    if(upc_rank == 0)
    {
        for(int i = 1; i < upc_size; ++i)
        {
            upcxx::future<std::string> fut = upcxx::rpc(i, remote_serialize);
            while(!fut.ready())
                upcxx::progress();
            fut.wait();
            results[i] = recv_serialize(fut.result());
        }
        results[upc_rank] = get();
    }

    upcxx::barrier(upcxx::world());

    if(upc_rank != 0)
        return dmp_result_t(1, get());
    else
        return results;
#endif
}

//======================================================================================//

template <typename ObjectType>
void storage<ObjectType, true>::external_print(std::false_type)
{
    base::storage::stop_profiler();

    if(!m_initialized && !m_finalized)
        return;

    auto                  requires_json = trait::requires_json<ObjectType>::value;
    auto                  label         = ObjectType::label();
    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;

    if(!singleton_t::is_master(this))
    {
        singleton_t::master_instance()->merge(this);
        finalize();
    }
    else if(settings::auto_output())
    {
        merge();
        finalize();

        bool _json_forced = requires_json;
        bool _file_output = settings::file_output();
        bool _cout_output = settings::cout_output();
        bool _json_output = settings::json_output() || _json_forced;
        bool _text_output = settings::text_output();

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
        auto _results     = this->get();
        auto _dmp_results = this->dmp_get();
        dmp::barrier();

        if(settings::debug())
            printf("[%s]|%i> dmp results size: %i\n", label.c_str(), m_node_rank,
                   (int) _dmp_results.size());

        // bool return_nonzero_mpi = (dmp::using_mpi() && !settings::mpi_output_per_node()
        // &&
        //                           !settings::mpi_output_per_rank());

        if(_dmp_results.size() > 0)
        {
            if(m_node_rank != 0)
                return;
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
                    std::cout << _hierarchy[i];
                    if(i + 1 < _hierarchy.size())
                        std::cout << "/";
                }
                std::cout << std::endl;
            }
            printf("\n");
        }
#endif

        int64_t _width     = ObjectType::get_width();
        int64_t _max_depth = 0;
        int64_t _max_laps  = 0;
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
                int64_t _len = itr_prefix.length();
                _width       = std::max(_len, _width);
                _max_depth   = std::max<int64_t>(_max_depth, itr_depth);
                _max_laps    = std::max<int64_t>(_max_laps, itr_obj.nlaps());
            }
        }

        int64_t              _width_laps  = std::log10(_max_laps) + 1;
        int64_t              _width_depth = std::log10(_max_depth) + 1;
        std::vector<int64_t> _widths      = { _width, _width_laps, _width_depth };

        // return type of get() function
        using get_return_type = decltype(std::declval<const ObjectType>().get());

        auto_lock_t flk(type_mutex<std::ofstream>(), std::defer_lock);
        auto_lock_t slk(type_mutex<decltype(std::cout)>(), std::defer_lock);

        if(!flk.owns_lock())
            flk.lock();

        if(!slk.owns_lock())
            slk.lock();

        std::ofstream*       fout = nullptr;
        decltype(std::cout)* cout = nullptr;

        //--------------------------------------------------------------------------//
        // output to json file
        //
        if((_file_output && _json_output) || _json_forced)
        {
            printf("\n");
            auto jname = settings::compose_output_filename(label, ".json");
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
                        // ensure json write final block during destruction
                        // before the file is closed
                        //  Option args: precision, spacing, indent size
                        cereal::JSONOutputArchive::Options opts(12, spacing, 2);
                        cereal::JSONOutputArchive          oa(ofs, opts);
                        oa.setNextName("timemory");
                        oa.startNode();
                        oa.setNextName("ranks");
                        oa.startNode();
                        oa.makeArray();
                        for(uint64_t i = 0; i < _dmp_results.size(); ++i)
                        {
                            oa.startNode();
                            oa(cereal::make_nvp("rank", i));
                            oa(cereal::make_nvp("concurrency", num_instances));
                            serial_write_t::serialize(*this, oa, 1, _dmp_results.at(i));
                            oa.finishNode();
                        }
                        oa.finishNode();
                        oa.finishNode();
                    }
                    if(ofs)
                        ofs << std::endl;
                    ofs.close();
                }
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

        for(auto itr = _results.begin(); itr != _results.end(); ++itr)
        {
            auto& itr_obj    = std::get<1>(*itr);
            auto& itr_prefix = std::get<2>(*itr);
            auto& itr_depth  = std::get<3>(*itr);

            if(itr_depth < 0 || itr_depth > settings::max_depth())
                continue;
            std::stringstream _pss;
            // if we are not at the bottom of the call stack (i.e. completely
            // inclusive)
            if(itr_depth < _max_depth)
            {
                // get the next iteration
                auto eitr = itr;
                std::advance(eitr, 1);
                // counts the number of non-exclusive values
                int64_t nexclusive = 0;
                // the sum of the exclusive values
                get_return_type exclusive_values;
                // continue while not at end of graph until first sibling is
                // encountered
                if(eitr != _results.end())
                {
                    auto eitr_depth = std::get<3>(*eitr);
                    while(eitr_depth != itr_depth)
                    {
                        auto& eitr_obj = std::get<1>(*eitr);

                        // if one level down, this is an exclusive value
                        if(eitr_depth == itr_depth + 1)
                        {
                            // if first exclusive value encountered: assign; else:
                            // combine
                            if(nexclusive == 0)
                                exclusive_values = eitr_obj.get();
                            else
                                math::combine(exclusive_values, eitr_obj.get());
                            // increment. beyond 0 vs. 1, this value plays no role
                            ++nexclusive;
                        }
                        // increment iterator for next while check
                        ++eitr;
                        if(eitr == _results.end())
                            break;
                        eitr_depth = std::get<3>(*eitr);
                    }
                    // if there were exclusive values encountered
                    if(nexclusive > 0 && trait::is_available<ObjectType>::value)
                    {
                        math::print_percentage(
                            _pss,
                            math::compute_percentage(exclusive_values, itr_obj.get()));
                    }
                }
            }

            auto _laps = itr_obj.nlaps();

            std::stringstream _oss;
            operation::print<ObjectType>(itr_obj, _oss, itr_prefix, _laps, itr_depth,
                                         _widths, true, _pss.str());
            // for(const auto& itr : itr->hierarchy())
            //    _oss << itr << "//";
            // _oss << "\n";

            if(cout != nullptr)
                *cout << _oss.str() << std::flush;
            if(fout != nullptr)
                *fout << _oss.str() << std::flush;
        }

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
            if(operation::echo_measurement<ObjectType>::lowercase(dtype) !=
               operation::echo_measurement<ObjectType>::lowercase(label))
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

                auto& itr_obj       = itr.data();
                auto& itr_hierarchy = itr.hierarchy();
                operation::echo_measurement<ObjectType>(itr_obj, itr_hierarchy);
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

//======================================================================================//

template <typename ObjectType>
void storage<ObjectType, true>::external_print(std::true_type)
{
    base::storage::stop_profiler();

    if(!singleton_t::is_master(this))
    {
        singleton_t::master_instance()->merge(this);
        finalize();
    }
    else if(settings::auto_output())
    {
        merge();
        finalize();
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

//======================================================================================//

template <typename ObjectType>
template <typename Archive>
void
storage<ObjectType, true>::serialize_me(std::false_type, Archive& ar,
                                        const unsigned int    version,
                                        const result_array_t& graph_list)
{
    if(graph_list.size() == 0)
        return;

    ar(cereal::make_nvp("type", ObjectType::label()),
       cereal::make_nvp("description", ObjectType::description()),
       cereal::make_nvp("unit_value", ObjectType::unit()),
       cereal::make_nvp("unit_repr", ObjectType::display_unit()));
    ObjectType::serialization_policy(ar, version);
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(auto& itr : graph_list)
    {
        ar.startNode();
        ar(cereal::make_nvp("hash", itr.hash()), cereal::make_nvp("prefix", itr.prefix()),
           cereal::make_nvp("depth", itr.depth()), cereal::make_nvp("entry", itr.data()));
        ar.finishNode();
    }
    ar.finishNode();
}

//======================================================================================//

template <typename ObjectType>
template <typename Archive>
void
storage<ObjectType, true>::serialize_me(std::true_type, Archive& ar,
                                        const unsigned int    version,
                                        const result_array_t& graph_list)
{
    if(graph_list.size() == 0)
        return;

    // remove those const in case not marked const
    auto& _graph_list = const_cast<result_array_t&>(graph_list);

    ObjectType& obj           = _graph_list.front().data();
    auto        labels        = obj.label_array();
    auto        descripts     = obj.descript_array();
    auto        units         = obj.unit_array();
    auto        display_units = obj.display_unit_array();
    ar(cereal::make_nvp("type", labels), cereal::make_nvp("description", descripts),
       cereal::make_nvp("unit_value", units),
       cereal::make_nvp("unit_repr", display_units));
    ObjectType::serialization_policy(ar, version);
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(auto& itr : graph_list)
    {
        ar.startNode();
        ar(cereal::make_nvp("hash", itr.hash()), cereal::make_nvp("prefix", itr.prefix()),
           cereal::make_nvp("depth", itr.depth()), cereal::make_nvp("entry", itr.data()));
        ar.finishNode();
    }
    ar.finishNode();
}

//======================================================================================//

}  // namespace impl

//======================================================================================//

}  // namespace tim

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

#include "timemory/manager.hpp"

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

template <typename ObjectType>
void
tim::impl::storage<ObjectType, true>::get_shared_manager()
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
            this->stack_clear();
            this_type::get_singleton().reset(this);
        };
        m_manager->add_finalizer(ObjectType::label(), std::move(_finalize), _is_master);
    }
}

//--------------------------------------------------------------------------------------//

template <typename ObjectType>
void
tim::impl::storage<ObjectType, false>::get_shared_manager()
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
            this->stack_clear();
            this_type::get_singleton().reset(this);
        };
        m_manager->add_finalizer(ObjectType::label(), std::move(_finalize), _is_master);
    }
}

//======================================================================================//
