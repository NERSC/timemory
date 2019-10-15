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

#include "timemory/components.hpp"
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
namespace details
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
bool
is_finite(const _Tp& val)
{
#if defined(_WINDOWS)
    const _Tp _infv = std::numeric_limits<_Tp>::infinity();
    const _Tp _inf  = (val < 0.0) ? -_infv : _infv;
    return (val == val && val != _inf);
#else
    return std::isfinite(val);
#endif
}

//--------------------------------------------------------------------------------------//

inline std::atomic<int>&
storage_once_flag()
{
    static std::atomic<int> _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

template <typename _Pred, typename _Tp>
void
reduce_merge(_Pred lhs, _Pred rhs)
{
    *lhs += *rhs;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct combine_plus
{
    combine_plus(_Tp& lhs, const _Tp& rhs) { lhs += rhs; }
};

//--------------------------------------------------------------------------------------//
//
//      Combining daughter data
//
//--------------------------------------------------------------------------------------//

template <typename... _Types>
void
combine(std::tuple<_Types...>& lhs, const std::tuple<_Types...>& rhs)
{
    using apply_t = std::tuple<combine_plus<_Types>...>;
    apply<void>::access2<apply_t>(lhs, rhs);
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _ExtraArgs,
          template <typename, typename...> class _Container>
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
compute_percentage(const std::tuple<_Types...>&, const std::tuple<_Types...>&)
{
    std::tuple<_Types...> _one;
    return _one;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp, typename... _ExtraArgs,
          template <typename, typename...> class _Container, typename _Ret = _Tp>
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

template <typename _Tp, typename... _ExtraArgs,
          template <typename, typename...> class _Container>
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
{
}

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

}  // namespace details

//--------------------------------------------------------------------------------------//

namespace impl
{
//--------------------------------------------------------------------------------------//
//
//              Storage functions for implemented types
//
//--------------------------------------------------------------------------------------//

template <typename ObjectType>
void
storage<ObjectType, true>::merge(this_type* itr)
{
    // don't merge self
    if(itr == this)
        return;

    // if merge was not initialized return
    if(itr && !itr->is_initialized())
        return;
    // create lock but don't immediately lock
    auto_lock_t l(type_mutex<this_type>(), std::defer_lock);

    // lock if not already owned
    if(!l.owns_lock())
        l.lock();

    for(const auto& _itr : (*itr->get_hash_ids()))
        if(m_hash_ids->find(_itr.first) == m_hash_ids->end())
            (*m_hash_ids)[_itr.first] = _itr.second;

    // if self is not initialized but itr is, copy data
    if(itr && itr->is_initialized() && !this->is_initialized())
    {
        graph().insert_subgraph_after(_data().head(), itr->data().head());
        m_initialized = itr->m_initialized;
        m_finalized   = itr->m_finalized;
        return;
    }

    auto _this_beg = graph().begin();
    auto _this_end = graph().end();

    bool _merged = false;
    for(auto _this_itr = _this_beg; _this_itr != _this_end; ++_this_itr)
    {
        if(_this_itr == itr->data().head())
        {
            auto _iter_beg = itr->graph().begin();
            auto _iter_end = itr->graph().end();
            graph().merge(_this_itr, _this_end, _iter_beg, _iter_end, false, true);
            _merged = true;
            break;
        }
    }

    if(_merged)
    {
        using predicate_type = decltype(_this_beg);
        auto _compare        = [](predicate_type lhs, predicate_type rhs) {
            return (*lhs == *rhs);
        };
        auto _reduce = [](predicate_type lhs, predicate_type rhs) { *lhs += *rhs; };
        _this_beg    = graph().begin();
        _this_end    = graph().end();
        graph().reduce(_this_beg, _this_end, _this_beg, _this_end, _compare, _reduce);
    }
    else
    {
        auto_lock_t lerr(type_mutex<decltype(std::cerr)>());
        std::cerr << "Failure to merge graphs!" << std::endl;
        auto g = graph();
        graph().insert_subgraph_after(_data().current(), itr->data().head());
    }
}

//======================================================================================//

template <typename ObjectType>
void storage<ObjectType, true>::external_print(std::false_type)
{
    if(!m_initialized && !m_finalized)
        return;

    auto num_instances = instance_count().load();
    auto requires_json = trait::requires_json<ObjectType>::value;
    auto label         = ObjectType::label();

    if(!singleton_t::is_master(this))
    {
        singleton_t::master_instance()->merge(this);
        finalize();
    }
    else if(settings::auto_output())
    {
        merge();

        {
            auto _iter_beg       = _data().graph().begin();
            auto _iter_end       = _data().graph().end();
            using predicate_type = decltype(_iter_beg);
            auto _compare        = [](predicate_type lhs, predicate_type rhs) {
                return (*lhs == *rhs);
            };
            auto _reduce = [](predicate_type lhs, predicate_type rhs) { *lhs += *rhs; };
            _data().graph().reduce(_iter_beg, _iter_beg, _iter_beg, _iter_end, _compare,
                                   _reduce);
        }

        finalize();

        // no entries
        if(_data().graph().size() <= 1)
        {
            instance_count().store(0);
            return;
        }

        // disable gperf if profiling
        try
        {
            if(details::storage_once_flag()++ == 0)
                gperf::profiler_stop();
        }
        catch(std::exception& e)
        {
#if defined(TIMEMORY_USE_GPERF) || defined(TIMEMORY_USE_GPERF_CPU_PROFILER) ||           \
    defined(TIMEMORY_USE_GPERF_HEAP_PROFILER)
            std::cerr << "Error calling gperf::profiler_stop(): " << e.what()
                      << ". Continuing..." << std::endl;
#else
            consume_parameters(e);
#endif
        }

        int64_t _min = std::numeric_limits<int64_t>::max();
        for(const auto& itr : _data().graph())
            _min = std::min<int64_t>(itr.depth(), _min);

        if(!(_min < 0))
        {
            for(auto& itr : _data().graph())
                itr.depth() -= 1;
        }

        if(!settings::file_output() && !settings::cout_output())
        {
            instance_count().store(0);
            return;
        }

        //------------------------------------------------------------------------------//
        //
        //  Compute the node prefix
        //
        //------------------------------------------------------------------------------//
        auto _get_node_prefix = [&]() {
            if(!m_node_init)
                return std::string("> ");

            // prefix spacing
            static uint16_t width = 1;
            if(m_node_size > 9)
                width = std::max(width, (uint16_t)(log10(m_node_size) + 1));
            std::stringstream ss;
            ss.fill('0');
            ss << "|" << std::setw(width) << m_node_rank << "> ";
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

            int64_t _depth = itr.depth();
            if(_depth > 0)
            {
                for(int64_t ii = 0; ii < _depth - 1; ++ii)
                    _indent += "  ";
                _indent += "|_";
            }

            return _node_prefix + _indent + _prefix;
        };

        _data().current()  = _data().head();
        int64_t _width     = ObjectType::get_width();
        int64_t _max_depth = 0;
        int64_t _max_laps  = 0;
        // find the max width
        for(const auto& itr : _data().graph())
        {
            if(itr.depth() < 0 || itr.depth() > settings::max_depth())
                continue;
            int64_t _len = _compute_modified_prefix(itr).length();
            _width       = std::max(_len, _width);
            _max_depth   = std::max<int64_t>(_max_depth, itr.depth());
            _max_laps    = std::max<int64_t>(_max_laps, itr.obj().nlaps());
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
        if((settings::file_output() && settings::json_output()) || requires_json)
        {
            printf("\n");
            auto jname = settings::compose_output_filename(label, ".json", m_node_init,
                                                           &m_node_rank);
            if(jname.length() > 0)
            {
                printf("[%s]> Outputting '%s'...\n", ObjectType::label().c_str(),
                       jname.c_str());
                serialize_storage(jname, *this, num_instances, m_node_rank);
            }
        }
        else if(settings::file_output() && settings::text_output())
        {
            printf("\n");
        }

        //--------------------------------------------------------------------------//
        // output to text file
        //
        if(settings::file_output() && settings::text_output())
        {
            auto fname = settings::compose_output_filename(label, ".txt");
            if(fname.length() > 0)
            {
                fout = new std::ofstream(fname.c_str());
                if(fout && *fout)
                {
                    printf("[%s]> Outputting '%s'...\n", ObjectType::label().c_str(),
                           fname.c_str());
                }
                else
                {
                    delete fout;
                    fout = nullptr;
                    fprintf(stderr, "[storage<%s>::%s @ %i]> Error opening '%s'...\n",
                            ObjectType::label().c_str(), __FUNCTION__, __LINE__,
                            fname.c_str());
                }
            }
        }

        //--------------------------------------------------------------------------//
        // output to cout
        //
        if(settings::cout_output())
        {
            cout = &std::cout;
            printf("\n");
        }

        // std::stringstream _mss;
        for(auto pitr = _data().graph().begin(); pitr != _data().graph().end(); ++pitr)
        {
            auto itr = *pitr;
            if(itr.depth() < 0 || itr.depth() > settings::max_depth())
                continue;
            std::stringstream _pss;
            // if we are not at the bottom of the call stack (i.e. completely inclusive)
            if(itr.depth() < _max_depth)
            {
                // get the next iteration
                auto eitr = pitr;
                std::advance(eitr, 1);
                // counts the number of non-exclusive values
                int64_t nexclusive = 0;
                // the sum of the exclusive values
                get_return_type exclusive_values;
                // continue while not at end of graph until first sibling is encountered
                while(eitr->depth() != itr.depth() && eitr != _data().graph().end())
                {
                    // if one level down, this is an exclusive value
                    if(eitr->depth() == itr.depth() + 1)
                    {
                        // if first exclusive value encountered: assign; else: combine
                        if(nexclusive == 0)
                            exclusive_values = eitr->obj().get();
                        else
                            details::combine(exclusive_values, eitr->obj().get());
                        // increment. beyond 0 vs. 1, this value plays no role
                        ++nexclusive;
                    }
                    // increment iterator for next while check
                    ++eitr;
                }
                // if there were exclusive values encountered
                if(nexclusive > 0 && trait::is_available<ObjectType>::value)
                {
                    details::print_percentage(
                        _pss,
                        details::compute_percentage(exclusive_values, itr.obj().get()));
                }
            }

            auto _obj    = itr.obj();
            auto _prefix = _compute_modified_prefix(itr);
            auto _laps   = _obj.nlaps();
            auto _depth  = itr.depth();

            std::stringstream _oss;
            operation::print<ObjectType>(_obj, _oss, _prefix, _laps, _depth, _widths,
                                         true, _pss.str());
            // operation::print<ObjectType>(_obj, _mss, false);
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

        // if(cout != nullptr)
        //    *cout << std::endl;

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
            auto get_hierarchy = [&](iterator itr) {
                std::vector<std::string> _hierarchy;
                if(!itr || itr->depth() == 0)
                    return _hierarchy;
                iterator _parent = _data().graph().parent(itr);
                do
                {
                    _hierarchy.push_back(get_prefix(_parent));
                    if(_parent->depth() == 0)
                        break;
                    auto _new_parent = graph_t::parent(_parent);
                    if(_parent == _new_parent)
                        break;
                    _parent = _new_parent;
                } while(_parent && !(_parent->depth() < 0));
                std::reverse(_hierarchy.begin(), _hierarchy.end());
                return _hierarchy;
            };
            printf("\n");
            uint64_t _nitr = 0;
            for(auto itr = _data().graph().begin(); itr != _data().graph().end(); ++itr)
            {
                if(itr->depth() < 0 || itr->depth() > settings::max_depth())
                    continue;
                // if only a specific number of measurements should be echoed
                if(settings::dart_count() > 0 && _nitr >= settings::dart_count())
                    continue;
                auto _obj       = itr->obj();
                auto _hierarchy = get_hierarchy(itr);
                _hierarchy.push_back(get_prefix(itr));
                operation::echo_measurement<ObjectType>(_obj, _hierarchy);
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
                                        const unsigned int version)
{
    using tuple_type = std::tuple<int64_t, ObjectType, string_t, int64_t>;
    using array_type = std::vector<tuple_type>;

    // convert graph to a vector
    auto convert_graph = [&]() {
        array_type _list;
        for(const auto& itr : _data().graph())
        {
            if(itr.depth() < 0)
                continue;
            _list.push_back(
                tuple_type(itr.id(), itr.obj(), get_prefix(itr), itr.depth()));
        }
        return _list;
    };

    auto graph_list = convert_graph();
    ar(serializer::make_nvp("type", ObjectType::label()),
       serializer::make_nvp("description", ObjectType::description()),
       serializer::make_nvp("unit_value", ObjectType::unit()),
       serializer::make_nvp("unit_repr", ObjectType::display_unit()));
    ObjectType::serialization_policy(ar, version);
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(auto& itr : graph_list)
    {
        ar.startNode();
        ar(serializer::make_nvp("hash", std::get<0>(itr)),
           serializer::make_nvp("prefix", std::get<2>(itr)),
           serializer::make_nvp("depth", std::get<3>(itr)),
           serializer::make_nvp("entry", std::get<1>(itr)));
        ar.finishNode();
    }
    ar.finishNode();
}

//======================================================================================//

template <typename ObjectType>
template <typename Archive>
void
storage<ObjectType, true>::serialize_me(std::true_type, Archive& ar,
                                        const unsigned int version)
{
    using tuple_type = std::tuple<int64_t, ObjectType, string_t, int64_t>;
    using array_type = std::vector<tuple_type>;

    // convert graph to a vector
    auto convert_graph = [&]() {
        array_type _list;
        for(const auto& itr : _data().graph())
            _list.push_back(
                tuple_type(itr.id(), itr.obj(), get_prefix(itr), itr.depth()));
        return _list;
    };

    auto graph_list = convert_graph();
    if(graph_list.size() == 0)
        return;
    ObjectType& obj           = std::get<1>(graph_list.front());
    auto        labels        = obj.label_array();
    auto        descripts     = obj.descript_array();
    auto        units         = obj.unit_array();
    auto        display_units = obj.display_unit_array();
    ar(serializer::make_nvp("type", labels),
       serializer::make_nvp("description", descripts),
       serializer::make_nvp("unit_value", units),
       serializer::make_nvp("unit_repr", display_units));
    ObjectType::serialization_policy(ar, version);
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(auto& itr : graph_list)
    {
        ar.startNode();
        ar(serializer::make_nvp("hash", std::get<0>(itr)),
           serializer::make_nvp("prefix", std::get<2>(itr)),
           serializer::make_nvp("depth", std::get<3>(itr)),
           serializer::make_nvp("entry", std::get<1>(itr)));
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
tim::serialize_storage(const std::string& fname, const _Tp& obj, int64_t concurrency,
                       int64_t rank)
{
    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
    // std::stringstream     ss;
    std::ofstream ofs(fname.c_str());
    if(ofs)
    {
        // ensure json write final block during destruction before the file is closed
        //                                  args: precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 2);
        cereal::JSONOutputArchive          oa(ofs, opts);
        oa.setNextName("rank");
        oa.startNode();
        oa(cereal::make_nvp("rank_id", rank));
        oa(cereal::make_nvp("concurrency", concurrency));
        oa(cereal::make_nvp("data", obj));
        oa(cereal::make_nvp("environment", *env_settings::instance()));
        oa.finishNode();
    }
    if(ofs)
        ofs << std::endl;
    ofs.close();
}

//======================================================================================//
