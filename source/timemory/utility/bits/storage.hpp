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
compute_percentage(const std::tuple<_Types...>&, const std::tuple<_Types...>&)
{
    std::tuple<_Types...> _one;
    return _one;
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
    using pre_order_iterator = typename graph_t::pre_order_iterator;

    // don't merge self
    if(itr == this)
        return;

    // if merge was not initialized return
    if(itr && !itr->is_initialized())
        return;

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
        graph().insert_subgraph_after(_data().head(), itr->data().head());
        /*
        pre_order_iterator _nitr = itr->data().head();
        if(_nitr.begin())
            _nitr = _nitr.begin();
        graph().append_child(_data().head(), _nitr);
        */
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
                pre_order_iterator _pos   = _titr;
                pre_order_iterator _other = _nitr.begin();
                graph().append_child(_pos, _other);
                _merged = true;
                break;
            }

            if(!_merged)
            {
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
void
storage<ObjectType, true>::mpi_reduce()
{}

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

        // disable gperf if profiling
#if defined(TIMEMORY_USE_GPERF) || defined(TIMEMORY_USE_GPERF_CPU_PROFILER) ||           \
    defined(TIMEMORY_USE_GPERF_HEAP_PROFILER)
        try
        {
            if(details::storage_once_flag()++ == 0)
                gperf::profiler_stop();
        } catch(std::exception& e)
        {
            std::cerr << "Error calling gperf::profiler_stop(): " << e.what()
                      << ". Continuing..." << std::endl;
        }
#endif

        if(!_file_output && !_cout_output && !_json_forced)
        {
            instance_count().store(0);
            return;
        }

        auto _results = this->get();

#if defined(DEBUG)
        if(tim::settings::debug() && tim::settings::verbose() > 3)
        {
            printf("\n");
            size_t w = 0;
            for(const auto& itr : _results)
                w = std::max<size_t>(w, std::get<2>(itr).length());
            for(const auto& itr : _results)
            {
                std::cout << std::setw(w) << std::left << std::get<2>(itr) << " : "
                          << std::get<1>(itr);
                auto _hierarchy = std::get<5>(itr);
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
        for(const auto& itr : _results)
        {
            const auto& itr_obj    = std::get<1>(itr);
            const auto& itr_prefix = std::get<2>(itr);
            const auto& itr_depth  = std::get<3>(itr);
            if(itr_depth < 0 || itr_depth > settings::max_depth())
                continue;
            int64_t _len = itr_prefix.length();
            _width       = std::max(_len, _width);
            _max_depth   = std::max<int64_t>(_max_depth, itr_depth);
            _max_laps    = std::max<int64_t>(_max_laps, itr_obj.nlaps());
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
            auto jname = settings::compose_output_filename(label, ".json", m_node_init,
                                                           &m_node_rank);
            if(jname.length() > 0)
            {
                printf("[%s]> Outputting '%s'...\n", ObjectType::label().c_str(),
                       jname.c_str());
                serialize_storage(jname, *this, num_instances, m_node_rank);
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
        if(_cout_output)
        {
            cout = &std::cout;
            printf("\n");
        }

        // std::stringstream _mss;
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
                if(eitr == _results.end())
                    continue;
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
                            details::combine(exclusive_values, eitr_obj.get());
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
                    details::print_percentage(_pss, details::compute_percentage(
                                                        exclusive_values, itr_obj.get()));
                }
            }

            auto _laps = itr_obj.nlaps();

            std::stringstream _oss;
            operation::print<ObjectType>(itr_obj, _oss, itr_prefix, _laps, itr_depth,
                                         _widths, true, _pss.str());
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
            printf("\n");
            uint64_t _nitr = 0;
            for(auto& itr : _results)
            {
                auto& itr_depth = std::get<3>(itr);

                if(itr_depth < 0 || itr_depth > settings::max_depth())
                    continue;

                // if only a specific number of measurements should be echoed
                if(settings::dart_count() > 0 && _nitr >= settings::dart_count())
                    continue;

                auto& itr_obj       = std::get<1>(itr);
                auto& itr_hierarchy = std::get<5>(itr);
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
    auto&& graph_list = get();
    if(graph_list.size() == 0)
        return;

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
    auto&& graph_list = get();
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

#include "timemory/manager.hpp"
//#include "timemory/utility/singleton.hpp"

namespace tim
{
namespace impl
{
//--------------------------------------------------------------------------------------//

template <typename ObjectType>
void
storage<ObjectType, true>::get_shared_manager()
{
    m_manager         = ::tim::manager::instance();
    using func_t      = ::tim::manager::finalizer_func_t;
    bool   _is_master = singleton_t::is_master(this);
    func_t _finalize  = [&]() { this_type::get_singleton().reset(this); };
    m_manager->add_finalizer(std::move(_finalize), _is_master);
}

//--------------------------------------------------------------------------------------//

template <typename ObjectType>
void
storage<ObjectType, false>::get_shared_manager()
{
    m_manager         = ::tim::manager::instance();
    using func_t      = ::tim::manager::finalizer_func_t;
    bool   _is_master = singleton_t::is_master(this);
    func_t _finalize  = [&]() { this_type::get_singleton().reset(this); };
    m_manager->add_finalizer(std::move(_finalize), _is_master);
}

//--------------------------------------------------------------------------------------//
}  // namespace impl
}  // namespace tim
//======================================================================================//
