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
}  // namespace tim

//--------------------------------------------------------------------------------------//

template <typename ObjectType>
void
tim::storage<ObjectType>::merge(this_type* itr)
{
    if(itr == this)
        return;

    // create lock but don't immediately lock
    auto_lock_t l(singleton_t::get_mutex(), std::defer_lock);

    // lock if not already owned
    if(!l.owns_lock())
        l.lock();

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
            return ((lhs->id() == rhs->id() && lhs->depth() == rhs->depth()) ||
                    (lhs->prefix() == rhs->prefix() && lhs->depth() == rhs->depth()));
        };
        auto _reduce = [](predicate_type lhs, predicate_type rhs) {
            tim::details::reduce_merge<predicate_type, ObjectType>(lhs, rhs);
        };
        _this_beg = graph().begin();
        _this_end = graph().end();
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
void tim::storage<ObjectType>::external_print(std::false_type)
{
    auto num_instances = instance_count().load();

    if(!singleton_t::is_master(this))
    {
        singleton_t::master_instance()->merge(this);
        ObjectType::thread_finalize_policy();
    }
    else if(settings::auto_output())
    {
        merge();

        auto _iter_beg = _data().graph().begin();
        auto _iter_end = _data().graph().end();
        _data().graph().reduce(_iter_beg, _iter_beg, _iter_beg, _iter_end);

        ObjectType::thread_finalize_policy();
        ObjectType::global_finalize_policy();

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

        // fix up the prefix based on the actual depth
        auto _compute_modified_prefix = [](const graph_node& itr) {
            std::string _prefix   = itr.prefix();
            auto        _ebracket = _prefix.find("]");
            auto        _boffset  = _prefix.find("|_");

            // if depth == 0
            if(itr.depth() < 1)
            {
                // account for |_ in the prefix
                if(_boffset != std::string::npos)
                {
                    auto _beg =
                        (_ebracket != std::string::npos) ? (_ebracket + 2) : _boffset;
                    auto _len = (_boffset + 2) - _beg;
                    _prefix   = _prefix.erase(_beg, _len);
                }
                return _prefix;
            }

            // if |_ found
            if(_boffset == std::string::npos)
            {
                for(int64_t _i = 0; _i < itr.depth() - 1; ++_i)
                {
                    _prefix.insert(_ebracket + 2, "  ");
                    _ebracket += 2;
                }
                _prefix.insert(_ebracket + 2, "|_");
            }
            else  // if |_ not found
            {
                int _diff = (_boffset - (_ebracket + 2));
                int _expd = 2 * (itr.depth() - 1);
                // if indent is less than depth
                if(_expd > _diff)
                {
                    int ninsert = (_expd - _diff);
                    for(auto i = 0; i < ninsert; ++i)
                    {
                        _prefix.insert(_ebracket + 1, " ");
                    }
                }
                // if indent is more than depth
                else if(_diff > _expd)
                {
                    int nstrip = (_diff - _expd);
                    for(auto i = 0; i < nstrip; ++i)
                    {
                        _prefix = _prefix.erase(_ebracket + 1, 1);
                    }
                }
            }
            return _prefix;
        };

        _data().current()  = _data().head();
        int64_t _width     = ObjectType::get_width();
        int64_t _max_depth = 0;
        int64_t _max_laps  = 0;
        // find the max width
        for(const auto& itr : _data().graph())
        {
            if(itr.depth() < 0)
                continue;
            int64_t _len = _compute_modified_prefix(itr).length();
            _width       = std::max(_len, _width);
            _max_depth   = std::max<int64_t>(_max_depth, itr.depth());
            _max_laps    = std::max<int64_t>(_max_laps, itr.obj().laps);
        }
        int64_t              _width_laps  = std::log10(_max_laps) + 1;
        int64_t              _width_depth = std::log10(_max_depth) + 1;
        std::vector<int64_t> _widths      = { _width, _width_laps, _width_depth };

        // return type of get() function
        using get_return_type = decltype(std::declval<const ObjectType>().get());

        std::stringstream _oss;
        // std::stringstream _mss;
        for(auto pitr = _data().graph().begin(); pitr != _data().graph().end(); ++pitr)
        {
            auto itr = *pitr;
            if(itr.depth() < 0)
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
                            tim::details::combine(exclusive_values, eitr->obj().get());
                        // increment. beyond 0 vs. 1, this value plays no role
                        ++nexclusive;
                    }
                    // increment iterator for next while check
                    ++eitr;
                }
                // if there were exclusive values encountered
                if(nexclusive > 0 && trait::is_available<ObjectType>::value)
                {
                    tim::details::print_percentage(
                        _pss, tim::details::compute_percentage(exclusive_values,
                                                               itr.obj().get()));
                }
            }

            auto _obj    = itr.obj();
            auto _prefix = _compute_modified_prefix(itr);
            auto _laps   = _obj.laps;
            auto _depth  = itr.depth();

            operation::print<ObjectType>(_obj, _oss, _prefix, _laps, _depth, _widths,
                                         true, _pss.str());
            // operation::print<ObjectType>(_obj, _mss, false);
        }

        if((settings::file_output() || trait::requires_json<ObjectType>::value) &&
           _oss.str().length() > 0)
        {
            printf("\n");
            auto label = ObjectType::label();
            //--------------------------------------------------------------------------//
            // output to text
            //
            if(settings::text_output() && settings::file_output())
            {
                auto fname = tim::settings::compose_output_filename(label, ".txt");
                std::ofstream ofs(fname.c_str());
                if(ofs)
                {
                    auto_lock_t l(type_mutex<std::ofstream>());
                    std::cout << "[" << ObjectType::label() << "]> Outputting '" << fname
                              << "'... " << std::flush;
                    ofs << _oss.str();
                    std::cout << "Done" << std::endl;
                    ofs.close();
                }
                else
                {
                    auto_lock_t l(type_mutex<decltype(std::cout)>());
                    fprintf(stderr, "[storage<%s>::%s @ %i]> Error opening '%s'...\n",
                            ObjectType::label().c_str(), __FUNCTION__, __LINE__,
                            fname.c_str());
                    std::cout << _oss.str();
                }
            }

            //--------------------------------------------------------------------------//
            // output to json
            //
            if(settings::json_output() || trait::requires_json<ObjectType>::value)
            {
                auto_lock_t l(type_mutex<std::ofstream>());
                auto jname = tim::settings::compose_output_filename(label, ".json");
                printf("[%s]> Outputting '%s'... ", ObjectType::label().c_str(),
                       jname.c_str());
                serialize_storage(jname, *this, num_instances);
                printf("Done\n");
            }
        }

        if(settings::cout_output() && _oss.str().length() > 0)
        {
            printf("\n");
            auto_lock_t l(type_mutex<decltype(std::cout)>());
            std::cout << _oss.str() << std::flush;
        }

        if(settings::dart_output() && _oss.str().length() > 0)
        {
            auto get_hierarchy = [&](iterator itr) {
                std::vector<std::string> _hierarchy;
                if(!itr || itr->depth() == 0)
                    return _hierarchy;
                iterator _parent = _data().graph().parent(itr);
                do
                {
                    _hierarchy.push_back(_parent->prefix());
                    if(_parent->depth() == 0)
                        break;
                    auto _new_parent = graph_t::parent(_parent);
                    if(_parent == _new_parent)
                        break;
                    _parent = _new_parent;
                    // _parent = _data().graph().parent(itr);
                } while(_parent && !(_parent->depth() < 0));
                std::reverse(_hierarchy.begin(), _hierarchy.end());
                return _hierarchy;
            };
            printf("\n");
            for(auto itr = _data().graph().begin(); itr != _data().graph().end(); ++itr)
            {
                if(itr->depth() < 0)
                    continue;
                auto _obj       = itr->obj();
                auto _hierarchy = get_hierarchy(itr);
                _hierarchy.push_back(itr->prefix());
                operation::echo_measurement<ObjectType>(_obj, _hierarchy);
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
void tim::storage<ObjectType>::external_print(std::true_type)
{
    if(!singleton_t::is_master(this))
    {
        singleton_t::master_instance()->merge(this);
        ObjectType::thread_finalize_policy();
    }
    else if(settings::auto_output())
    {
        merge();
        ObjectType::thread_finalize_policy();
        ObjectType::global_finalize_policy();
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
tim::storage<ObjectType>::serialize_me(std::false_type, Archive& ar,
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
            _list.push_back(itr);
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
tim::storage<ObjectType>::serialize_me(std::true_type, Archive& ar,
                                       const unsigned int version)
{
    using tuple_type = std::tuple<int64_t, ObjectType, string_t, int64_t>;
    using array_type = std::vector<tuple_type>;

    // convert graph to a vector
    auto convert_graph = [&]() {
        array_type _list;
        for(const auto& itr : _data().graph())
            _list.push_back(itr);
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

template <typename _Tp>
void
tim::serialize_storage(const std::string& fname, const _Tp& obj, int64_t concurrency)
{
    static constexpr auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
    std::stringstream     ss;
    {
        // ensure json write final block during destruction before the file is closed
        //                                  args: precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 4);
        cereal::JSONOutputArchive          oa(ss, opts);
        oa.setNextName("rank");
        oa.startNode();
        auto rank = tim::mpi::rank();
        oa(cereal::make_nvp("rank_id", rank));
        oa(cereal::make_nvp("concurrency", concurrency));
        oa(cereal::make_nvp("data", obj));
        oa(cereal::make_nvp("environment", *env_settings::instance()));
        oa.finishNode();
    }
    std::ofstream ofs(fname.c_str());
    if(ofs)
        ofs << ss.str() << std::endl;
}

//======================================================================================//
