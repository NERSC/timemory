// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file manager.hpp
 * \headerfile manager.hpp "timemory/manager.hpp"
 * Static singleton handler of auto-timers
 *
 */

#pragma once

//--------------------------------------------------------------------------------------//

#include "timemory/apply.hpp"
#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpi.hpp"
#include "timemory/serializer.hpp"
#include "timemory/singleton.hpp"
#include "timemory/storage.hpp"
#include "timemory/utility.hpp"

//--------------------------------------------------------------------------------------//

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

class manager;

namespace details
{
struct manager_deleter;
}

//--------------------------------------------------------------------------------------//

tim_api class manager
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef manager                                      this_type;
    typedef singleton<manager, details::manager_deleter> singleton_t;
    typedef std::size_t                                  size_type;
    typedef std::ostream                                 ostream_t;
    typedef std::ofstream                                ofstream_t;
    typedef std::tuple<MPI_Comm, int32_t>                comm_group_t;
    typedef std::mutex                                   mutex_t;
    typedef uomap<uintmax_t, mutex_t>                    mutex_map_t;
    typedef std::lock_guard<mutex_t>                     auto_lock_t;
    typedef singleton_t::pointer                         pointer;
    typedef singleton_t::unique_pointer                  unique_pointer;
    typedef std::set<this_type*>                         daughter_list_t;
    typedef std::function<intmax_t()>                    get_num_threads_func_t;
    typedef std::atomic<uintmax_t>                       counter_t;

public:
    // Constructor and Destructors
    manager();
    virtual ~manager();

public:
    // Public static functions
    static pointer                  instance();
    static pointer                  master_instance();
    static pointer                  noninit_instance();
    static pointer                  noninit_master_instance();
    static void                     write_json(path_t _fname);
    static std::pair<int32_t, bool> write_json(ostream_t& os);
    static int  get_instance_count() { return f_manager_instance_count().load(); }
    static void set_get_num_threads_func(get_num_threads_func_t f)
    {
        f_get_num_threads() = std::bind(f);
    }

    static void    enable(bool val = true) { counted_object<void>::enable(val); }
    static bool    is_enabled() { return counted_object<void>::enable(); }
    static int32_t max_depth() { return counted_object<void>::max_depth(); }
    static void    max_depth(const int32_t& val) { set_max_depth(val); }
    static int32_t get_max_depth() { return max_depth(); }
    static void    set_max_depth(const int32_t& val)
    {
        counted_object<void>::set_max_depth(val);
    }

protected:
    static void                     write_json_no_mpi(path_t _fname);
    static void                     write_json_no_mpi(ostream_t& os);
    static void                     write_json_mpi(path_t _fname);
    static std::pair<int32_t, bool> write_json_mpi(ostream_t& os);

public:
    // Public member functions
    template <typename _Tp>
    _Tp& get(const string_t& key, const string_t& tag = "cxx", int32_t ncount = 0,
             int32_t nhash = 0);

    void set_output_stream(const path_t&);
    void set_output_stream(ostream_t& = std::cout);
    void write_report(path_t _fname, bool ign_cutoff = false);
    void write_report(ostream_t& os = std::cout, bool ign_cutoff = false,
                      bool endline = true);
    void write_serialization(const path_t& _fname) const { write_json(_fname); }
    void write_serialization(ostream_t& os = std::cout) const { write_json(os); }
    void report(bool ign_cutoff = false, bool endline = true) const;
    void report(ostream_t& os, bool ign_cutoff = false, bool endline = true) const;

    void                   merge();
    void                   merge(pointer);
    void                   clear();
    size_type              size() const;
    void                   print(bool ign_cutoff = false, bool endline = true);
    void                   sync_hierarchy();
    daughter_list_t&       daughters() { return m_daughters; }
    const daughter_list_t& daughters() const { return m_daughters; }
    void                   add(pointer ptr);
    void                   remove(pointer ptr);
    void                   set_merge(bool val) { m_merge.store(val); }
    bool                   is_reporting_to_file() const;
    ostream_t*             get_output_stream() const { return m_report; }
    uintmax_t              laps() const { return compute_total_laps(); }
    uintmax_t              total_laps() const;
    void                   update_total_timer_format();
    int32_t                instance_count() const { return m_instance_count; }
    void                   self_cost(bool) {}
    bool                   self_cost() const { return false; }

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version);

    friend std::ostream& operator<<(std::ostream& os, const manager& man)
    {
        std::stringstream ss;
        man.report(ss, true, false);
        os << ss.str();
        return os;
    }

    /*template <typename _Tp, enable_if_t<std::is_same<_Tp, tim::timer>::value, int> = 0>
    void pop_graph()
    {
        //timer_data.pop_graph();
    }

    template <typename _Tp, enable_if_t<std::is_same<_Tp, tim::usage>::value, int> = 0>
    void pop_graph()
    {
        memory_data.pop_graph();
    }*/

public:
    // serialization function

protected:
    // protected static functions
    static comm_group_t get_communicator_group();

protected:
    // protected functions
    inline uintmax_t string_hash(const string_t&) const;
    string_t         get_prefix() const;
    uintmax_t        compute_total_laps() const;
    void             insert_global_timer();
    void             compute_self();

protected:
    // protected static variables
    static get_num_threads_func_t& f_get_num_threads();

private:
    // private functions
    void        report(ostream_t*, bool ign_cutoff = false, bool endline = true) const;
    ofstream_t* get_ofstream(ostream_t* m_os) const;

private:
    // private static variables
    /// for temporary enabling/disabling
    // static bool f_enabled();
    /// number of timing manager instances
    static std::atomic_int& f_manager_instance_count();

private:
    // private variables
    /// merge checking
    std::atomic<bool> m_merge;
    /// self format
    bool m_self_format;
    /// instance id
    int32_t m_instance_count;
    /// total laps
    counter_t m_laps;
    /// mutex
    mutex_t m_mutex;
    /// daughter list
    daughter_list_t m_daughters;
    /// output stream for total timing report
    ostream_t* m_report;

public:
    // typedef data_storage<tim::timer>                timer_data_t;
    // typedef data_storage<tim::usage>                memory_data_t;
    // typedef std::tuple<timer_data_t, memory_data_t> tuple_data_t;

private:
    // tuple_data_t   m_tuple_data;
    // timer_data_t&  timer_data  = std::get<0>(m_tuple_data);
    // memory_data_t& memory_data = std::get<1>(m_tuple_data);

public:
    // tuple_data_t&        get_data() { return m_tuple_data; }
    // timer_data_t&        get_timer_data() { return timer_data; }
    // memory_data_t&       get_memory_data() { return memory_data; }
    // const tuple_data_t&  get_data() const { return m_tuple_data; }
    // const timer_data_t&  get_timer_data() const { return timer_data; }
    // const memory_data_t& get_memory_data() const { return memory_data; }

private:
    static pointer f_instance;
};

//--------------------------------------------------------------------------------------//
template <typename _Tp>
std::deque<data_tuple<_Tp>>
list_convert(const data_storage<_Tp>& _storage)
{
    std::deque<data_tuple<_Tp>> _list;
    for(const auto& itr : _storage)
        _list.push_back(itr);
    return _list;
}
//--------------------------------------------------------------------------------------//
template <typename Archive>
inline void
manager::serialize(Archive& ar, const unsigned int /*version*/)
{
    auto _nthreads = (f_get_num_threads())();
    if(_nthreads == 1)
        _nthreads = f_manager_instance_count();
    bool _self_cost = this->self_cost();
    ar(serializer::make_nvp("concurrency", _nthreads));
    ar(serializer::make_nvp("self_cost", _self_cost));
    // ar(serializer::make_nvp("timers", list_convert(timer_data)));
    // ar(serializer::make_nvp("memory", list_convert(memory_data)));
}
//--------------------------------------------------------------------------------------//
inline uintmax_t
manager::string_hash(const string_t& str) const
{
    return std::hash<string_t>()(str);
}
//--------------------------------------------------------------------------------------//
inline uintmax_t
manager::total_laps() const
{
    return m_laps + compute_total_laps();
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
std::string
apply_format(const data_tuple<_Tp>& node)
{
    std::stringstream ss;
    node.data().report(ss, false, true);
    return ss.str();
}
//--------------------------------------------------------------------------------------//
inline void
manager::report(ostream_t* os, bool /*ign_cutoff*/, bool /*endline*/) const
{
    const_cast<this_type*>(this)->merge();

    auto check_stream = [&](ostream_t*& _os, const string_t& id) {
        if(_os == &std::cout || _os == &std::cerr)
            return;
        ofstream_t* fos = get_ofstream(_os);
        if(fos && !(fos->is_open() && fos->good()))
        {
            _os = &std::cout;
            tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
            std::cerr << "Output stream for " << id << " is not open/valid. "
                      << "Redirecting to stdout..." << std::endl;
        }
    };

    if(os == m_report)
        check_stream(os, "total timing report");

    // for(const auto& itr : timer_data)
    //    if(!itr.data().is_valid())
    //        const_cast<tim_timer_t&>(itr.data()).stop();

    if(mpi_is_initialized())
        *os << "> rank " << mpi_rank() << std::endl;

    /*auto format = [&](const data_tuple<tim::timer>& node) {
        std::stringstream ss;
        node.data().report(ss, false, true);
        return ss.str();
    };

    tim::print_graph(timer_data.graph(), apply_format<tim::timer>, *os);
    if(endline)
        *os << std::endl;
    tim::print_graph(memory_data.graph(), apply_format<tim::usage>, *os);
    if(endline)
        *os << std::endl;*/

    os->flush();
}
//--------------------------------------------------------------------------------------//
inline manager::ofstream_t*
manager::get_ofstream(ostream_t* m_os) const
{
    return (m_os != &std::cout && m_os != &std::cerr) ? static_cast<ofstream_t*>(m_os)
                                                      : nullptr;
}
//--------------------------------------------------------------------------------------//
inline void
manager::print(bool ign_cutoff, bool endline)
{
    this->report(ign_cutoff, endline);
}
//--------------------------------------------------------------------------------------//
inline void
manager::write_report(ostream_t& os, bool ign_cutoff, bool endline)
{
    report(os, ign_cutoff, endline);
}
//--------------------------------------------------------------------------------------//
inline void
manager::report(ostream_t& os, bool ign_cutoff, bool endline) const
{
    report(&os, ign_cutoff, endline);
}
//--------------------------------------------------------------------------------------//
inline bool
manager::is_reporting_to_file() const
{
    return (m_report != &std::cout) && (m_report != &std::cerr);
}
//--------------------------------------------------------------------------------------//
inline manager::size_type
manager::size() const
{
    // return timer_data.graph().size();
    return 0UL;
}
//--------------------------------------------------------------------------------------//
template <typename _Tp>
inline _Tp&
manager::get(const string_t& key, const string_t& tag, int32_t ncount, int32_t nhash)
{
    // typedef data_tuple<_Tp>             tuple_t;
    // typedef data_storage<_Tp>           storage_t;
    // typedef typename storage_t::graph_t graph_t;
    // typedef _Tp                         value_t;

    // get a reference to the storage_t object in tuple_data
    // storage_t& _data = std::get<index_of<storage_t,
    // tuple_data_t>::value>(m_tuple_data);
    // compute the hash
    uintmax_t ref = (string_hash(key) + string_hash(tag)) * (ncount + 2) * (nhash + 2);
    consume_parameters(std::move(ref));

    // if already exists, return it
    /*if(_data.map().find(ref) != _data.map().end())
    {
        auto& m_graph_itr = _data.current();
        auto  _orig       = m_graph_itr;
        for(typename graph_t::sibling_iterator itr = m_graph_itr.begin();
            itr != m_graph_itr.end(); ++itr)
            if(std::get<0>(*itr) == ref)
            {
                m_graph_itr = itr;
                break;
            }

        if(_orig == m_graph_itr)
            for(auto itr = _data.begin(); itr != _data.end(); ++itr)
                if(std::get<0>(*itr) == ref)
                {
                    m_graph_itr = itr;
                    break;
                }

        if(_orig == m_graph_itr)
        {
            std::stringstream ss;
            ss << _orig->tag() << " did not find key = " << ref << " in :\n";
            for(typename graph_t::sibling_iterator itr = m_graph_itr.begin();
                itr != m_graph_itr.end(); ++itr)
            {
                ss << itr->key();
                if(std::distance(itr, m_graph_itr.end()) < 1)
                    ss << ", ";
            }
            ss << std::endl;
            std::cerr << ss.str();
        }
        return *(_data.map()[ref].get());
    }*/

    // synchronize format with level 1 and make sure MPI prefix is up-to-date
    update_total_timer_format();

    std::stringstream ss;
    // designated as [cxx], [pyc], etc.
    ss << get_prefix() << "[" << tag << "] ";

    // indent
    for(intmax_t i = 0; i < ncount; ++i)
    {
        if(i + 1 == ncount)
            ss << "|_";
        else
            ss << "  ";
    }

    ss << std::left << key;
    // format_t::propose_default_width(ss.str().length());

    //_data.map()[ref] = pointer_t(new value_t(
    //    format_t(ss.str(), format_t::default_format(), format_t::default_unit(),
    //    true)));

    // if(m_instance_count > 0)
    //    _data.map()[ref]->thread_timing(true);

    std::stringstream tag_ss;
    tag_ss << tag << "_" << std::left << key;
    // tuple_t _tuple(ref, ncount, graph_t::depth(_data.current()) + 1, tag_ss.str(),
    //               _data.map()[ref]);
    //_data.current() = _data.graph().append_child(_data.current(), _tuple);

    // return *(_data.map()[ref].get());
}

//======================================================================================//

namespace details
{
//--------------------------------------------------------------------------------------//

struct manager_deleter
{
    using singleton_t = singleton<tim::manager, manager_deleter>;

    void operator()(tim::manager* ptr)
    {
        tim::manager*   master     = singleton_t::master_instance_ptr();
        std::thread::id master_tid = singleton_t::master_thread_id();

        if(std::this_thread::get_id() == master_tid)
            delete ptr;
        else
        {
            if(master && ptr != master)
            {
                master->remove(ptr);
            }
            delete ptr;
        }
    }
};

//--------------------------------------------------------------------------------------//

inline manager::singleton_t&
manager_singleton()
{
    static manager::singleton_t _instance = manager::singleton_t::instance();
    return _instance;
}

//--------------------------------------------------------------------------------------//

}  // namespace details

//======================================================================================//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#include "timemory/manager.icpp"
