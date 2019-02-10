// MIT License
//
// Copyright (c) 2018, The Regents of the University of California,
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

#include "timemory/graph.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpi.hpp"
#include "timemory/serializer.hpp"
#include "timemory/singleton.hpp"
#include "timemory/string.hpp"
#include "timemory/timer.hpp"
#include "timemory/utility.hpp"

//--------------------------------------------------------------------------------------//

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

//--------------------------------------------------------------------------------------//

namespace tim
{
//--------------------------------------------------------------------------------------//

namespace internal
{
typedef std::tuple<uint64_t, uint64_t, uint64_t, tim::string, std::shared_ptr<tim::timer>>
    base_timer_tuple_t;
}

//--------------------------------------------------------------------------------------//

struct tim_api timer_tuple : public internal::base_timer_tuple_t
{
    typedef timer_tuple                  this_type;
    typedef tim::string                  string_t;
    typedef tim::timer                   tim_timer_t;
    typedef std::shared_ptr<tim_timer_t> timer_ptr_t;
    typedef internal::base_timer_tuple_t base_type;

    //------------------------------------------------------------------------//
    //      constructors
    //
    // default and full initialization
    timer_tuple(uint64_t _a = 0, uint64_t _b = 0, uint64_t _c = 0, string_t _d = "",
                timer_ptr_t _e = timer_ptr_t(nullptr))
    : base_type(_a, _b, _c, _d, _e)
    {
    }
    // copy constructors
    timer_tuple(const timer_tuple&) = default;
    timer_tuple(const base_type& _data)
    : base_type(_data)
    {
    }
    // assignment operator
    timer_tuple& operator=(const timer_tuple& rhs)
    {
        if(this == &rhs)
            return *this;
        internal::base_timer_tuple_t::operator=(rhs);
        return *this;
    }
    // move operator
    timer_tuple& operator=(timer_tuple&& rhs)
    {
        if(this == &rhs)
            return *this;
        internal::base_timer_tuple_t::operator=(std::move(rhs));
        return *this;
    }

    //------------------------------------------------------------------------//
    //
    //
    uint64_t&       key() { return std::get<0>(*this); }
    const uint64_t& key() const { return std::get<0>(*this); }

    uint64_t&       level() { return std::get<1>(*this); }
    const uint64_t& level() const { return std::get<1>(*this); }

    uint64_t&       offset() { return std::get<2>(*this); }
    const uint64_t& offset() const { return std::get<2>(*this); }

    string_t        tag() { return std::get<3>(*this); }
    const string_t& tag() const { return std::get<3>(*this); }

    tim_timer_t&       timer() { return *(std::get<4>(*this).get()); }
    const tim_timer_t& timer() const { return *(std::get<4>(*this).get()); }

    //------------------------------------------------------------------------//
    //
    //
    timer_tuple& operator=(const base_type& rhs)
    {
        if(this == &rhs)
            return *this;
        base_type::operator=(rhs);
        return *this;
    }

    //------------------------------------------------------------------------//
    //
    //
    friend bool operator==(const this_type& lhs, const this_type& rhs)
    {
        return (lhs.key() == rhs.key() && lhs.level() == rhs.level() &&
                lhs.tag() == rhs.tag());
    }

    //------------------------------------------------------------------------//
    //
    //
    friend bool operator!=(const this_type& lhs, const this_type& rhs)
    {
        return !(lhs == rhs);
    }

    //------------------------------------------------------------------------//
    //
    //
    this_type& operator+=(const this_type& rhs)
    {
        timer() += rhs.timer();
        return *this;
    }

    //------------------------------------------------------------------------//
    //
    //
    const this_type operator+(const this_type& rhs) const
    {
        return this_type(*this) += rhs;
    }

    //------------------------------------------------------------------------//
    // serialization function
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(serializer::make_nvp("timer.key", key()),
           serializer::make_nvp("timer.level", level()),
           serializer::make_nvp("timer.offset", offset()),
           serializer::make_nvp("timer.tag", std::string(tag().c_str())),
           serializer::make_nvp("timer.ref", timer()));
    }

    //------------------------------------------------------------------------//
    //
    //
    friend std::ostream& operator<<(std::ostream& os, const timer_tuple& t)
    {
        std::stringstream ss;
        ss << std::setw(2 * t.level()) << ""
           << "[" << t.tag() << "]" << std::endl;
        /*
        ss << "["
           << "Key = " << t.key() << ", "
           << "Count = " << t.level() << ", "
           << "Offset = " << t.offset() << ", "
           << "Tag = " << t.tag()
           << "]";
        */
        os << ss.str();
        return os;
    }
};

//--------------------------------------------------------------------------------------//
tim_api class manager
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef manager                          this_type;
    typedef tim::timer                       tim_timer_t;
    typedef singleton<manager>               singleton_t;
    typedef singleton<tim_timer_t>           timer_singleton_t;
    typedef std::shared_ptr<tim_timer_t>     timer_ptr_t;
    typedef tim_timer_t::string_t            string_t;
    typedef timer_tuple                      timer_tuple_t;
    typedef std::deque<timer_tuple_t>        timer_list_t;
    typedef timer_list_t::iterator           iterator;
    typedef timer_list_t::const_iterator     const_iterator;
    typedef timer_list_t::size_type          size_type;
    typedef std::pair<uint64_t, timer_ptr_t> timer_pair_t;
    typedef uomap<uint64_t, timer_ptr_t>     timer_map_t;
    typedef tim::graph<timer_tuple_t>        timer_graph_t;
    typedef typename timer_graph_t::iterator timer_graph_itr;
    typedef tim_timer_t::ostream_t           ostream_t;
    typedef tim_timer_t::ofstream_t          ofstream_t;
    typedef std::tuple<MPI_Comm, int32_t>    comm_group_t;
    typedef std::mutex                       mutex_t;
    typedef uomap<uint64_t, mutex_t>         mutex_map_t;
    typedef std::lock_guard<mutex_t>         auto_lock_t;
    typedef singleton_t::pointer             pointer;
    typedef singleton_t::shared_pointer      shared_pointer;
    typedef std::set<this_type*>             daughter_list_t;
    typedef tim_timer_t::rss_type            rss_type;
    typedef rss_type::base_type              base_rss_type;
    typedef std::function<intmax_t()>        get_num_threads_func_t;
    typedef std::atomic<uint64_t>            counter_t;

public:
    // Constructor and Destructors
    manager();
    virtual ~manager();

public:
    // Public static functions
    static pointer        instance();
    static pointer        master_instance();
    static pointer        noninit_instance();
    static pointer        noninit_master_instance();
    static void           enable(bool val = true) { f_enabled = val; }
    static const int32_t& max_depth() { return f_max_depth; }
    static void           max_depth(const int32_t& val) { f_max_depth = val; }
    static void           set_max_depth(const int32_t& val) { f_max_depth = val; }
    static int32_t        get_max_depth() { return f_max_depth; }
    static bool           is_enabled() { return f_enabled; }
    static void           write_json(path_t _fname);
    static std::pair<int32_t, bool> write_json(ostream_t& os);
    static int  get_instance_count() { return f_manager_instance_count.load(); }
    static void set_get_num_threads_func(get_num_threads_func_t f)
    {
        f_get_num_threads = std::bind(f);
    }

protected:
    static void                     write_json_no_mpi(path_t _fname);
    static void                     write_json_no_mpi(ostream_t& os);
    static void                     write_json_mpi(path_t _fname);
    static std::pair<int32_t, bool> write_json_mpi(ostream_t& os);

public:
    // Public member functions
    tim_timer_t& timer(const string_t& key, const string_t& tag = "cxx",
                       int32_t ncount = 0, int32_t nhash = 0);

    /// time a function with a return type and no arguments
    template <typename _Ret, typename _Func>
    _Ret time(const string_t& key, _Func);

    /// time a function with a return type and arguments
    template <typename _Ret, typename _Func, typename... _Args>
    _Ret time(const string_t& key, _Func, _Args...);

    /// time a function with no return type and no arguments
    template <typename _Func>
    void time(const string_t& key, _Func);

    /// time a function with no return type and arguments
    template <typename _Func, typename... _Args>
    void time(const string_t& key, _Func, _Args...);

    void set_output_stream(const path_t&);
    void set_output_stream(ostream_t& = std::cout);
    void write_missing(const path_t& _fname, tim_timer_t* = nullptr,
                       tim_timer_t* = nullptr, bool rank_zero_only = true);
    void write_missing(ostream_t& = std::cout, tim_timer_t* = nullptr,
                       tim_timer_t* = nullptr, bool rank_zero_only = true);
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
    tim_timer_t&           at(size_t i) { return m_timer_list->at(i).timer(); }
    void                   print(bool ign_cutoff = false, bool endline = true);
    iterator               begin() { return m_timer_list->begin(); }
    const_iterator         begin() const { return m_timer_list->cbegin(); }
    const_iterator         cbegin() const { return m_timer_list->cbegin(); }
    iterator               end() { return m_timer_list->end(); }
    const_iterator         end() const { return m_timer_list->cend(); }
    const_iterator         cend() const { return m_timer_list->cend(); }
    timer_map_t&           map() { return m_timer_map; }
    timer_list_t&          list() { return m_timer_list_norm; }
    const timer_map_t&     map() const { return m_timer_map; }
    const timer_list_t&    list() const { return m_timer_list_norm; }
    counter_t&             hash() { return m_hash; }
    counter_t&             count() { return m_count; }
    counter_t&             parent_hash() { return m_p_hash; }
    counter_t&             parent_count() { return m_p_count; }
    const counter_t&       hash() const { return m_hash; }
    const counter_t&       count() const { return m_count; }
    const counter_t&       parent_hash() const { return m_p_hash; }
    const counter_t&       parent_count() const { return m_p_count; }
    void                   pop_graph();
    timer_graph_itr*       get_graph_iterator() const { return m_graph_itr; }
    timer_graph_t*         get_timer_graph() const { return m_timer_graph; }
    void                   sync_hierarchy();
    daughter_list_t&       daughters() { return m_daughters; }
    const daughter_list_t& daughters() const { return m_daughters; }
    void                   add(pointer ptr);
    void                   remove(pointer ptr);
    void                   set_merge(bool val) { m_merge.store(val); }
    void                   operator+=(const base_rss_type& rhs);
    void                   operator-=(const base_rss_type& rhs);
    bool                   is_reporting_to_file() const;
    ostream_t*             get_output_stream() const { return m_report; }
    tim_timer_t            compute_missing(tim_timer_t* timer_ref = nullptr);
    uint64_t               laps() const { return compute_total_laps(); }
    uint64_t               total_laps() const;
    void                   update_total_timer_format();
    void                   start_total_timer();
    void                   stop_total_timer();
    void                   reset_total_timer();
    tim_timer_t*           missing_timer() const { return m_missing_timer.get(); }
    int32_t                instance_count() const { return m_instance_count; }
    void                   self_cost(bool val);
    bool                   self_cost() const;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version);

    friend std::ostream& operator<<(std::ostream& os, const manager& man)
    {
        std::stringstream ss;
        man.report(ss, true, false);
        os << ss.str();
        return os;
    }

public:
    // serialization function

protected:
    // protected static functions
    static comm_group_t get_communicator_group();

protected:
    // protected functions
    inline uint64_t string_hash(const string_t&) const;
    string_t        get_prefix() const;
    uint64_t        compute_total_laps() const;
    void            insert_global_timer();
    void            compute_self();

protected:
    // protected static variables
    static mutex_t                f_mutex;
    static get_num_threads_func_t f_get_num_threads;

private:
    // private functions
    void        report(ostream_t*, bool ign_cutoff = false, bool endline = true) const;
    ofstream_t* get_ofstream(ostream_t* m_os) const;

private:
    // private static variables
    /// for temporary enabling/disabling
    static bool f_enabled;
    /// max depth of timers
    static int32_t f_max_depth;
    /// number of timing manager instances
    static std::atomic<int> f_manager_instance_count;

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
    /// hash counting
    counter_t m_hash;
    /// auto timer counting
    counter_t m_count;
    /// parent auto timer sync point for hashing
    counter_t m_p_hash;
    /// parent auto timer sync point
    counter_t m_p_count;
    /// hashed string map for fast lookup
    timer_map_t m_timer_map;
    /// ordered list for output (outputs in order of timer instantiation)
    timer_list_t m_timer_list_norm;
    /// ordered list for output (self-cost format)
    timer_list_t m_timer_list_self;
    /// mutex
    mutex_t m_mutex;
    /// daughter list
    daughter_list_t m_daughters;
    /// baseline rss
    base_rss_type m_rss_usage;
    /// missing timer
    timer_ptr_t m_missing_timer;
    /// global timer
    timer_ptr_t m_total_timer;
    /// current timer list (either standard or self)
    timer_list_t* m_timer_list;
    /// output stream for total timing report
    ostream_t* m_report;
    /// graph for storage
    timer_graph_t* m_timer_graph;
    /// current graph iterator
    timer_graph_itr* m_graph_itr;
};

//--------------------------------------------------------------------------------------//
inline void
manager::operator+=(const base_rss_type& rhs)
{
    for(auto& itr : m_timer_list_norm)
    {
        bool _restart = itr.timer().is_running();
        if(_restart)
            itr.timer().stop();
        itr.timer() += rhs;
        if(_restart)
            itr.timer().start();
    }
}
//--------------------------------------------------------------------------------------//
inline void
manager::operator-=(const base_rss_type& rhs)
{
    for(auto& itr : m_timer_list_norm)
    {
        bool _restart = itr.timer().is_running();
        if(_restart)
            itr.timer().stop();
        itr.timer() -= rhs;
        if(_restart)
            itr.timer().start();
    }
}
//--------------------------------------------------------------------------------------//
template <typename _Ret, typename _Func>
inline _Ret
manager::time(const string_t& key, _Func func)
{
    tim_timer_t& _t = this->instance()->timer(key);
    _t.start();
    _Ret _ret = func();
    _t.stop();
    return _ret;
}
//--------------------------------------------------------------------------------------//
template <typename _Ret, typename _Func, typename... _Args>
inline _Ret
manager::time(const string_t& key, _Func func, _Args... args)
{
    tim_timer_t& _t = this->instance()->timer(key);
    _t.start();
    _Ret _ret = func(args...);
    _t.stop();
    return _ret;
}
//--------------------------------------------------------------------------------------//
template <typename _Func>
inline void
manager::time(const string_t& key, _Func func)
{
    tim_timer_t& _t = this->instance()->timer(key);
    _t.start();
    func();
    _t.stop();
}
//--------------------------------------------------------------------------------------//
template <typename _Func, typename... _Args>
inline void
manager::time(const string_t& key, _Func func, _Args... args)
{
    tim_timer_t& _t = this->instance()->timer(key);
    _t.start();
    func(args...);
    _t.stop();
}
//--------------------------------------------------------------------------------------//
template <typename Archive>
inline void
manager::serialize(Archive& ar, const unsigned int /*version*/)
{
    uint32_t _nthreads = f_get_num_threads();
    if(_nthreads == 1)
        _nthreads = f_manager_instance_count;
    bool _self_cost = this->self_cost();
    ar(serializer::make_nvp("concurrency", _nthreads));
    ar(serializer::make_nvp("self_cost", _self_cost));
    ar(serializer::make_nvp("timers", *m_timer_list));
}
//--------------------------------------------------------------------------------------//
inline uint64_t
manager::string_hash(const string_t& str) const
{
    return std::hash<string_t>()(str);
}
//--------------------------------------------------------------------------------------//
inline uint64_t
manager::compute_total_laps() const
{
    uint64_t _laps = 0;
    for(const auto& itr : *this)
        _laps += itr.timer().laps();
    return _laps;
}
//--------------------------------------------------------------------------------------//
inline uint64_t
manager::total_laps() const
{
    return m_laps + compute_total_laps();
}
//--------------------------------------------------------------------------------------//
inline void
manager::start_total_timer()
{
    m_total_timer->stop();
}
//--------------------------------------------------------------------------------------//
inline void
manager::stop_total_timer()
{
    m_total_timer->stop();
}
//--------------------------------------------------------------------------------------//
inline void
manager::reset_total_timer()
{
    bool _restart = m_total_timer->is_running();
    if(_restart)
        m_total_timer->stop();
    m_total_timer->reset();
    if(_restart)
        m_total_timer->start();
}
//--------------------------------------------------------------------------------------//
inline void
manager::report(ostream_t* os, bool ign_cutoff, bool endline) const
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

    for(const auto& itr : *this)
        if(!itr.timer().is_valid())
            const_cast<tim_timer_t&>(itr.timer()).stop();

    if(mpi_is_initialized())
        *os << "> rank " << mpi_rank() << std::endl;

    // temporarily store output width
    // auto _width = tim::format::timer::default_width();
    // reset output width
    // tim::format::timer::default_width(10);
    // redo output width calc, removing no displayed funcs
    // for(const auto& itr : *this)
    //     if(itr.timer().above_cutoff(ign_cutoff) || ign_cutoff)
    //         tim::format::timer::propose_default_width(
    //             itr.timer().format()->prefix().length());
    // don't make it longer
    // if(_width > 10 && _width < tim::format::timer::default_width())
    //    tim::format::timer::default_width(_width);

    auto format = [&](const timer_tuple_t& node) {
        std::stringstream ss;
        node.timer().report(ss, false, true);
        return ss.str();
    };

    tim::print_graph(*m_timer_graph, format, *os);
    if(endline)
        *os << std::endl;

    // for(auto itr = this->cbegin(); itr != this->cend(); ++itr)
    //    itr->timer().report(*os, (itr + 1 == this->cend()) ? endline : true,
    //    ign_cutoff);

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
inline void
manager::self_cost(bool val)
{
    m_timer_list = (val) ? &m_timer_list_self : &m_timer_list_norm;
}
//--------------------------------------------------------------------------------------//
inline void
manager::pop_graph()
{
    *m_graph_itr = m_timer_graph->parent(*m_graph_itr);
}
//--------------------------------------------------------------------------------------//
inline bool
manager::self_cost() const
{
    return (m_timer_list == &m_timer_list_self);
}
//--------------------------------------------------------------------------------------//
inline manager::size_type
manager::size() const
{
    size_type n = 0;
    for(auto itr = get_timer_graph()->begin(); itr != get_timer_graph()->end(); ++itr)
        ++n;
    return n;
}
//--------------------------------------------------------------------------------------//
inline timer&
manager::timer(const string_t& key, const string_t& tag, int32_t ncount, int32_t nhash)
{
#if defined(DEBUG)
    if(key.find(" ") != string_t::npos)
    {
        std::stringstream ss;
        ss << "tim::manager::" << __FUNCTION__ << " Warning! Space found in tag: \""
           << key << "\"";
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        std::cerr << ss.str() << std::endl;
    }
#endif

    uint64_t ref = (string_hash(key) + string_hash(tag)) * (ncount + 2) * (nhash + 2);

    // if already exists, return it
    if(m_timer_map.find(ref) != m_timer_map.end())
    {
#if defined(HASH_DEBUG)
        for(const auto& itr : m_timer_list_norm)
        {
            tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
            if(&(std::get<3>(itr)) == &(m_timer_map[ref]))
                std::cout << "tim::manager::" << __FUNCTION__ << " Found : " << itr
                          << std::endl;
        }
#endif
        timer_graph_itr _orig = *m_graph_itr;
        for(typename timer_graph_t::sibling_iterator itr = m_graph_itr->begin();
            itr != m_graph_itr->end(); ++itr)
            if(std::get<0>(*itr) == ref)
            {
                *m_graph_itr = itr;
                break;
            }

        if(_orig == *m_graph_itr)
            for(auto itr = m_timer_graph->begin(); itr != m_timer_graph->end(); ++itr)
                if(std::get<0>(*itr) == ref)
                {
                    *m_graph_itr = itr;
                    break;
                }

        if(_orig == *m_graph_itr)
        {
            std::stringstream ss;
            ss << _orig->tag() << " did not find key = " << ref << " in :\n";
            for(typename timer_graph_t::sibling_iterator itr = m_graph_itr->begin();
                itr != m_graph_itr->end(); ++itr)
            {
                ss << itr->key();
                if(std::distance(itr, m_graph_itr->end()) < 1)
                    ss << ", ";
            }
            ss << std::endl;
            std::cerr << ss.str();
        }
        return *(m_timer_map[ref].get());
    }

    // synchronize format with level 1 and make sure MPI prefix is up-to-date
    if(m_timer_list_norm.size() < 2)
        update_total_timer_format();

    std::stringstream ss;
    // designated as [cxx], [pyc], etc.
    ss << get_prefix() << "[" << tag << "] ";

    // indent
    for(int64_t i = 0; i < ncount; ++i)
    {
        if(i + 1 == ncount)
            ss << "|_";
        else
            ss << "  ";
    }

    ss << std::left << key;
    tim::format::timer::propose_default_width(ss.str().length());

    m_timer_map[ref] = timer_ptr_t(new tim_timer_t(
        tim::format::timer(ss.str(), tim::format::timer::default_format(),
                           tim::format::timer::default_unit(),
                           tim::format::timer::default_rss_format(), true)));

    if(m_instance_count > 0)
        m_timer_map[ref]->thread_timing(true);

    std::stringstream tag_ss;
    tag_ss << tag << "_" << std::left << key;
    timer_tuple_t _tuple(ref, ncount, master_instance()->list().size(), tag_ss.str(),
                         m_timer_map[ref]);
    m_timer_list_norm.push_back(_tuple);
    *m_graph_itr = m_timer_graph->append_child(*m_graph_itr, _tuple);

#if defined(HASH_DEBUG)
    std::cout << "tim::manager::" << __FUNCTION__ << " Created : " << _tuple << std::endl;
#endif

    return *(m_timer_map[ref].get());
}
//--------------------------------------------------------------------------------------//

}  // namespace tim
