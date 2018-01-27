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
//

/** \file timing_manager.hpp
 * Static singleton handler of auto-timers
 *
 */

#ifndef timing_manager_hpp_
#define timing_manager_hpp_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wuninitialized"

//----------------------------------------------------------------------------//

#include <unordered_map>
#include <deque>
#include <string>
#include <thread>
#include <mutex>
#include <cstdint>

#ifdef _OPENMP
#   include <omp.h>
#endif

#include <cereal/cereal.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/access.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/chrono.hpp>
#include <cereal/archives/adapters.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>

#include "timemory/namespace.hpp"
#include "timemory/utility.hpp"
#include "timemory/timer.hpp"
#include "timemory/mpi.hpp"

namespace NAME_TIM
{

//----------------------------------------------------------------------------//

inline int32_t get_max_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

//----------------------------------------------------------------------------//

namespace internal
{
typedef std::tuple<uint64_t, uint64_t, std::string,
                   std::shared_ptr<NAME_TIM::timer>> base_timer_tuple_t;
}

struct timer_tuple : public internal::base_timer_tuple_t
{
    typedef timer_tuple                     this_type;
    typedef std::string                     string_t;
    typedef NAME_TIM::timer           tim_timer_t;
    typedef std::shared_ptr<tim_timer_t>    timer_ptr_t;
    typedef uint64_t                        first_type;
    typedef uint64_t                        second_type;
    typedef string_t                        third_type;
    typedef timer_ptr_t                     fourth_type;
    typedef internal::base_timer_tuple_t     base_type;

    timer_tuple(const base_type& _data) : base_type(_data) { }
    timer_tuple(first_type _b, second_type _s, third_type _t, fourth_type _f)
    : base_type(_b, _s, _t, _f) { }

    timer_tuple& operator=(const base_type& rhs)
    {
        if(this == &rhs)
            return *this;
        base_type::operator =(rhs);
        return *this;
    }

    bool operator==(const this_type& rhs) const
    {
        return (key() == rhs.key() && level() == rhs.level() &&
                tag() == rhs.tag());
    }

    bool operator!=(const this_type& rhs) const
    {
        return !(*this == rhs);
    }

    this_type& operator+=(const this_type& rhs)
    {
        timer() += rhs.timer();
        return *this;
    }

    const this_type operator+(const this_type& rhs) const
    {
        return this_type(*this) += rhs;
    }

    first_type& key() { return std::get<0>(*this); }
    const first_type& key() const { return std::get<0>(*this); }

    second_type& level() { return std::get<1>(*this); }
    const second_type& level() const { return std::get<1>(*this); }

    third_type tag() { return std::get<2>(*this); }
    const third_type tag() const { return std::get<2>(*this); }

    tim_timer_t& timer() { return *(std::get<3>(*this).get()); }
    const tim_timer_t& timer() const { return *(std::get<3>(*this).get()); }

    // serialization function
    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(cereal::make_nvp("timer.key", key()),
           cereal::make_nvp("timer.level", level()),
           cereal::make_nvp("timer.tag", tag()),
           cereal::make_nvp("timer.ref", timer()));
    }

    friend std::ostream& operator<<(std::ostream& os, const timer_tuple& t)
    {
        std::stringstream ss;
        ss << "Key = " << std::get<0>(t) << ", "
           << "Count = " << std::get<1>(t) << ", "
           << "Tag = " << std::get<2>(t);
        os << ss.str();
        return os;
    }

};

//----------------------------------------------------------------------------//

class timing_manager
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef timing_manager                  this_type;
    typedef NAME_TIM::timer                 tim_timer_t;
    typedef std::shared_ptr<tim_timer_t>    timer_ptr_t;
    typedef tim_timer_t::string_t           string_t;
    typedef timer_tuple                     timer_tuple_t;
    typedef std::deque<timer_tuple_t>       timer_list_t;
    typedef timer_list_t::iterator          iterator;
    typedef timer_list_t::const_iterator    const_iterator;
    typedef timer_list_t::size_type         size_type;
    typedef uomap<uint64_t, timer_ptr_t>    timer_map_t;
    typedef tim_timer_t::ostream_t          ostream_t;
    typedef tim_timer_t::ofstream_t         ofstream_t;
    typedef NAME_TIM::timer_field           timer_field;
    typedef std::tuple<MPI_Comm, int32_t>   comm_group_t;
    typedef std::mutex                      mutex_t;
    typedef uomap<uint64_t, mutex_t>        mutex_map_t;
    typedef std::lock_guard<mutex_t>        auto_lock_t;
    typedef this_type*                      pointer_type;
    typedef std::set<this_type*>            daughter_list_t;
    typedef tim_timer_t::rss_usage_t        rss_usage_t;

public:
    // Constructor and Destructors
    timing_manager();
    virtual ~timing_manager();

public:
    // Public static functions
    static pointer_type instance();
    static bool is_enabled() { return f_enabled; }
    static void enable(bool val = true);
    static void write_json(string_t _fname);
    static std::pair<int32_t, bool> write_json(ostream_t& os);
    static int32_t& max_depth() { return f_max_depth; }
    void merge(bool div_clock = true);

protected:
    static void write_json_no_mpi(string_t _fname);
    static void write_json_mpi(string_t _fname);
    static void write_json_no_mpi(ostream_t& os);
    static std::pair<int32_t, bool> write_json_mpi(ostream_t& os);

public:
    // Public member functions
    size_type size() const { return m_timer_list.size(); }
    void clear();

    tim_timer_t& timer(const string_t& key,
                       const string_t& tag = "cxx",
                       int32_t ncount = 0,
                       int32_t nhash = 0);

    tim_timer_t& at(size_t i) { return m_timer_list.at(i).timer(); }

    // time a function with a return type and no arguments
    template <typename _Ret, typename _Func>
    _Ret time(const string_t& key, _Func);

    // time a function with a return type and arguments
    template <typename _Ret, typename _Func, typename... _Args>
    _Ret time(const string_t& key, _Func, _Args...);

    // time a function with no return type and no arguments
    template <typename _Func>
    void time(const string_t& key, _Func);

    // time a function with no return type and arguments
    template <typename _Func, typename... _Args>
    void time(const string_t& key, _Func, _Args...);

    // serialization function
    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/);

    // iteration of timers
    iterator        begin()         { return m_timer_list.begin(); }
    const_iterator  begin() const   { return m_timer_list.cbegin(); }
    const_iterator  cbegin() const  { return m_timer_list.cbegin(); }

    iterator        end()           { return m_timer_list.end(); }
    const_iterator  end() const     { return m_timer_list.cend(); }
    const_iterator  cend() const    { return m_timer_list.cend(); }

    void report(bool no_min = false) const;
    void report(std::ostream& os, bool no_min = false) const { report(&os, no_min); }
    void set_output_stream(ostream_t&);
    void set_output_stream(const string_t&);
    void print(bool no_min = false) { this->report(no_min); }
    void set_max_depth(int32_t d) { f_max_depth = d; }
    int32_t get_max_depth() { return f_max_depth; }
    void write_serialization(string_t _fname) const { write_json(_fname); }
    void write_serialization(std::ostream& os) const;

    void add(pointer_type ptr);

    timer_map_t& map() { return m_timer_map; }
    timer_list_t& list() { return m_timer_list; }

    const timer_map_t& map() const { return m_timer_map; }
    const timer_list_t& list() const { return m_timer_list; }

    uint64_t& hash() { return m_hash; }
    uint64_t& count() { return m_count; }

    const uint64_t& hash() const { return m_hash; }
    const uint64_t& count() const { return m_count; }

    daughter_list_t& daughters() { return m_daughters; }
    const daughter_list_t& daughters() const { return m_daughters; }

    void set_merge(bool val) { m_merge.store(val); }

    void operator+=(const rss_usage_t& rhs);
    void operator-=(const rss_usage_t& rhs);

protected:
    // protected functions
    inline uint64_t string_hash(const string_t&) const;
    string_t get_prefix() const;
    void const_merge(bool div = true) const { const_cast<this_type*>(this)->merge(div); }

protected:
    // protected static functions and vars
    static comm_group_t get_communicator_group();
    static mutex_t  f_mutex;

private:
    // Private functions
    ofstream_t* get_ofstream(ostream_t* m_os) const;
    void report(ostream_t*, bool no_min = false) const;

private:
    // Private variables
    static pointer_type     f_instance;
    // for temporary enabling/disabling
    static bool             f_enabled;
    // max depth of timers
    static int32_t          f_max_depth;
    // merge checking
    std::atomic<bool>       m_merge;
    // hash counting
    uint64_t                m_hash;
    // auto timer counting
    uint64_t                m_count;
    // hashed string map for fast lookup
    timer_map_t             m_timer_map;
    // ordered list for output (outputs in order of timer instantiation)
    timer_list_t            m_timer_list;
    // output stream for total timing report
    ostream_t*              m_report;
    // mutex
    mutex_t                 m_mutex;
    // daughter list
    daughter_list_t         m_daughters;
    // baseline rss
    rss_usage_t             m_rss_usage;
};

//----------------------------------------------------------------------------//
inline void
timing_manager::operator+=(const rss_usage_t& rhs)
{
    for(auto& itr : m_timer_list)
        *(std::get<3>(itr).get()) += rhs;
}
//----------------------------------------------------------------------------//
inline void
timing_manager::operator-=(const rss_usage_t& rhs)
{
    for(auto& itr : m_timer_list)
        *(std::get<3>(itr).get()) -= rhs;
}
//----------------------------------------------------------------------------//
template <typename _Ret, typename _Func>
inline _Ret
timing_manager::time(const string_t& key, _Func func)
{
    tim_timer_t& _t = this->instance()->timer(key);
    _t.start();
    _Ret _ret = func();
    _t.stop();
    return _ret;
}
//----------------------------------------------------------------------------//
template <typename _Ret, typename _Func, typename... _Args>
inline _Ret
timing_manager::time(const string_t& key, _Func func, _Args... args)
{
    tim_timer_t& _t = this->instance()->timer(key);
    _t.start();
    _Ret _ret = func(args...);
    _t.stop();
    return _ret;
}
//----------------------------------------------------------------------------//
template <typename _Func>
inline void
timing_manager::time(const string_t& key, _Func func)
{
    tim_timer_t& _t = this->instance()->timer(key);
    _t.start();
    func();
    _t.stop();
}
//----------------------------------------------------------------------------//
template <typename _Func, typename... _Args>
inline void
timing_manager::time(const string_t& key, _Func func, _Args... args)
{
    tim_timer_t& _t = this->instance()->timer(key);
    _t.start();
    func(args...);
    _t.stop();
}
//----------------------------------------------------------------------------//
template <typename Archive>
inline void
timing_manager::serialize(Archive& ar, const unsigned int /*version*/)
{
    uint32_t omp_nthreads = get_max_threads();
    ar(cereal::make_nvp("omp_concurrency", omp_nthreads));
    ar(cereal::make_nvp("timers", m_timer_list));
}
//----------------------------------------------------------------------------//
inline uint64_t
timing_manager::string_hash(const string_t& str) const
{
    return std::hash<string_t>()(str);
}
//----------------------------------------------------------------------------//

} // namespace NAME_TIM

#pragma GCC diagnostic pop

#endif // timing_manager_hpp_
