// MIT License
//
// Copyright (c) 2018 Jonathan R. Madsen
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

#ifndef timing_manager_hpp_
#define timing_manager_hpp_

//----------------------------------------------------------------------------//

#include <unordered_map>
#include <deque>
#include <string>
#include <thread>
#include <mutex>

#include <mpi.h>

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

namespace NAME_TIM
{
namespace util
{

//----------------------------------------------------------------------------//

inline bool mpi_is_initialized()
{
    int32_t _init = 0;
    MPI_Initialized(&_init);
    return (_init != 0) ? true : false;
}

//----------------------------------------------------------------------------//

inline int32_t mpi_rank(MPI_Comm comm = MPI_COMM_WORLD)
{
    int32_t _rank = 0;
    if(mpi_is_initialized())
        MPI_Comm_rank(comm, &_rank);
    return std::max(_rank, (int32_t) 0);
}

//----------------------------------------------------------------------------//

inline int32_t mpi_size(MPI_Comm comm = MPI_COMM_WORLD)
{
    int32_t _size = 1;
    if(mpi_is_initialized())
        MPI_Comm_size(comm, &_size);
    return std::max(_size, (int32_t) 1);
}

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

struct timer_tuple : public std::tuple<uint64_t, uint64_t, std::string,
                                       NAME_TIM::util::timer&>
{
    typedef std::string                                 string_t;
    typedef NAME_TIM::util::timer                       tim_timer_t;
    typedef uint64_t                                    first_type;
    typedef uint64_t                                    second_type;
    typedef string_t                                    third_type;
    typedef tim_timer_t                                 fourth_type;
    typedef std::tuple<uint64_t, uint64_t, string_t,
                       tim_timer_t&>                    base_type;

    timer_tuple(const base_type& _data) : base_type(_data) { }
    timer_tuple(first_type _b, second_type _s, third_type _t, fourth_type& _f)
    : base_type(_b, _s, _t, _f) { }

    timer_tuple& operator=(const base_type& rhs)
    {
        if(this == &rhs)
            return *this;
        base_type::operator =(rhs);
        return *this;
    }

    first_type& key() { return std::get<0>(*this); }
    const first_type& key() const { return std::get<0>(*this); }

    second_type& level() { return std::get<1>(*this); }
    const second_type& level() const { return std::get<1>(*this); }

    third_type tag() { return std::get<2>(*this); }
    const third_type tag() const { return std::get<2>(*this); }

    fourth_type& timer() { return std::get<3>(*this); }
    const fourth_type& timer() const { return std::get<3>(*this); }

    // serialization function
    template <typename Archive> void
    serialize(Archive& ar, const unsigned int /*version*/)
    {
        ar(cereal::make_nvp("timer.key", std::get<0>(*this)),
           cereal::make_nvp("timer.level", std::get<1>(*this)),
           cereal::make_nvp("timer.tag", std::get<2>(*this)),
           cereal::make_nvp("timer.ref", std::get<3>(*this)));
    }

};

//----------------------------------------------------------------------------//

class timing_manager
{
public:
    template <typename _Key, typename _Mapped>
    using uomap = std::unordered_map<_Key, _Mapped>;

    typedef NAME_TIM::util::timer           tim_timer_t;
    typedef tim_timer_t::string_t           string_t;
    typedef timer_tuple                     timer_tuple_t;
    typedef std::deque<timer_tuple_t>       timer_list_t;
    typedef timer_list_t::iterator          iterator;
    typedef timer_list_t::const_iterator    const_iterator;
    typedef timer_list_t::size_type         size_type;
    typedef uomap<uint64_t, tim_timer_t>    timer_map_t;
    typedef tim_timer_t::ostream_t          ostream_t;
    typedef tim_timer_t::ofstream_t         ofstream_t;
    typedef NAME_TIM::timer_field           timer_field;
    typedef std::tuple<MPI_Comm, int32_t>   comm_group_t;
    typedef std::mutex                      mutex_t;
    typedef uomap<uint64_t, mutex_t>        mutex_map_t;
    typedef std::lock_guard<mutex_t>        auto_lock_t;

public:
	// Constructor and Destructors
    timing_manager();
    virtual ~timing_manager();

public:
    // Public static functions
    static timing_manager* instance();
    static bool is_enabled() { return fgEnabled; }
    static void enable(bool val = true);
    static void write_json(string_t _fname);
    static int32_t& max_depth() { return fgMaxDepth; }

protected:
    static void write_json_mpi(string_t _fname);
    static void write_json_no_mpi(string_t _fname);

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
    void print() { this->report(); }
    void set_max_depth(int32_t d) { fgMaxDepth = d; }
    int32_t get_max_depth() { return fgMaxDepth; }
    void write_serialization(string_t _fname) const { write_json(_fname); }

protected:
    inline uint64_t string_hash(const string_t&) const;
    string_t get_prefix() const;
    static comm_group_t get_communicator_group();

private:
    // Private functions
    ofstream_t* get_ofstream(ostream_t* m_os) const;
    void report(ostream_t*, bool no_min = false) const;

private:
	// Private variables
    static timing_manager*  fgInstance;
    // for temporary enabling/disabling
    static bool             fgEnabled;
    // max depth of timers
    static int32_t          fgMaxDepth;
    // hashed string map for fast lookup
    timer_map_t             m_timer_map;
    // ordered list for output (outputs in order of timer instantiation)
    timer_list_t            m_timer_list;
    // output stream for total timing report
    ostream_t*              m_report;
    // locks for timers
    mutex_map_t             m_mutex_map;
};

//----------------------------------------------------------------------------//
inline void
timing_manager::clear()
{
    m_timer_list.clear();
    m_timer_map.clear();
    details::base_timer::get_instance_count() = 0;
    details::base_timer::get_instance_hash() = 0;

    ofstream_t* m_fos = get_ofstream(m_report);
    for(int32_t i = 0; i < mpi_size(); ++i)
    {
        if(mpi_is_initialized())
            MPI_Barrier(MPI_COMM_WORLD);
        if(mpi_rank() != i)
            continue;

        if(m_fos->good() && m_fos->is_open())
        {
            if(mpi_rank()+1 >= mpi_size())
            {
                m_fos->flush();
                m_fos->close();
                delete m_fos;
            }
            else
            {
                m_fos->flush();
                m_fos->close();
                delete m_fos;
            }
        }
    }
    m_report = &std::cout;
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

} // namespace util

} // namespace NAME_TIM

#endif // timing_manager_hpp_
