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

/** \file manager.cpp
 * Static singleton handler of auto-timers
 *
 */

#include "timemory/manager.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/serializer.hpp"
#include "timemory/timer.hpp"
#include "timemory/environment.hpp"
#include "timemory/timemory.hpp"

#include <sstream>
#include <algorithm>
#include <thread>
#include <cstdint>

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#   define TIMEMORY_DEFAULT_ENABLED true
#endif

//============================================================================//
// Easier to type than colons
typedef std::thread::id thread_id_t;

//============================================================================//
// Helper function
extern thread_id_t get_thread_id();

namespace tim
{

//============================================================================//

manager::pointer_type& local_instance()
{
    //std::cout << __PRETTY_FUNCTION__ << std::endl;
    tim_static_thread_local manager::pointer_type& _instance = _tim_manager_ptr();
    return _instance;
}

//============================================================================//

manager::pointer_type& global_instance()
{
    //std::cout << __PRETTY_FUNCTION__ << std::endl;
    static manager::pointer_type& _instance = _tim_manager_ptr();
    return _instance;
}

//============================================================================//

int32_t manager::f_max_depth = std::numeric_limits<uint16_t>::max();

//============================================================================//

std::atomic<int> manager::f_manager_instance_count;

//============================================================================//
// static function
manager::pointer_type manager::instance()
{
    if(!local_instance())
    {
        _tim_manager_initialization();
        global_instance()->add(local_instance());
    }

    if(local_instance() != global_instance())
        global_instance()->set_merge(true);

    return local_instance();
}

//============================================================================//
// static function
manager::pointer_type manager::master_instance()
{
    if(!local_instance())
    {
        _tim_manager_initialization();
        global_instance()->add(local_instance());
    }

    if(local_instance() != global_instance())
        global_instance()->set_merge(true);

    return global_instance();
}

//============================================================================//

bool manager::f_enabled = TIMEMORY_DEFAULT_ENABLED;

//============================================================================//

manager::mutex_t manager::f_mutex;

//============================================================================//

manager::get_num_threads_func_t
        manager::f_get_num_threads = std::bind(&get_max_threads);

//============================================================================//

void manager::set_get_num_threads_func(get_num_threads_func_t f)
{
    f_get_num_threads = std::bind(f);
}

//============================================================================//

manager::manager()
: m_merge(false),
  m_instance_count(f_manager_instance_count++),
  m_laps(0),
  m_hash(0),
  m_count(0),
  m_p_hash((global_instance()) ? global_instance()->hash().load() : 0),
  m_p_count((global_instance()) ? global_instance()->count().load() : 0),
  m_report(&std::cout),
  m_overhead_timer(nullptr),
  m_total_timer(nullptr)
{
    if(!global_instance())
        tim::env::parse();

    m_overhead_timer = new tim_timer_t();
    m_total_timer = timer_ptr_t(
                        new tim_timer_t(
                            tim::format::timer(
                                string_t("> [exe] total"),
                                tim::format::timer::default_format(),
                                tim::format::timer::default_unit(),
                                tim::format::timer::default_rss_format(),
                                true)));
#if defined(DEBUG)
    if(tim::env::verbose > 2)
    {
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        std::cout << "tim::manager creation " << m_instance_count
                  << "..." << std::endl;
    }
#endif

    if(tim::env::disable_timer_memory > 0)
    {
        tim::timer::default_record_memory(false);
        m_overhead_timer->record_memory(false);
        m_total_timer->record_memory(false);
    }

    std::stringstream ss;
    ss << "TiMemory total unrecorded time (manager "
       << (m_instance_count) << ")";
    m_overhead_timer->format()->prefix(ss.str());
    m_overhead_timer->start();

    if(!global_instance())
    {
        global_instance() = this;
        insert_global_timer();
    }
    else if(global_instance() && local_instance())
    {
        std::ostringstream ss;
        ss << "manager singleton has already been created";
        throw std::runtime_error( ss.str().c_str() );
    }
}

//============================================================================//

manager::~manager()
{
    if(this == global_instance() && tim::env::output_total)
        std::cout << "\n" << m_total_timer.get()->as_string() << std::endl;

#if defined(DEBUG)
    if(tim::env::verbose > 2)
    {
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        std::cout << "tim::manager deletion " << m_instance_count
                  << "..." << std::endl;
    }
#endif

    m_overhead_timer->stop();
    --f_manager_instance_count;
    auto close_ostream = [&] (ostream_t*& m_os)
    {
        ofstream_t* m_fos = get_ofstream(m_os);
        if(!m_fos)
            return;
        if(!m_fos->good() || !m_fos->is_open())
            return;
        m_fos->close();
    };

#if defined(DEBUG)
    if(tim::env::verbose > 2)
        std::cout << "tim::manager::" << __FUNCTION__
                  << " deleting thread-local instance of manager..."
                  << "\nglobal instance: \t" << global_instance()
                  << "\nlocal instance:  \t" << local_instance()
                  << std::endl;
#endif

    close_ostream(m_report);

    if(global_instance() == local_instance())
        global_instance() = nullptr;

    for(auto& itr : m_daughters)
        if(itr != this)
            delete itr;

    m_daughters.clear();
    m_timer_list.clear();
    m_timer_map.clear();
    delete m_overhead_timer;

    if(this == _tim_manager_ptr() && get_thread_id() == _tim_manager_tid())
        _tim_manager_ptr() = nullptr;
}

//============================================================================//

void manager::update_total_timer_format()
{
    if(this == global_instance())
    {
        m_total_timer->format()->prefix(this->get_prefix() +
                                        string_t("[exe] total execution time"));
        m_total_timer->format()->format(tim::format::timer::default_format());
        m_total_timer->format()->unit(tim::format::timer::default_unit());
        m_total_timer->format()->rss_format(tim::format::timer::default_rss_format());
        tim::format::timer::propose_default_width(
                    m_total_timer->format()->prefix().length());
    }
}

//============================================================================//

void manager::insert_global_timer()
{
    if(this == global_instance() &&
       m_timer_map.size() == 0 &&
       m_timer_list.size() == 0)
    {
        update_total_timer_format();
        m_timer_map[0] = m_total_timer;
        m_timer_list.push_back(
                    timer_tuple_t(0, m_count, "exe_global_time",
                                  m_total_timer));
        m_total_timer->start();
        if(m_count == 0)
            m_count += 1;
    }
}

//============================================================================//

void manager::clear()
{
#if defined(DEBUG)
    if(tim::env::verbose > 1)
        std::cout << "tim::manager::" << __FUNCTION__ << " Clearing "
                  << local_instance() << "..." << std::endl;
#endif

    if(this == global_instance())
        tim::format::timer::default_width(10);

    m_laps += compute_total_laps();
    m_timer_list.clear();
    m_timer_map.clear();
    for(auto& itr : m_daughters)
        if(itr != this && itr)
            itr->clear();

    ofstream_t* m_fos = get_ofstream(m_report);

    if(m_fos)
    {
        for(int32_t i = 0; i < mpi_size(); ++i)
        {
            mpi_barrier(MPI_COMM_WORLD);
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
    }

    m_report = &std::cout;

    insert_global_timer();
}

//============================================================================//

manager::string_t
manager::get_prefix() const
{
    if(!mpi_is_initialized())
        return "> ";

    static string_t* _prefix = nullptr;
    if(!_prefix)
    {
        // prefix spacing
        static uint16_t width = 1;
        if(mpi_size() > 9)
            width = std::max(width, (uint16_t) ( log10(mpi_size()) + 1 ));
        std::stringstream ss;
        ss.fill('0');
        ss << "|" << std::setw(width) << mpi_rank() << "> ";
        _prefix = new string_t(ss.str());
    }
    return *_prefix;
}

//============================================================================//

timer&
manager::timer(const string_t& key,
               const string_t& tag,
               int32_t ncount,
               int32_t nhash)
{
#if defined(DEBUG)
    if(key.find(" ") != string_t::npos)
    {
        std::stringstream ss;
        ss << "tim::manager::" << __FUNCTION__
           << " Warning! Space found in tag: \"" << key << "\"";
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        std::cerr << ss.str() << std::endl;
    }
#endif

    uint64_t ref = (string_hash(key) + string_hash(tag)) * (ncount+2) * (nhash+2);

    // thread-safe
    //auto_lock_t lock(f_mutex);

    // if already exists, return it
    if(m_timer_map.find(ref) != m_timer_map.end())
    {
#if defined(HASH_DEBUG)
        for(const auto& itr : m_timer_list)
        {
            tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
            if(&(std::get<3>(itr)) == &(m_timer_map[ref]))
                std::cout << "tim::manager::" << __FUNCTION__ << " Found : "
                          << itr << std::endl;
        }
#endif
        return *(m_timer_map[ref].get());
    }

    // synchronize format with level 1 and make sure MPI prefix is up-to-date
    if(m_timer_list.size() < 2)
        update_total_timer_format();

    std::stringstream ss;
    // designated as [cxx], [pyc], etc.
    ss << get_prefix() << "[" << tag << "] ";

    // indent
    for(int64_t i = 0; i < ncount; ++i)
    {
        if(i+1 == ncount)
            ss << "|_";
        else
            ss << "  ";
    }

    ss << std::left << key;
    tim::format::timer::propose_default_width(ss.str().length());

    m_timer_map[ref] =
            timer_ptr_t(
                new tim_timer_t(
                    tim::format::timer(
                        ss.str(),
                        tim::format::timer::default_format(),
                        tim::format::timer::default_unit(),
                        tim::format::timer::default_rss_format(),
                        true)));

    std::stringstream tag_ss;
    tag_ss << tag << "_" << std::left << key;
    timer_tuple_t _tuple(ref, ncount, tag_ss.str(), m_timer_map[ref]);
    m_timer_list.push_back(_tuple);

#if defined(HASH_DEBUG)
    std::cout << "tim::manager::" << __FUNCTION__ << " Created : "
              << _tuple << std::endl;
#endif

    return *(m_timer_map[ref].get());
}

//============================================================================//

void manager::report(bool ign_cutoff, bool endline) const
{
    const_cast<this_type*>(this)->merge();

    int32_t _default = (mpi_is_initialized()) ? 1 : 0;
    int32_t _verbose = tim::get_env<int32_t>("TIMEMORY_VERBOSE", _default);

    if(mpi_rank() == 0 && _verbose > 0)
    {
        std::stringstream _info;
        if(mpi_is_initialized())
            _info << "[" << mpi_rank() << "] ";
        _info << "Reporting timing output..." << std::endl;
        std::cout << _info.str();
    }

    int nitr = std::max(mpi_size(), 1);
    for(int32_t i = 0; i < nitr; ++i)
    {
        // MPI blocking
        if(mpi_is_initialized())
        {
            mpi_barrier(MPI_COMM_WORLD);
            // only 1 at a time
            if(i != mpi_rank() )
                continue;
        }
        report(m_report, ign_cutoff, endline);
    }
}

//============================================================================//

void manager::report(ostream_t* os, bool ign_cutoff, bool endline) const
{
    const_cast<this_type*>(this)->merge();

    auto check_stream = [&] (ostream_t*& _os, const string_t& id)
    {
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
    auto _width = tim::format::timer::default_width();
    // reset output width
    tim::format::timer::default_width(10);

    // redo output width calc, removing no displayed funcs
    for(const auto& itr : *this)
        if(itr.timer().above_cutoff(ign_cutoff) || ign_cutoff)
            tim::format::timer::propose_default_width(itr.timer().format()->prefix().length());

    // don't make it longer
    if(_width > 10 && _width < tim::format::timer::default_width())
        tim::format::timer::default_width(_width);

    for(auto itr = this->cbegin(); itr != this->cend(); ++itr)
        itr->timer().report(*os, (itr+1 == this->cend()) ? endline : true,
                            ign_cutoff);

    os->flush();
}

//============================================================================//

void manager::set_output_stream(ostream_t& _os)
{
    m_report = &_os;
}

//============================================================================//

void manager::set_output_stream(const path_t& fname)
{
    if(fname.find(fname.os()) != std::string::npos)
        tim::makedir(fname.substr(0, fname.find_last_of(fname.os())));

    auto ostreamop = [&] (ostream_t*& m_os, const string_t& _fname)
    {
        if(m_os != &std::cout)
            delete (ofstream_t*) m_os;

        ofstream_t* _fos = new ofstream_t;
        for(int32_t i = 0; i < mpi_size(); ++i)
        {
            mpi_barrier(MPI_COMM_WORLD);
            if(mpi_rank() != i)
                continue;

            if(mpi_rank() == 0)
                _fos->open(_fname);
            else
                _fos->open(_fname, std::ios_base::out | std::ios_base::app);
        }

        if(_fos->is_open() && _fos->good())
            m_os = _fos;
        else
        {
#if defined(DEBUG)
    if(tim::env::verbose > 2)
            {
                tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
                std::cerr << "Warning! Unable to open file " << _fname << ". "
                          << "Redirecting to stdout..." << std::endl;
            }
#endif
            _fos->close();
            delete _fos;
            m_os = &std::cout;
        }
    };

    ostreamop(m_report, fname);

}

//============================================================================//

manager::ofstream_t*
manager::get_ofstream(ostream_t* m_os) const
{
    return (m_os != &std::cout && m_os != &std::cerr) 
        ? static_cast<ofstream_t*>(m_os)
        : nullptr;
}

//============================================================================//

void manager::add(pointer_type ptr)
{
    auto_lock_t lock(m_mutex);
#if defined(DEBUG)
    if(tim::env::verbose > 2)
        std::cout << "tim::manager::" << __FUNCTION__ << " Adding "
                  << ptr << " to " << this << "..." << std::endl;
#endif
    m_daughters.insert(ptr);
}

//============================================================================//

void manager::merge(bool div_clock)
{
    if(!m_merge.load())
        return;

    m_merge.store(false);

    bool restart = m_overhead_timer->is_running();
    if(restart)
        m_overhead_timer->stop();

#if defined(DEBUG)
    if(tim::env::verbose > 2)
    {
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        std::cout << "instance " << m_instance_count << " : "
                  << __FUNCTION__
                  << " (div_clock = " << std::boolalpha
                  << div_clock << ") ..." << std::endl;
    }
#endif

    auto_lock_t lock(m_mutex);

    uomap<uint64_t, uint64_t> clock_div_count;
    for(auto& itr : m_daughters)
    {
        if(itr == global_instance())
            continue;

#if defined(DEBUG)
    if(tim::env::verbose > 2)
        {
            tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
            std::cout << "\tinstance " << m_instance_count << " merging "
                      << itr->instance_count() << "..." << std::endl;
        }
#endif

        //itr->overhead_timer()->stop();

#if defined(DEBUG)
    if(tim::env::verbose > 1)
        std::cout << "tim::manager::" << __FUNCTION__ << " Merging " << itr
                  << "..." << std::endl;
#endif

        for(const auto& mitr : itr->map())
        {
            if(m_timer_map.find(mitr.first) == m_timer_map.end())
                m_timer_map[mitr.first] = mitr.second;
            else
                clock_div_count[mitr.first] += 1;
        }

        //*m_overhead_timer += *(itr->overhead_timer());
        //itr->overhead_timer()->reset();

        for(const auto& litr : itr->list())
        {
            bool found = false;
            for(auto& mlitr : m_timer_list)
                if(mlitr == litr)
                {
                    mlitr += litr;
                    found = true;
                    break;
                }
            if(!found)
                m_timer_list.push_back(litr);
        }

        //itr->overhead_timer()->start();
    }

    if(div_clock)
        for(auto& itr : clock_div_count)
            *(m_timer_map[itr.first].get()) /= itr.second;

    for(auto& itr : m_daughters)
        if(itr != this)
            itr->clear();

    if(restart)
        m_overhead_timer->start();
}

//============================================================================//

void manager::sync_hierarchy()
{
    for(auto& itr : m_daughters)
    {
        if(itr->hash() > 0)
            // add difference to current hash
            itr->hash() += (m_hash - itr->parent_hash());
        // set the parent count to this count
        itr->parent_hash() = m_hash.load();

        if(itr->count() > 0)
            // add difference to current hash
            itr->count() += (m_count - itr->parent_count());
        // set the parent count to this count
        itr->parent_count() = m_count.load();

        itr->sync_hierarchy();
    }
}

//============================================================================//

manager::timer_pair_t
manager::compute_overhead(tim_timer_t* timer_ref)
{
    typedef std::set<uint64_t> key_set_t;

    tim_timer_t* _ref = (timer_ref) ? timer_ref : m_overhead_timer;
    bool restart = _ref->is_running();
    if(restart)
        _ref->stop();

    this->merge();

    if(_ref->is_running())
        _ref->stop();

    string_t _f_prefix = _ref->format()->prefix();
    string_t _s_prefix = _f_prefix;
    _s_prefix.replace(_s_prefix.find("total unrecorded time"),
                      string_t("total unrecorded time").length(),
                      "approximate overhead ");

    timer_pair_t _overhead_timers(tim_timer_t(_ref, _f_prefix, true),
                                  tim_timer_t(_ref, _s_prefix, true));

    if(restart)
        _ref->start();

    key_set_t _depths;
    for(const auto& itr : *this)
        _depths.insert(itr.level());

    if(_depths.size() == 0)
        return timer_pair_t(tim_timer_t(_f_prefix),
                            tim_timer_t(_s_prefix));

    for(const auto& itr : *this)
        if(itr.level() == *(_depths.begin()))
            _overhead_timers.first -= itr.timer();

    // remove unaccounted time
    _overhead_timers.second.accum() -= _overhead_timers.first.accum();
    _overhead_timers.second.accum() -= _overhead_timers.first.accum();

    return _overhead_timers;
}

//============================================================================//

void manager::write_overhead(const path_t& _fname, tim_timer_t* timer_ref,
                             timer_pair_t* _overhead_p)
{
    std::ofstream _os(_fname.c_str());
    if(_os)
        write_overhead(_os, timer_ref, _overhead_p);
    else
    {
        std::cerr << "Warning! Unable to open \"" << _fname << "\" ["
                  << __FUNCTION__ << "@'" << __FILE__ << "'..." << std::endl;
        write_overhead(std::cout);
    }
}

//============================================================================//

void manager::write_overhead(ostream_t& _os, tim_timer_t* timer_ref,
                             timer_pair_t* _overhead_p)
{
    timer_pair_t _overhead;
    if(!_overhead_p)
        _overhead = compute_overhead(timer_ref);
    else
    {
        _overhead = *_overhead_p;
        if(timer_ref)
        {
            _overhead.first -= *timer_ref;
            _overhead.second -= *timer_ref;
        }
    }

    std::stringstream _ss;
    string_t::size_type _w = format::timer::default_width();
    string_t _p1 = "TiMemory auto-timer laps since last reset ";
    string_t _p2 = "Total TiMemory auto-timer laps ";
    _w = std::max(_w, std::max(_p1.length() + 4, _p2.length() + 4));

    std::stringstream _sp1, _sp2;
    _sp1 << std::setw(_w + 1) << std::left << _p1 << " : ";
    _sp2 << std::setw(_w + 1) << std::left << _p2 << " : ";
    _ss << _sp1.str() << laps() << std::endl;
    _ss << _sp2.str() << total_laps() << std::endl;

    _overhead.first.format()->width(_w);
    _overhead.second.format()->width(_w);
    _overhead.first.format()->align_width(false);
    _overhead.second.format()->align_width(false);

    _ss << _overhead.first.as_string() << std::endl;
    if(_overhead.second.wall_elapsed() > 1.0e-9)
        _ss << _overhead.second.as_string() << std::endl;
    _os << "\n" << _ss.str();
}

//============================================================================//
//
//  Static functions for writing JSON output
//
//============================================================================//

// static function
void manager::write_report(path_t _fname, bool ign_cutoff)
{
    if(_fname.find(_fname.os()) != std::string::npos)
        tim::makedir(_fname.substr(0, _fname.find_last_of(_fname.os())));

    ofstream_t ofs(_fname.c_str());
    if(ofs)
        manager::master_instance()->report(ofs, ign_cutoff);
    else
        manager::master_instance()->report(ign_cutoff);
    ofs.close();
}

//============================================================================//
// static function
void manager::write_json(path_t _fname)
{
    if(_fname.find(_fname.os()) != std::string::npos)
        tim::makedir(_fname.substr(0, _fname.find_last_of(_fname.os())));

    (mpi_is_initialized()) ? write_json_mpi(_fname) : write_json_no_mpi(_fname);
}

//============================================================================//
// static function
std::pair<int32_t, bool> manager::write_json(ostream_t& ofss)
{
    if(mpi_is_initialized())
        return write_json_mpi(ofss);
    else
    {
        write_json_no_mpi(ofss);
        return std::pair<int32_t, bool>(0, true);
    }
}

//============================================================================//
// static function
void manager::write_json_no_mpi(ostream_t& fss)
{
    fss << "{\n\"ranks\": [" << std::endl;

    // ensure json write final block during destruction before the file
    // is closed
    {
        auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
        // precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 4);
        cereal::JSONOutputArchive oa(fss, opts);

        oa(cereal::make_nvp("manager", *manager::instance()));
    }

    fss << "]" << "\n}" << std::endl;
}

//============================================================================//
// static function
void manager::write_json_no_mpi(path_t _fname)
{
    int32_t _verbose = tim::get_env<int32_t>("TIMEMORY_VERBOSE", 0);

    if(mpi_rank() == 0 && _verbose > 0)
    {
        // notify so if it takes too long, user knows why
        std::stringstream _info;
        _info << "Writing serialization file: "
              << _fname << std::endl;
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        std::cout << _info.str();
    }

    std::stringstream fss;
    write_json_no_mpi(fss);

    // write to file
    std::ofstream ofs(_fname.c_str());
    if(ofs)
        ofs << fss.str() << std::endl;
    else
        std::cerr << "Warning! Unable to write JSON output to \""
                  << _fname << "\"" << std::endl;
    ofs.close();
}


//============================================================================//
// static function
std::pair<int32_t, bool> manager::write_json_mpi(ostream_t& ofss)
{
    const int32_t mpi_root = 0;
    comm_group_t mpi_comm_group = get_communicator_group();
    MPI_Comm& local_mpi_comm = std::get<0>(mpi_comm_group);
    int32_t local_mpi_file = std::get<1>(mpi_comm_group);

    // output stream
    std::stringstream fss;

    // ensure json write final block during destruction before the file
    // is closed
    {
        auto spacing = cereal::JSONOutputArchive::Options::IndentChar::tab;
        // precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 1);
        cereal::JSONOutputArchive oa(fss, opts);

        oa(cereal::make_nvp("manager", *manager::instance()));
    }

    // if another entry follows
    if(mpi_rank(local_mpi_comm)+1 < mpi_size(local_mpi_comm))
        fss << ",";

    // the JSON output as a string
    string_t fss_str = fss.str();
    // limit the iteration loop. Occasionally it seems that this will create
    // an infinite loop even though it shouldn't...
    const uint64_t itr_limit = fss_str.length();
    // compact the JSON
    for(auto citr : { "\n", "\t", "  " })
    {
        string_t itr(citr);
        string_t::size_type fpos = 0;
        uint64_t nitr = 0;
        do
        {
            fpos = fss_str.find(itr, fpos);
            if(fpos != string_t::npos)
                fss_str.replace(fpos, itr.length(), " ");
            ++nitr;
        }
        while(nitr < itr_limit && fpos != string_t::npos);
    }

    // now we need to gather the lengths of each serialization string
    int fss_len = fss_str.length();
    int* recvcounts = nullptr;

    // Only root has the received data
    if (mpi_rank(local_mpi_comm) == mpi_root)
        recvcounts = (int*) malloc( mpi_size(local_mpi_comm) * sizeof(int)) ;

    MPI_Gather(&fss_len, 1, MPI_INT,
               recvcounts, 1, MPI_INT,
               mpi_root, local_mpi_comm);

    // Figure out the total length of string, and displacements for each rank
    int fss_tot_len = 0;
    int* fss_tot = nullptr;
    char* totalstring = nullptr;

    if (mpi_rank(local_mpi_comm) == mpi_root)
    {
        fss_tot = (int*) malloc( mpi_size(local_mpi_comm) * sizeof(int) );

        fss_tot[0] = 0;
        fss_tot_len += recvcounts[0]+1;

        for(int32_t i = 1; i < mpi_size(local_mpi_comm); ++i)
        {
            // plus one for space or \0 after words
            fss_tot_len += recvcounts[i]+1;
            fss_tot[i] = fss_tot[i-1] + recvcounts[i-1] + 1;
        }

        // allocate string, pre-fill with spaces and null terminator
        totalstring = (char*) malloc(fss_tot_len * sizeof(char));
        for(int32_t i = 0; i < fss_tot_len-1; ++i)
            totalstring[i] = ' ';
        totalstring[fss_tot_len-1] = '\0';
    }

    // Now we have the receive buffer, counts, and displacements, and
    // can gather the strings

    char* cfss = (char*) fss_str.c_str();
    MPI_Gatherv(cfss, fss_len, MPI_CHAR,
                totalstring, recvcounts, fss_tot, MPI_CHAR,
                mpi_root, local_mpi_comm);

    if (mpi_rank(local_mpi_comm) == mpi_root)
    {
        ofss << "{\n\"ranks\": [" << std::endl;
        ofss << totalstring << std::endl;
        ofss << "]" << "\n}" << std::endl;
        free(totalstring);
        free(fss_tot);
        free(recvcounts);
    }

    bool write_rank = mpi_rank(local_mpi_comm) == mpi_root;

    MPI_Comm_free(&local_mpi_comm);

    return std::pair<int32_t, bool>(local_mpi_file, write_rank);
}

//============================================================================//
// static function
void manager::write_json_mpi(path_t _fname)
{
    {
        // notify so if it takes too long, user knows why
        std::stringstream _info;
        _info << "[" << mpi_rank() << "] Writing serialization file: "
              << _fname << std::endl;
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        std::cout << _info.str();
    }

    std::stringstream ofss;
    auto ret = write_json_mpi(ofss);
    int32_t local_mpi_file = ret.first;
    bool write_rank = ret.second;

    if(write_rank)
    {
        std::stringstream _rss;
        _rss << "_" << local_mpi_file;
        _fname.insert(_fname.find_last_of("."), _rss.str());

        ofstream_t ofs;
        ofs.open(_fname);
        if(ofs)
            ofs << ofss.str();
        ofs.close();
    }
}

//============================================================================//
// static function
manager::comm_group_t
manager::get_communicator_group()
{
    int32_t max_concurrency = std::thread::hardware_concurrency();
    // We want on-node communication only
    int32_t nthreads = tim::env::num_threads;
    if(nthreads == 0)
        nthreads = 1;
    int32_t max_processes = max_concurrency / nthreads;
    int32_t mpi_node_default = mpi_size() / max_processes;
    if(mpi_node_default < 1)
        mpi_node_default = 1;
    int32_t mpi_node_count = tim::get_env<int32_t>("TIMEMORY_NODE_COUNT",
                                                     mpi_node_default);
    int32_t mpi_split_size = mpi_rank() / (mpi_size() / mpi_node_count);

    // Split the communicator based on the number of nodes and use the
    // original rank for ordering
    MPI_Comm local_mpi_comm;
    MPI_Comm_split(MPI_COMM_WORLD, mpi_split_size, mpi_rank(), &local_mpi_comm);

#if defined(DEBUG)
    if(tim::env::verbose > 1)
    {
        int32_t local_mpi_rank = mpi_rank(local_mpi_comm);
        int32_t local_mpi_size = mpi_size(local_mpi_comm);
        int32_t local_mpi_file = mpi_rank() / local_mpi_size;

        std::stringstream _info;
        _info << "\t" << mpi_rank() << " Rank      : " << mpi_rank() << std::endl;
        _info << "\t" << mpi_rank() << " Size      : " << mpi_size() << std::endl;
        _info << "\t" << mpi_rank() << " Node      : " << mpi_node_count << std::endl;
        _info << "\t" << mpi_rank() << " Local Size: " << local_mpi_size << std::endl;
        _info << "\t" << mpi_rank() << " Local Rank: " << local_mpi_rank << std::endl;
        _info << "\t" << mpi_rank() << " Local File: " << local_mpi_file << std::endl;
        std::cout << "tim::manager::" << __FUNCTION__ << "\n" << _info.str();
    }
#endif

    return comm_group_t(local_mpi_comm, mpi_rank() / mpi_size(local_mpi_comm));
}

//============================================================================//

} // namespace tim

//============================================================================//
