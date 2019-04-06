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

/** \file manager.cpp
 * Static singleton handler of auto-timers
 *
 */

#include "timemory/manager.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/environment.hpp"
#include "timemory/macros.hpp"
#include "timemory/papi.hpp"
#include "timemory/serializer.hpp"
#include "timemory/signal_detection.hpp"
#include "timemory/singleton.hpp"
#include "timemory/utility.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <sstream>
#include <thread>

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#    define TIMEMORY_DEFAULT_ENABLED true
#endif

//======================================================================================//

namespace tim
{
//======================================================================================//

int32_t manager::f_max_depth = std::numeric_limits<uint16_t>::max();

//======================================================================================//

std::atomic<int> manager::f_manager_instance_count;

//======================================================================================//
// this forces master manager instance to get created at the beginning of application
//
manager::pointer manager::f_instance = manager::master_instance();

//======================================================================================//
// get either master or thread-local instance
//
manager::pointer
manager::instance()
{
    return details::manager_singleton().instance();
}

//======================================================================================//
// get master instance
//
manager::pointer
manager::master_instance()
{
    return details::manager_singleton().master_instance();
}

//======================================================================================//
// static function
manager::pointer
manager::noninit_instance()
{
    return details::manager_singleton().instance_ptr();
}

//======================================================================================//
// static function
manager::pointer
manager::noninit_master_instance()
{
    return details::manager_singleton().master_instance_ptr();
}

//======================================================================================//

bool manager::f_enabled = TIMEMORY_DEFAULT_ENABLED;

//======================================================================================//

manager::mutex_t manager::f_mutex;

//======================================================================================//

manager::get_num_threads_func_t manager::f_get_num_threads = std::bind(&get_max_threads);

//======================================================================================//

manager::manager()
: m_merge(false)
, m_instance_count(f_manager_instance_count++)
, m_laps(0)
, m_report(&std::cout)
//, m_tuple_data(
//      std::make_tuple(timer_data_t(m_instance_count), memory_data_t(m_instance_count)))
{
    printf("############## %s:'%s'@%i ##############\n", __FUNCTION__, __FILE__,
           __LINE__);
    tim::papi::init();

    if(!singleton_t::master_instance_ptr())
    {
        m_merge = true;
        tim::env::parse();
    }
    else
    {
        manager* master = singleton_t::master_instance_ptr();
        master->set_merge(true);
        master->add(this);
        // timer_data.set_head(*master->get_timer_data().current());
        // timer_data.set_head(master->get_timer_data().current());
        // memory_data.set_head(&master->get_memory_data().current());
    }

    if(!singleton_t::master_instance_ptr())
        insert_global_timer();
    else if(singleton_t::master_instance_ptr() && singleton_t::instance_ptr())
    {
        std::ostringstream errss;
        errss << "manager singleton has already been created";
        throw std::runtime_error(errss.str().c_str());
    }
}

//======================================================================================//

manager::~manager()
{
#if defined(DEBUG)
    if(tim::env::verbose > 2)
        std::cout << "tim::manager::" << __FUNCTION__
                  << " deleting thread-local instance of manager..."
                  << "\nglobal instance: \t" << singleton_t::master_instance_ptr()
                  << "\nlocal instance:  \t" << singleton_t::instance_ptr() << std::endl;
#endif

    tim::papi::shutdown();
    details::manager_singleton().destroy();
    printf("############## %s:'%s'@%i ##############\n", __FUNCTION__, __FILE__,
           __LINE__);
}

//======================================================================================//

void
manager::update_total_timer_format()
{
    if((this == singleton_t::master_instance_ptr() || m_instance_count == 0))
    {
        // timer_data.total()->format()->prefix(this->get_prefix() +
        //                                     string_t("[exe] total execution time"));
        // timer_data.total()->format()->format(tim::format::timer::default_format());
        // timer_data.total()->format()->unit(tim::format::timer::default_unit());
        // timer_data.total()->format()->precision(tim::format::timer::default_precision());
        // tim::format::timer::propose_default_width(
        //    timer_data.total()->format()->prefix().length());
    }
}

//======================================================================================//

void
manager::insert_global_timer()
{
    if((this == singleton_t::master_instance_ptr() || m_instance_count == 0)
       // && timer_data.map().size() == 0
    )
    {
        update_total_timer_format();
        // timer_data.map()[0]           = timer_data.total();
        // timer_data_t::tuple_t _global = { 0, 0, 0, "exe_global_time",
        //                                  timer_data.total() };
        // timer_data.current()          = timer_data.graph().set_head(_global);
        // if(!timer_data.total()->is_running())
        //    timer_data.total()->start();
    }
}

//======================================================================================//

void
manager::clear()
{
#if defined(DEBUG)
    if(tim::env::verbose > 1)
        std::cout << "tim::manager::" << __FUNCTION__ << " Clearing " << instance()
                  << "..." << std::endl;
#endif

    // if(this == singleton_t::master_instance_ptr())
    // tim::format::timer::default_width(8);

    // m_laps += compute_total_laps();
    // timer_data.graph().clear();
    // timer_data.current() = nullptr;
    // timer_data.map().clear();
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
                if(mpi_rank() + 1 >= mpi_size())
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

//======================================================================================//

tim::string_t
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
            width = std::max(width, (uint16_t)(log10(mpi_size()) + 1));
        std::stringstream ss;
        ss.fill('0');
        ss << "|" << std::setw(width) << mpi_rank() << "> ";
        _prefix = new string_t(ss.str());
    }
    return *_prefix;
}

//======================================================================================//

void
manager::report(bool ign_cutoff, bool endline) const
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
            if(i != mpi_rank())
                continue;
        }
        report(m_report, ign_cutoff, endline);
    }
}

//======================================================================================//

void
manager::set_output_stream(ostream_t& _os)
{
    m_report = &_os;
}

//======================================================================================//

void
manager::set_output_stream(const path_t& fname)
{
    if(fname.find(fname.os()) != tim::string::npos)
        tim::makedir(fname.substr(0, fname.find_last_of(fname.os())));

    auto ostreamop = [&](ostream_t*& m_os, const string_t& _fname) {
        if(m_os != &std::cout)
            delete(ofstream_t*) m_os;

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

//======================================================================================//

void
manager::add(pointer ptr)
{
    auto_lock_t lock(m_mutex);
#if defined(DEBUG)
    if(tim::env::verbose > 2)
        std::cout << "tim::manager::" << __FUNCTION__ << " Adding " << ptr << " to "
                  << this << "..." << std::endl;
#endif
    m_merge = true;
    m_daughters.insert(ptr);
}

//======================================================================================//

void
manager::remove(pointer ptr)
{
    if(ptr == this)
        return;

    auto_lock_t lock(m_mutex);

#if defined(DEBUG)
    if(tim::env::verbose > 2)
        std::cout << "tim::manager::" << __FUNCTION__ << " - Removing " << ptr << " from "
                  << this << "..." << std::endl;
#endif

    merge(ptr);
    if(m_daughters.find(ptr) != m_daughters.end())
        m_daughters.erase(m_daughters.find(ptr));
}

//======================================================================================//

void
manager::merge(pointer itr)
{
    if(itr == this)
        return;

#if defined(DEBUG)
    if(tim::env::verbose > 2)
    {
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        std::cout << "\tinstance " << m_instance_count << " merging "
                  << itr->instance_count() << " with size: " << itr->size() << "..."
                  << std::endl;
    }

    if(tim::env::verbose > 1)
        std::cout << "tim::manager::" << __FUNCTION__ << " Merging " << itr << "..."
                  << std::endl;
#endif

    /*
    auto _this_beg = this->get_timer_data().graph().begin();
    auto _this_end = this->get_timer_data().graph().end();

    for(auto _this_itr = _this_beg; _this_itr != _this_end; ++_this_itr)
    {
        if(_this_itr == itr->get_timer_data().head())
        {
            auto _iter_beg = itr->get_timer_data().graph().begin();
            auto _iter_end = itr->get_timer_data().graph().end();
            this->get_timer_data().graph().merge(_this_itr, _this_end, _iter_beg,
                                                 _iter_end, false, true);
            break;
        }
    }

    typedef decltype(_this_beg) predicate_type;
    auto _reduce = [](predicate_type lhs, predicate_type rhs) { *lhs += *rhs; };
    _this_beg    = this->get_timer_data().graph().begin();
    _this_end    = this->get_timer_data().graph().end();
    this->get_timer_data().graph().reduce(_this_beg, _this_end, _this_beg, _this_end,
                                          _reduce);
    compute_self();
    */
}

//======================================================================================//

void
manager::merge()
{
    compute_self();
    if(!m_merge.load())
        return;
    if(m_daughters.size() == 0)
        return;
    m_merge.store(false);

    auto_lock_t lock(m_mutex);
    for(auto& itr : m_daughters)
        merge(itr);
    for(auto& itr : m_daughters)
        if(itr != this)
            itr->clear();
}

//======================================================================================//

void
manager::sync_hierarchy()
{
}

//======================================================================================//

void
manager::compute_self()
{
    if(this != singleton_t::master_instance_ptr())
        return;

    // m_timer_list_self.clear();

    // typedef std::unique_ptr<tim_timer_t> _timer_ptr_t;
    // using std::get;

    /*for(const auto& itr : m_timer_list_norm)
    {
        timer_tuple_t _ref = itr;
        timer_tuple_t _dup =
            std::make_tuple(_ref.key(), _ref.level(), _ref.offset(), _ref.tag(),
                            _timer_ptr_t(new tim_timer_t(_ref.timer())));
        _dup.timer().grab_metadata(_ref.timer());
        m_timer_list_self.push_back(_dup);
    }*/

    /*auto itr1 = m_timer_list_self.begin();
    ++itr1;  // skip global timer
    for(; itr1 != m_timer_list_self.end(); ++itr1)
    {
        uintmax_t _depth1 = itr1->level();
        for(auto itr2 = itr1; itr2 != m_timer_list_self.end(); ++itr2)
        {
            uintmax_t _depth2 = itr2->level();
            if(_depth2 == _depth1 + 1)
                itr1->timer() -= itr2->timer();
            if(itr2 != itr1 && _depth1 == _depth2)
                break;  // same level move onto next
        }
    }*/
}

//======================================================================================//
/*
manager::tim_timer_t
manager::compute_missing(tim_timer_t* timer_ref)
{
    // typedef std::set<uintmax_t> key_set_t;

    tim_timer_t* _ref    = (timer_ref) ? timer_ref : &timer_data.missing();
    bool         restart = _ref->is_running();
    if(restart)
        _ref->stop();

    this->merge();

    if(_ref->is_running())
        _ref->stop();

    tim_timer_t _missing_timer(_ref, _f_prefix, true);

    if(restart)
        _ref->start();

    auto max_depth = timer_data.graph().depth(timer_data.graph().end());
    if(max_depth < 1)
        return tim_timer_t(_f_prefix);

    decltype(max_depth)      current_depth = 0;
    std::vector<tim_timer_t> depth_timers(max_depth);
    for(auto itr = timer_data.begin(); itr != timer_data.end(); ++itr)
    {
        auto cdepth = timer_data.graph().depth(itr);
        depth_timers[cdepth] += itr->data();
    }

    for(uintmax_t i = 2; i < max_depth; ++i)
        _missing_timer += (depth_timers[i - 1] - depth_timers[i]);

    return _missing_timer;

    return *timer_ref;
}
//======================================================================================//

void
manager::write_missing(const path_t& _fname, tim_timer_t* timer_ref,
                       tim_timer_t* _missing_p, bool rank_zero_only)
{
    if(rank_zero_only && tim::mpi_rank() != 0)
        return;

    std::ofstream _os(_fname.c_str());
    if(_os)
        write_missing(_os, timer_ref, _missing_p);
    else
    {
        std::cerr << "Warning! Unable to open \"" << _fname << "\" [" << __FUNCTION__
                  << "@'" << __FILE__ << "'..." << std::endl;
        write_missing(std::cout);
    }
}

//======================================================================================//

void
manager::write_missing(ostream_t& _os, tim_timer_t* timer_ref, tim_timer_t* _missing_p,
                       bool rank_zero_only)
{
    if(rank_zero_only && tim::mpi_rank() != 0)
        return;

    tim_timer_t _missing;
    if(!_missing_p)
        _missing = compute_missing(timer_ref);
    else
    {
        _missing = *_missing_p;
        if(timer_ref)
            _missing -= *timer_ref;
    }

    std::stringstream   _ss;
    string_t::size_type _w  = 10;
    string_t            _p1 = "TiMemory auto-timer laps since last reset ";
    string_t            _p2 = "Total TiMemory auto-timer laps ";
    // find max of strings
    // use ternary because of std::max issues on Windows
    string_t::size_type _smax =
        4 + ((_p1.length() > _p2.length()) ? _p1.length() : _p2.length());
    // find max width
    // use ternary because of std::max issues on Windows
    _w = (_w > _smax) ? _w : _smax;

    std::stringstream _sp1, _sp2;
    _sp1 << std::setw(_w + 1) << std::left << _p1 << " : ";
    _sp2 << std::setw(_w + 1) << std::left << _p2 << " : ";
    _ss << _sp1.str() << laps() << std::endl;
    _ss << _sp2.str() << total_laps() << std::endl;

    //_missing.format()->width(_w);
    //_missing.format()->align_width(false);

    _ss << _missing.as_string() << std::endl;
    _os << "\n" << _ss.str();
}
*/
//======================================================================================//
//
//  Static functions for writing JSON output
//
//======================================================================================//

// static function
void
manager::write_report(path_t _fname, bool ign_cutoff)
{
    if(_fname.find(_fname.os()) != tim::string::npos)
        tim::makedir(_fname.substr(0, _fname.find_last_of(_fname.os())));

    ofstream_t ofs(_fname.c_str());
    if(ofs)
        manager::master_instance()->report(ofs, ign_cutoff);
    else
        manager::master_instance()->report(ign_cutoff);
    ofs.close();
}

//======================================================================================//
// static function
void
manager::write_json(path_t _fname)
{
    if(_fname.find(_fname.os()) != tim::string::npos)
        tim::makedir(_fname.substr(0, _fname.find_last_of(_fname.os())));

    (mpi_is_initialized()) ? write_json_mpi(_fname) : write_json_no_mpi(_fname);
}

//======================================================================================//
// static function
std::pair<int32_t, bool>
manager::write_json(ostream_t& ofss)
{
    if(mpi_is_initialized())
        return write_json_mpi(ofss);
    else
    {
        write_json_no_mpi(ofss);
        return std::pair<int32_t, bool>(0, true);
    }
}

//======================================================================================//
// static function
void
manager::write_json_no_mpi(ostream_t& fss)
{
    fss << "{\n\"ranks\": [" << std::endl;

    // ensure json write final block during destruction before the file
    // is closed
    {
        auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
        // precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 4);
        cereal::JSONOutputArchive          oa(fss, opts);

        // oa(cereal::make_nvp("manager", *manager::instance()));
    }

    fss << "]"
        << "\n}" << std::endl;
}

//======================================================================================//
// static function
void
manager::write_json_no_mpi(path_t _fname)
{
    int32_t _verbose = tim::get_env<int32_t>("TIMEMORY_VERBOSE", 0);

    if(mpi_rank() == 0 && _verbose > 0)
    {
        // notify so if it takes too long, user knows why
        std::stringstream _info;
        _info << "Writing serialization file: " << _fname << std::endl;
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
        std::cerr << "Warning! Unable to write JSON output to \"" << _fname << "\""
                  << std::endl;
    ofs.close();
}

//======================================================================================//
// static function
std::pair<int32_t, bool>
manager::write_json_mpi(ostream_t& ofss)
{
    const int32_t mpi_root       = 0;
    comm_group_t  mpi_comm_group = get_communicator_group();
    MPI_Comm&     local_mpi_comm = std::get<0>(mpi_comm_group);
    int32_t       local_mpi_file = std::get<1>(mpi_comm_group);

    // output stream
    std::stringstream fss;

    // ensure json write final block during destruction before the file
    // is closed
    {
        auto spacing = cereal::JSONOutputArchive::Options::IndentChar::tab;
        // precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 1);
        cereal::JSONOutputArchive          oa(fss, opts);

        // oa(cereal::make_nvp("manager", *manager::instance()));
    }

    // if another entry follows
    if(mpi_rank(local_mpi_comm) + 1 < mpi_size(local_mpi_comm))
        fss << ",";

    // the JSON output as a string
    string_t fss_str = fss.str();
    // limit the iteration loop. Occasionally it seems that this will create
    // an infinite loop even though it shouldn't...
    const uintmax_t itr_limit = fss_str.length();
    // compact the JSON
    for(auto citr : { "\n", "\t", "  " })
    {
        string_t            itr(citr);
        string_t::size_type fpos = 0;
        uintmax_t           nitr = 0;
        do
        {
            fpos = fss_str.find(itr, fpos);
            if(fpos != string_t::npos)
                fss_str.replace(fpos, itr.length(), " ");
            ++nitr;
        } while(nitr < itr_limit && fpos != string_t::npos);
    }

    // now we need to gather the lengths of each serialization string
    int  fss_len    = fss_str.length();
    int* recvcounts = nullptr;

    // Only root has the received data
    if(mpi_rank(local_mpi_comm) == mpi_root)
        recvcounts = (int*) malloc(mpi_size(local_mpi_comm) * sizeof(int));

    MPI_Gather(&fss_len, 1, MPI_INT, recvcounts, 1, MPI_INT, mpi_root, local_mpi_comm);

    // Figure out the total length of string, and displacements for each rank
    int*  fss_tot     = nullptr;
    char* totalstring = nullptr;

    if(mpi_rank(local_mpi_comm) == mpi_root)
    {
        int fss_tot_len = 0;
        fss_tot         = (int*) malloc(mpi_size(local_mpi_comm) * sizeof(int));

        fss_tot[0] = 0;
        fss_tot_len += recvcounts[0] + 1;

        for(int32_t i = 1; i < mpi_size(local_mpi_comm); ++i)
        {
            // plus one for space or \0 after words
            fss_tot_len += recvcounts[i] + 1;
            fss_tot[i] = fss_tot[i - 1] + recvcounts[i - 1] + 1;
        }

        // allocate string, pre-fill with spaces and null terminator
        totalstring = (char*) malloc(fss_tot_len * sizeof(char));
        for(int32_t i = 0; i < fss_tot_len - 1; ++i)
            totalstring[i] = ' ';
        totalstring[fss_tot_len - 1] = '\0';
    }

    // Now we have the receive buffer, counts, and displacements, and
    // can gather the strings

    char* cfss = (char*) fss_str.c_str();
    MPI_Gatherv(cfss, fss_len, MPI_CHAR, totalstring, recvcounts, fss_tot, MPI_CHAR,
                mpi_root, local_mpi_comm);

    if(mpi_rank(local_mpi_comm) == mpi_root)
    {
        ofss << "{\n\"ranks\": [" << std::endl;
        ofss << totalstring << std::endl;
        ofss << "]"
             << "\n}" << std::endl;
        free(totalstring);
        free(fss_tot);
        free(recvcounts);
    }

    bool write_rank = mpi_rank(local_mpi_comm) == mpi_root;

    MPI_Comm_free(&local_mpi_comm);

    return std::pair<int32_t, bool>(local_mpi_file, write_rank);
}

//======================================================================================//
// static function
void
manager::write_json_mpi(path_t _fname)
{
    {
        // notify so if it takes too long, user knows why
        std::stringstream _info;
        _info << "[" << mpi_rank() << "] Writing serialization file: " << _fname
              << std::endl;
        tim::auto_lock_t lock(tim::type_mutex<std::iostream>());
        std::cout << _info.str();
    }

    std::stringstream ofss;
    auto              ret            = write_json_mpi(ofss);
    int32_t           local_mpi_file = ret.first;
    bool              write_rank     = ret.second;

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

//======================================================================================//
// static function
manager::comm_group_t
manager::get_communicator_group()
{
    int32_t max_concurrency = std::thread::hardware_concurrency();
    // We want on-node communication only
    int32_t nthreads = tim::env::num_threads;
    if(nthreads == 0)
        nthreads = 1;
    int32_t max_processes    = max_concurrency / nthreads;
    int32_t mpi_node_default = mpi_size() / max_processes;
    if(mpi_node_default < 1)
        mpi_node_default = 1;
    int32_t mpi_node_count =
        tim::get_env<int32_t>("TIMEMORY_NODE_COUNT", mpi_node_default);
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

//======================================================================================//

}  // namespace tim

//======================================================================================//
