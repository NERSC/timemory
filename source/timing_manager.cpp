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

/** \file timing_manager.cpp
 * Static singleton handler of auto-timers
 *
 */

#include "timemory/timing_manager.hpp"
#include "timemory/auto_timer.hpp"

#include <sstream>
#include <algorithm>
#include <thread>

//============================================================================//

CEREAL_CLASS_VERSION(NAME_TIM::timer_tuple, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(NAME_TIM::timing_manager, TIMEMORY_TIMER_VERSION)

namespace NAME_TIM
{

//============================================================================//

timing_manager::pointer_type timing_manager::f_instance = nullptr;

//============================================================================//

static thread_local timing_manager::pointer_type l_instance = nullptr;

//============================================================================//

int32_t timing_manager::f_max_depth = std::numeric_limits<uint16_t>::max();

//============================================================================//
// static function
timing_manager::pointer_type timing_manager::instance()
{
    if(!l_instance)
    {
        l_instance = new timing_manager();
        f_instance->add(l_instance);
    }

    if(l_instance != f_instance)
        f_instance->set_merge(true);

    return l_instance;
}

//============================================================================//

bool timing_manager::f_enabled = true;

//============================================================================//

timing_manager::mutex_t timing_manager::f_mutex;

//============================================================================//
// static function
void timing_manager::enable(bool val)
{
    f_enabled = val;
}

//============================================================================//

timing_manager::timing_manager()
: m_merge(false),
  m_hash((f_instance) ? f_instance->hash() : 0),
  m_count((f_instance) ? f_instance->count() : 0),
  m_report(&std::cout)
{
    if(!f_instance)
        f_instance = this;
    else if(f_instance && l_instance)
    {
        std::ostringstream ss;
        ss << "timing_manager singleton has already been created";
        throw std::runtime_error( ss.str().c_str() );
    }
}

//============================================================================//

timing_manager::~timing_manager()
{
    auto close_ostream = [&] (ostream_t*& m_os)
    {
        ofstream_t* m_fos = get_ofstream(m_os);
        if(!m_fos->good() || !m_fos->is_open())
            return;
        m_fos->close();
    };

#if defined(DEBUG)
    if(NAME_TIM::get_env("TIMEMORY_VERBOSE", 0) > 1)
        std::cout << "deleting thread-local instance of timing_manager..."
                  << "\nglobal instance: \t" << f_instance
                  << "\nlocal instance:  \t" << l_instance
                  << std::endl;
#endif

    close_ostream(m_report);

    if(f_instance == l_instance)
        f_instance = nullptr;

    for(auto& itr : m_daughters)
        if(itr != this)
            delete itr;

    m_daughters.clear();
    m_timer_list.clear();
    m_timer_map.clear();
}

//============================================================================//

void timing_manager::clear()
{
#if defined(DEBUG)
    if(NAME_TIM::get_env("TIMEMORY_VERBOSE", 0) > 1)
        std::cout << "Clearing " << l_instance << "..." << std::endl;
#endif

    if(this == f_instance)
        tim_timer_t::set_output_width(10);

    m_timer_list.clear();
    m_timer_map.clear();
    for(auto& itr : m_daughters)
        if(itr != this)
            itr->clear();

    ofstream_t* m_fos = get_ofstream(m_report);
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
    m_report = &std::cout;
}

//============================================================================//

timing_manager::string_t
timing_manager::get_prefix() const
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
timing_manager::timer(const string_t& key,
                      const string_t& tag,
                      int32_t ncount,
                      int32_t nhash)
{
#if defined(DEBUG)
    if(key.find(" ") != string_t::npos)
    {
        std::stringstream ss;
        ss << "Error! Space found in tag: \"" << key << "\"";
        throw std::runtime_error(ss.str().c_str());
    }
#endif

    uint64_t ref = (string_hash(key) + string_hash(tag)) * (ncount+2) * (nhash+2);

    // thread-safe
    auto_lock_t lock(f_mutex);

    // if already exists, return it
    if(m_timer_map.find(ref) != m_timer_map.end())
    {
#if defined(HASH_DEBUG)
        for(const auto& itr : m_timer_list)
        {
            if(&(std::get<3>(itr)) == &(m_timer_map[ref]))
                std::cout << "Found : " << itr << std::endl;
        }
#endif
        return *(m_timer_map[ref].get());
    }

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
    timer::propose_output_width(ss.str().length());

    const uint16_t precision = tim_timer_t::default_precision;
    m_timer_map[ref] = timer_ptr_t(new tim_timer_t(ss.str(), string_t(""),
                                                   true, precision));

    std::stringstream tag_ss;
    tag_ss << tag << "_" << std::left << key;
    timer_tuple_t _tuple(ref, ncount, tag_ss.str(), m_timer_map[ref]);
    m_timer_list.push_back(_tuple);

#if defined(HASH_DEBUG)
    std::cout << "Created : " << _tuple << std::endl;
#endif

    return *(m_timer_map[ref].get());
}

//============================================================================//

void timing_manager::report(bool no_min) const
{
    const_cast<this_type*>(this)->merge();

    int32_t _default = (mpi_is_initialized()) ? 1 : 0;
    int32_t _verbose = NAME_TIM::get_env<int32_t>("TIMEMORY_VERBOSE", _default);

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
        report(m_report, no_min);
    }
}

//============================================================================//

void timing_manager::report(ostream_t* os, bool no_min) const
{
    const_cast<this_type*>(this)->merge();

    auto check_stream = [&] (ostream_t*& _os, const string_t& id)
    {
        if(_os == &std::cout || _os == &std::cerr)
            return;
        ofstream_t* fos = get_ofstream(_os);
        if(!(fos->is_open() && fos->good()))
        {
            std::cerr << "Output stream for " << id << " is not open/valid. "
                      << "Redirecting to stdout..." << std::endl;
            _os = &std::cout;
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
    uint64_t _width = tim_timer_t::get_output_width();
    // reset output width
    tim_timer_t::set_output_width(10);

    // redo output width calc, removing no displayed funcs
    for(const auto& itr : *this)
        if(itr.timer().above_min(no_min))
            tim_timer_t::propose_output_width(itr.timer().begin().length());

    // don't make it longer
    if(_width > 10 && _width < tim_timer_t::get_output_width())
        tim_timer_t::set_output_width(_width);

    for(const auto& itr : *this)
        itr.timer().report(*os, true, no_min);

    os->flush();
}

//============================================================================//

void timing_manager::set_output_stream(ostream_t& _os)
{
    m_report = &_os;
}

//============================================================================//

void timing_manager::set_output_stream(const string_t& fname)
{
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
            std::cerr << "Warning! Unable to open file " << _fname << ". "
                      << "Redirecting to stdout..." << std::endl;
            _fos->close();
            delete _fos;
            m_os = &std::cout;
        }
    };

    ostreamop(m_report, fname);

}

//============================================================================//

timing_manager::ofstream_t*
timing_manager::get_ofstream(ostream_t* m_os) const
{
    return static_cast<ofstream_t*>(m_os);
}

//============================================================================//

void timing_manager::add(pointer_type ptr)
{
    auto_lock_t lock(m_mutex);
#if defined(DEBUG)
    if(NAME_TIM::get_env("TIMEMORY_VERBOSE", 0) > 1)
        std::cout << "Adding " << ptr << " to " << this << "..." << std::endl;
#endif
    m_daughters.insert(ptr);
}

//============================================================================//

void timing_manager::merge(bool div_clock)
{
    if(!m_merge.load())
        return;

    m_merge.store(false);

    auto_lock_t lock(m_mutex);

    uomap<uint64_t, uint64_t> clock_div_count;
    for(auto& itr : m_daughters)
    {
        if(itr == f_instance)
            continue;

#if defined(DEBUG)
    if(NAME_TIM::get_env("TIMEMORY_VERBOSE", 0) > 1)
        std::cout << "Merging " << itr << "..." << std::endl;
#endif

        for(const auto& mitr : itr->map())
        {
            if(m_timer_map.find(mitr.first) == m_timer_map.end())
                m_timer_map[mitr.first] = mitr.second;
            else
                clock_div_count[mitr.first] += 1;

        }

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

    }

    if(div_clock)
        for(auto& itr : clock_div_count)
            *(m_timer_map[itr.first].get()) /= itr.second;

    for(auto& itr : m_daughters)
        if(itr != this)
            itr->clear();

}

//============================================================================//
//
//  Static functions for writing JSON output
//
//============================================================================//

// static function
std::pair<int32_t, bool> timing_manager::write_json(ostream_t& ofss)
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
void timing_manager::write_json(string_t _fname)
{
    (mpi_is_initialized()) ? write_json_mpi(_fname) : write_json_no_mpi(_fname);
}

//============================================================================//
// static function
void timing_manager::write_json_no_mpi(ostream_t& fss)
{
    fss << "{\n\"ranks\": [" << std::endl;

    // ensure json write final block during destruction before the file
    // is closed
    {
        auto spacing = cereal::JSONOutputArchive::Options::IndentChar::space;
        // precision, spacing, indent size
        cereal::JSONOutputArchive::Options opts(12, spacing, 4);
        cereal::JSONOutputArchive oa(fss, opts);

        oa(cereal::make_nvp("timing_manager", *timing_manager::instance()));
    }

    fss << "]" << "\n}" << std::endl;
}

//============================================================================//
// static function
void timing_manager::write_json_no_mpi(string_t _fname)
{
    int32_t _verbose = NAME_TIM::get_env<int32_t>("TIMEMORY_VERBOSE", 0);

    if(mpi_rank() == 0 && _verbose > 0)
    {
        // notify so if it takes too long, user knows why
        std::stringstream _info;
        _info << "Writing serialization file: "
              << _fname << std::endl;
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
std::pair<int32_t, bool> timing_manager::write_json_mpi(ostream_t& ofss)
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

        oa(cereal::make_nvp("timing_manager", *timing_manager::instance()));
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
void timing_manager::write_json_mpi(string_t _fname)
{
    {
        // notify so if it takes too long, user knows why
        std::stringstream _info;
        _info << "[" << mpi_rank() << "] Writing serialization file: "
              << _fname << std::endl;
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
timing_manager::comm_group_t
timing_manager::get_communicator_group()
{
    int32_t max_concurrency = std::thread::hardware_concurrency();
    // We want on-node communication only
    const int32_t nthreads = NAME_TIM::get_env<int32_t>("OMP_NUM_THREADS", 1);
    int32_t max_processes = max_concurrency / nthreads;
    int32_t mpi_node_default = mpi_size() / max_processes;
    if(mpi_node_default < 1)
        mpi_node_default = 1;
    int32_t mpi_node_count = NAME_TIM::get_env<int32_t>("TIMEMORY_NODE_COUNT",
                                                     mpi_node_default);
    int32_t mpi_split_size = mpi_rank() / (mpi_size() / mpi_node_count);

    // Split the communicator based on the number of nodes and use the
    // original rank for ordering
    MPI_Comm local_mpi_comm;
    MPI_Comm_split(MPI_COMM_WORLD, mpi_split_size, mpi_rank(), &local_mpi_comm);

#if defined(DEBUG)
    int32_t local_mpi_rank = mpi_rank(local_mpi_comm);
    int32_t local_mpi_size = mpi_size(local_mpi_comm);
    int32_t local_mpi_file = mpi_rank() / local_mpi_size;

    printf("WORLD RANK/SIZE: %d/%d --> ROW RANK/SIZE: %d/%d\n",
        mpi_rank(), mpi_size(), local_mpi_rank, local_mpi_size);

    std::stringstream _info;
    _info << mpi_rank() << " Rank      : " << mpi_rank() << std::endl;
    _info << mpi_rank() << " Node      : " << mpi_node_count << std::endl;
    _info << mpi_rank() << " Local Size: " << local_mpi_size << std::endl;
    _info << mpi_rank() << " Local Rank: " << local_mpi_rank << std::endl;
    _info << mpi_rank() << " Local File: " << local_mpi_file << std::endl;
    std::cout << _info.str();
#endif

    return comm_group_t(local_mpi_comm, mpi_rank() / mpi_size(local_mpi_comm));
}

//============================================================================//

} // namespace NAME_TIM
