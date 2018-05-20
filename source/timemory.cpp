//  MIT License
//
//  Copyright (c) 2018, The Regents of the University of California, 
// through Lawrence Berkeley National Laboratory (subject to receipt of any 
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "timemory/macros.hpp"
#include "timemory/timemory.hpp"
#include "timemory/auto_timer.hpp"
#include "timemory/utility.hpp"
#include "timemory/timer.hpp"
#include "timemory/manager.hpp"
#include "timemory/singleton.hpp"
#include "timemory/base_timer.hpp"
#include "timemory/serializer.hpp"
#include "timemory/signal_detection.hpp"

#if !defined(pfunc)
#   if defined(DEBUG)
#       define pfunc printf("TiMemory -- calling %s@\"%s\":%i...\n", __FUNCTION__, __FILE__, __LINE__)
#   else
#       define pfunc
#   endif
#endif

//============================================================================//
// These two functions are guaranteed to be called at load and
// unload of the library containing this code.
//__c_ctor__
void setup_timemory(void)
{
    pfunc;
    _timemory_initialization();
    pfunc;
}

//============================================================================//

//__c_dtor__
void cleanup_timemory(void)
{
    pfunc;
    _timemory_finalization();
    pfunc;
}

//============================================================================//
//
//                      C++ interface
//
//============================================================================//

extern "C" tim_api
void cxx_timemory_initialization(void)
{
    pfunc;
    _timemory_initialization();
    pfunc;
}

//============================================================================//

extern "C" tim_api
void cxx_timemory_finalization(void)
{
    pfunc;
    _timemory_finalization();
    pfunc;
}

//============================================================================//

extern "C" tim_api
int cxx_timemory_enabled(void)
{
    return (tim::auto_timer::alloc_next()) ? 1 : 0;
}

//============================================================================//

extern "C" tim_api
void* cxx_timemory_create_auto_timer(const char* timer_tag,
                                     int lineno,
                                     const char* lang_tag,
                                     int report)
{
    std::string cxx_timer_tag(timer_tag);
    char* _timer_tag = (char*) timer_tag;
    free(_timer_tag);
    return (void*) new auto_timer_t(cxx_timer_tag.c_str(),
                                    lineno,
                                    lang_tag,
                                    (report > 0) ? true : false);
}

//============================================================================//

extern "C" tim_api
void* cxx_timemory_delete_auto_timer(void* ctimer)
{
    auto_timer_t* cxxtimer = static_cast<auto_timer_t*>(ctimer);
    delete cxxtimer;
    ctimer = NULL;
    return ctimer;
}

//============================================================================//

extern "C" tim_api
const char* cxx_timemory_string_combine(const char* _a, const char* _b)
{
    char* buff = (char*) malloc(sizeof(char) * 256);
    sprintf(buff, "%s%s", _a, _b);
    return (const char*) buff;
}

//============================================================================//

extern "C" tim_api
const char* cxx_timemory_auto_timer_str(const char* _a, const char* _b,
                                        const char* _c, int _d)
{
    std::string _C = std::string(_c).substr(std::string(_c).find_last_of("/")+1);
    char* buff = (char*) malloc(sizeof(char) * 256);
    sprintf(buff, "%s%s@'%s':%i", _a, _b, _C.c_str(), _d);
    return (const char*) buff;
}

//============================================================================//

extern "C" tim_api
void cxx_timemory_report(const char* fname)
{
    std::string _fname(fname);
    for(auto itr : {".txt", ".out", ".json"})
    {
        if(_fname.find(itr) != std::string::npos)
            _fname = _fname.substr(0, _fname.find(itr));
    }
    _fname = _fname.substr(0, _fname.find_last_of("."));

    tim::path_t _fpath_report = _fname + std::string(".out");
    tim::path_t _fpath_serial = _fname + std::string(".json");
    tim::manager::master_instance()->set_output_stream(_fpath_report);
    tim::makedir(tim::dirname(_fpath_report));
    std::ofstream ofs_report(_fpath_report);
    std::ofstream ofs_serial(_fpath_serial);
    if(ofs_report)
        tim::manager::master_instance()->report(ofs_report);
    if(ofs_serial)
        tim::manager::master_instance()->write_json(ofs_serial);
}

//============================================================================//

extern "C" tim_api
void cxx_timemory_print(void)
{
    tim::manager::master_instance()->report(std::cout, true);
}

//============================================================================//

extern "C" tim_api
void cxx_timemory_record_memory(int _record_memory)
{
    tim::timer::default_record_memory((_record_memory > 0) ? true : false);
}

//============================================================================//
//
//                      Serialization
//
//============================================================================//

CEREAL_CLASS_VERSION(tim::timer_tuple, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(tim::manager, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(tim::timer, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(tim::internal::base_timer_data, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(tim::internal::base_timer, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_clock_t, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_clock_data_t, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_duration_t, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_time_point_t, TIMEMORY_TIMER_VERSION)
CEREAL_CLASS_VERSION(internal::base_time_pair_t, TIMEMORY_TIMER_VERSION)

//============================================================================//
