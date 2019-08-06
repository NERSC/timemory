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
//

#include "preload_api.hpp"
#include <execinfo.h>
#include <iostream>

static uint64_t                         uniqID = 0;
static string_t                         components_string;
static std::vector<string_t>            components_string_vector;
static std::vector<TIMEMORY_COMPONENT>  components_vector;
static std::map<std::string, uint64_t>  components_keys;
static std::map<uint64_t, auto_list_t*> record_map;
static string_t spacer = "---------------------------------------------------------";

//--------------------------------------------------------------------------------------//
//
//      TiMemory start/stop
//
//--------------------------------------------------------------------------------------//

void
record_start(const char* name, uint64_t* kernid)
{
    auto itr = components_keys.find(std::string(name));
    if(itr != components_keys.end())
    {
        *kernid = itr->second;
        record_map[*kernid]->reset();
        record_map[*kernid]->push();   // insert into call-graph tree
        record_map[*kernid]->start();  // start recording
    }
    else
    {
        *kernid               = uniqID++;
        components_keys[name] = *kernid;
        auto obj = new auto_list_t(name, *kernid, tim::language::cxx(), false);
        tim::initialize(*obj, components_vector);
        record_map[*kernid] = obj;
        record_map[*kernid]->push();   // insert into call-graph tree
        record_map[*kernid]->start();  // start recording
    }
}

//--------------------------------------------------------------------------------------//

void
record_stop(uint64_t kernid)
{
    record_map[kernid]->stop();  // stop recording
    record_map[kernid]->pop();   // pop off of call-graph tree
}

//--------------------------------------------------------------------------------------//
//
//      TiMemory symbols
//
//--------------------------------------------------------------------------------------//

extern "C" void
timemory_init_library(int argc, char** argv)
{
    printf("%s\n", spacer.c_str());
    printf("TiMemory Connector\n");
    printf("%s\n\n", spacer.c_str());

    tim::settings::auto_output() = true;   // print when destructing
    tim::settings::cout_output() = true;   // print to stdout
    tim::settings::text_output() = true;   // print text files
    tim::settings::json_output() = false;  // print to json
    tim::timemory_init(argc, argv);

    components_string =
        tim::get_env<string_t>("TIMEMORY_COMPONENTS", get_default_components());
    components_string_vector = tim::delimit(components_string);
    components_vector        = tim::enumerate_components(components_string_vector);
}

//--------------------------------------------------------------------------------------//

extern "C" void
timemory_finalize_library()
{
    printf("\n%s\n", spacer.c_str());
    printf("Finalization of TiMemory Connector.\n");
    printf("%s\n\n", spacer.c_str());

    for(auto& itr : record_map)
        delete itr.second;
    record_map.clear();

    // PGI and Intel compilers don't respect destruction order
#if defined(__PGI) || defined(__INTEL_COMPILER)
    tim::settings::auto_output() = false;
#endif

    // Compensate for Intel compiler not allowing auto output
#if defined(__INTEL_COMPILER)
    auto_list_t::component_type::print_storage();
#endif
}

//--------------------------------------------------------------------------------------//

extern "C" void
timemory_begin_record(const char* name, uint64_t* kernid)
{
    record_start(name, kernid);
}

//--------------------------------------------------------------------------------------//

extern "C" void
timemory_end_record(uint64_t kernid)
{
    record_stop(kernid);
}

//--------------------------------------------------------------------------------------//
