//  MIT License
//  
//  Copyright (c) 2018, The Regents of the University of California, 
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

/** \file environment.hpp
 * \headerfile environment.hpp "timemory/environment.hpp"
 * Handles TiMemory settings via environment
 *
 */

#ifndef environment_hpp_
#define environment_hpp_

//----------------------------------------------------------------------------//

#include <timemory/macros.hpp>

#include <cstdint>
#include <string>
#include <limits>
#include <cstring>

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#   define TIMEMORY_DEFAULT_ENABLED true
#endif

//----------------------------------------------------------------------------//

namespace tim
{

namespace env
{

//----------------------------------------------------------------------------//

typedef std::string string_t;

//----------------------------------------------------------------------------//

extern  int         verbose;
extern  bool        disable_timer_memory;
//extern  bool        output_total;
extern  string_t    env_num_threads;
extern  int         num_threads;
extern  int         max_depth;
extern  bool        enabled;

extern  string_t    timing_format;
extern  int16_t     timing_precision;
extern  int16_t     timing_width;
extern  string_t    timing_units;
extern  bool        timing_scientific;

extern  string_t    memory_format;
extern  int16_t     memory_precision;
extern  int16_t     memory_width;
extern  string_t    memory_units;
extern  bool        memory_scientific;

extern  string_t    timing_memory_format;
extern  int16_t     timing_memory_precision;
extern  int16_t     timing_memory_width;
extern  string_t    timing_memory_units;
extern  bool        timing_memory_scientific;

//----------------------------------------------------------------------------//

string_t tolower(string_t);
string_t toupper(string_t);
void parse();

//----------------------------------------------------------------------------//

} // namespace env

} // namespace tim

//----------------------------------------------------------------------------//

#endif

