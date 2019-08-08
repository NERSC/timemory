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

#pragma once

#if defined(TIMEMORY_USE_CALIPER)
#    include <caliper/cali.h>
#endif

#include "timemory/details/caliper_defs.hpp"
#include <string>

namespace tim
{
namespace cali
{
//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CALIPER)
using id_t = cali_id_t;
#else
using id_t = int;
#endif

//--------------------------------------------------------------------------------------//

template <typename... _Args>
inline void
cali_consume_parameters(_Args&&...)
{
}

//--------------------------------------------------------------------------------------//

inline void
init()
{
#if defined(TIMEMORY_USE_CALIPER)
    cali_init();
#endif
}

//--------------------------------------------------------------------------------------//

inline bool
is_initialized()
{
#if defined(TIMEMORY_USE_CALIPER)
    return static_cast<bool>(cali_is_initialized());
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

inline void
begin(const id_t& _id, const std::string& _label)
{
#if defined(TIMEMORY_USE_CALIPER)
    cali_begin_string(_id, _label.c_str());
#else
    cali_consume_parameters(_id, _label);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
begin(const std::string& _id, const std::string& _label)
{
#if defined(TIMEMORY_USE_CALIPER)
    cali_begin_string_byname(_id.c_str(), _label.c_str());
#else
    cali_consume_parameters(_id, _label);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
end(const id_t& _id)
{
#if defined(TIMEMORY_USE_CALIPER)
    cali_end(_id);
#else
    cali_consume_parameters(_id);
#endif
}

//--------------------------------------------------------------------------------------//

inline void
end(const std::string& _id)
{
#if defined(TIMEMORY_USE_CALIPER)
    cali_end_byname(_id.c_str());
#else
    cali_consume_parameters(_id);
#endif
}

//--------------------------------------------------------------------------------------//

template <typename _Type, typename _Attr>
inline id_t
create_attribute(const std::string& _id, _Type _type, _Attr _attr)
{
#if defined(TIMEMORY_USE_CALIPER)
    return cali_create_attribute(_id.c_str(), _type, _attr);
#else
    cali_consume_parameters(_id, _type, _attr);
    return 0;
#endif
}

//--------------------------------------------------------------------------------------//

}  // namespace cali
}  // namespace tim
