// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
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

/**
 * \file timemory/components/caliper/backends.hpp
 * \brief Implementation of the caliper functions/utilities
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

//======================================================================================//

#if defined(TIMEMORY_USE_CALIPER)

#    include <caliper/cali.h>

#else

#    define CALI_INV_ID 0xFFFFFFFFFFFFFFFF

using cali_id_t = uint64_t;

typedef enum
{
    CALI_TYPE_INV    = 0, /**< Invalid type               */
    CALI_TYPE_USR    = 1, /**< User-defined type (pointer to binary data) */
    CALI_TYPE_INT    = 2, /**< 64-bit signed integer      */
    CALI_TYPE_UINT   = 3, /**< 64-bit unsigned integer    */
    CALI_TYPE_STRING = 4, /**< String (\a char*)          */
    CALI_TYPE_ADDR   = 5, /**< 64-bit address             */
    CALI_TYPE_DOUBLE = 6, /**< Double-precision floating point type */
    CALI_TYPE_BOOL   = 7, /**< C or C++ boolean           */
    CALI_TYPE_TYPE   = 8, /**< Instance of cali_attr_type */
    CALI_TYPE_PTR    = 9  /**< Raw pointer. Internal use only. */
} cali_attr_type;

#    define CALI_MAXTYPE CALI_TYPE_PTR

typedef enum
{
    CALI_ATTR_DEFAULT       = 0,
    CALI_ATTR_ASVALUE       = 1,
    CALI_ATTR_NOMERGE       = 2,
    CALI_ATTR_SCOPE_PROCESS = 12, /* scope flags are mutually exclusive */
    CALI_ATTR_SCOPE_THREAD  = 20,
    CALI_ATTR_SCOPE_TASK    = 24,
    CALI_ATTR_SKIP_EVENTS   = 64,
    CALI_ATTR_HIDDEN        = 128,
    CALI_ATTR_NESTED        = 256,
    CALI_ATTR_GLOBAL        = 512
} cali_attr_properties;

#    define CALI_ATTR_SCOPE_MASK 60

typedef enum
{
    CALI_OP_SUM = 1,
    CALI_OP_MIN = 2,
    CALI_OP_MAX = 3
} cali_op;

typedef enum
{
    CALI_SUCCESS = 0,
    CALI_EBUSY,
    CALI_ELOCKED,
    CALI_EINV,
    CALI_ETYPE,
    CALI_ESTACK
} cali_err;

#endif  // TIMEMORY_USE_CALIPER

//======================================================================================//
//
namespace tim
{
namespace backend
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

template <typename... ArgsT>
inline void
cali_consume_parameters(ArgsT&&...)
{}

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

template <typename Tp, typename AttrT>
inline id_t
create_attribute(const std::string& _id, Tp _type, AttrT _attr)
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
}  // namespace backend
}  // namespace tim
//
//======================================================================================//
