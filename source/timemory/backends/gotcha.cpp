//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

#ifndef TIMEMORY_BACKENDS_GOTCHA_CPP_
#define TIMEMORY_BACKENDS_GOTCHA_CPP_

#include "timemory/backends/defines.hpp"

#if !defined(TIMEMORY_BACKENDS_HEADER_ONLY_MODE) ||                                      \
    (defined(TIMEMORY_BACKENDS_HEADER_ONLY_MODE) &&                                      \
     TIMEMORY_BACKENDS_HEADER_ONLY_MODE == 0)
#    include "timemory/backends/gotcha.hpp"
#endif

namespace tim
{
namespace backend
{
namespace gotcha
{
TIMEMORY_BACKENDS_INLINE
bool
initialize()
{
#if defined(TIMEMORY_USE_GOTCHA) && defined(GOTCHA_INIT_EXT) && GOTCHA_INIT_EXT > 0
    static auto _once = []() {
        const char* _env = TIMEMORY_SETTINGS_PREFIX "GOTCHA_WRAP_LIBDL";
        auto        _v   = get_env<bool>(_env, false);
        auto        _ret = ::gotcha_init_ext((_v) ? 1 : 0);
        if(_v && _ret != 0)
        {
            fprintf(stderr,
                    "[%s][gotcha_init] gotcha library is already initialized. %s=true "
                    "ignored.\n",
                    TIMEMORY_PROJECT_NAME, _env);
        }
        return (_ret == 0);
    }();
    return _once;
#else
    return false;
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
const char*
get_error(error_t err)
{
    switch(err)
    {
        case GOTCHA_SUCCESS: return "success";
        case GOTCHA_FUNCTION_NOT_FOUND: return "function not found";
        case GOTCHA_INTERNAL: return "internal error";
        case GOTCHA_INVALID_TOOL: return "invalid tool";
    }
    return "unknown";
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
error_t
set_priority(const char* _tool, int _priority)
{
    initialize();
    if(settings::debug())
    {
        printf("[gotcha::%s]> Setting priority for tool: %s to %i...\n", __FUNCTION__,
               _tool, _priority);
    }
#if defined(TIMEMORY_USE_GOTCHA)
    // return GOTCHA_SUCCESS;
    error_t _ret = gotcha_set_priority(_tool, _priority);
    if(_ret != GOTCHA_SUCCESS)
        printf("[gotcha::%s]> Warning! set_priority == %i failed for '%s'. err %i: %s\n",
               __FUNCTION__, _priority, _tool, static_cast<int>(_ret), get_error(_ret));
    return _ret;
#else
    if(settings::debug())
        printf("[gotcha::%s]> Warning! GOTCHA not truly enabled!", __FUNCTION__);
    return GOTCHA_SUCCESS;
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
error_t
get_priority(const char* _tool, int& _priority)
{
    initialize();
    if(settings::debug())
    {
        printf("[gotcha::%s]> Getting priority for tool: %s to %i...\n", __FUNCTION__,
               _tool, _priority);
    }
#if defined(TIMEMORY_USE_GOTCHA)
    // return GOTCHA_SUCCESS;
    error_t _ret = gotcha_get_priority(_tool, &_priority);
    if(_ret != GOTCHA_SUCCESS)
        printf("[gotcha::%s]> Warning! get_priority == %i failed for '%s'. err %i: %s\n",
               __FUNCTION__, _priority, _tool, static_cast<int>(_ret), get_error(_ret));
    return _ret;
#else
    if(settings::debug())
        printf("[gotcha::%s]> Warning! GOTCHA not truly enabled!", __FUNCTION__);
    return GOTCHA_SUCCESS;
#endif
}

//--------------------------------------------------------------------------------------//

TIMEMORY_BACKENDS_INLINE
error_t
wrap(binding_t& _bind, const char* _label)
{
    error_t _ret = GOTCHA_SUCCESS;

    initialize();
    if(settings::debug())
        printf("[gotcha::%s]> Adding tool: %s...\n", __FUNCTION__, _label);

#if defined(TIMEMORY_USE_GOTCHA)
    if(_ret == GOTCHA_SUCCESS)
        _ret = gotcha_wrap(&_bind, 1, _label);
#else
    if(settings::debug())
        printf("[gotcha::%s]> Warning! GOTCHA not truly enabled!", __FUNCTION__);
    consume_parameters(_bind);
#endif

    return _ret;
}
}  // namespace gotcha
}  // namespace backend
}  // namespace tim

#endif
