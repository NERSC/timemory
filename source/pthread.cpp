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

#include "timemory/backends/threading.hpp"
#include "timemory/compat/library.h"
#include "timemory/library.h"
#include "timemory/manager.hpp"
#include "timemory/operations/types/storage_initializer.hpp"
#include "timemory/runtime/configure.hpp"
#include "timemory/timemory.hpp"
#include "timemory/trace.hpp"
#include "timemory/utility/bits/signals.hpp"

#if !defined(_WINDOWS)

#    include <pthread.h>

using namespace tim::component;

//--------------------------------------------------------------------------------------//

struct pthread_gotcha : tim::component::base<pthread_gotcha, void>
{
    template <size_t... Idx>
    static auto init_storage(tim::index_sequence<Idx...>)
    {
        return TIMEMORY_RETURN_FOLD_EXPRESSION(tim::storage_initializer::get<Idx>());
    }

    struct wrapper
    {
        typedef void* (*routine_t)(void*);

        TIMEMORY_DEFAULT_OBJECT(wrapper)

        wrapper(routine_t _routine, void* _arg)
        : m_routine(_routine)
        , m_arg(_arg)
        {}

        void* operator()() const { return m_routine(m_arg); }

        static void* wrap(void* _arg)
        {
            if(!_arg)
                return nullptr;

            // create the manager and initialize the storage
            static thread_local auto _tlm = tim::manager::instance();
            auto _ini = init_storage(tim::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
            tim::consume_parameters(_tlm, _ini);
            // execute the original function
            wrapper* _wrapper = static_cast<wrapper*>(_arg);
            return (*_wrapper)();
        }

    private:
        routine_t m_routine = nullptr;
        void*     m_arg     = nullptr;
    };

    // pthread_create
    int operator()(pthread_t* thread, const pthread_attr_t* attr,
                   void* (*start_routine)(void*), void*     arg)
    {
        // tim::trace::lock<tim::trace::threading> lk{};
        // if(!lk)
        //    return;
        PRINT_HERE("%s", "wrapping pthread_create");
        auto  _obj  = wrapper(start_routine, arg);
        void* _vobj = static_cast<void*>(&_obj);
        return pthread_create(thread, attr, &wrapper::wrap, _vobj);
    }

    // pthread_join
    int operator()(pthread_t thread, void** retval)
    {
        // tim::trace::lock<tim::trace::threading> lk{};
        // if(!lk)
        //    return;
        PRINT_HERE("%s", "wrapping pthread_join");
        tim::manager::instance()->finalize();
        return pthread_join(thread, retval);
    }
};

//--------------------------------------------------------------------------------------//

using empty_tuple_t    = tim::component_tuple<>;
using pthread_gotcha_t = tim::component::gotcha<2, empty_tuple_t, pthread_gotcha>;
using pthread_bundle_t = tim::component_tuple<pthread_gotcha_t>;

//--------------------------------------------------------------------------------------//

auto
setup_pthread_gotcha()
{
#    if defined(TIMEMORY_USE_GOTCHA)
    pthread_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(pthread_gotcha_t, 0, pthread_create);
        TIMEMORY_C_GOTCHA(pthread_gotcha_t, 1, pthread_join);
    };
#    endif
    return std::make_shared<pthread_bundle_t>("pthread");
}

//--------------------------------------------------------------------------------------//

namespace
{
static auto pthread_gotcha_handle = setup_pthread_gotcha();
}

#else
namespace
{
static auto pthread_gotcha_handle = false;
}
#endif
