//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

/** \file init.cpp
 * This file defined the extern init
 *
 */

#include "timemory/extern/init.hpp"
#include "timemory/components.hpp"
#include "timemory/environment/declaration.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/plotting.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

using namespace tim::component;

#if defined(TIMEMORY_USE_EXTERN)

extern "C"
{
    extern ::tim::manager* timemory_manager_master_instance();
}

//======================================================================================//

#    if defined(TIMEMORY_USE_MPI)

extern "C"
{
    //----------------------------------------------------------------------------------//

    int timemory_MPI_Finalize(MPI_Comm, int, void*, void*)
    {
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Finalize!\n", __FUNCTION__,
                   __FILE__, __LINE__);
        }
        auto manager = timemory_manager_master_instance();
        if(manager)
            manager->finalize();
        ::tim::dmp::is_finalized() = true;
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory MPI_Finalize completed!\n", __FUNCTION__,
                   __FILE__, __LINE__);
        }
        return MPI_SUCCESS;
    }

    //----------------------------------------------------------------------------------//

    void timemory_MPI_Init(int* argc, char*** argv)
    {
        int comm_key = 0;
        MPI_Comm_create_keyval(MPI_NULL_COPY_FN, &timemory_MPI_Finalize, &comm_key, NULL);
        MPI_Comm_set_attr(MPI_COMM_SELF, comm_key, NULL);

        static auto _manager = timemory_manager_master_instance();
        tim::consume_parameters(_manager);
        ::tim::timemory_init(*argc, *argv);
    }

    //----------------------------------------------------------------------------------//
    //
#        if !defined(TIMEMORY_MPI_INIT) || (TIMEMORY_MPI_INIT > 0)
    //
    int MPI_Init(int* argc, char*** argv)
    {
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Init!\n", __FUNCTION__, __FILE__,
                   __LINE__);
        }
#            if defined(TIMEMORY_USE_TAU)
        Tau_init(*argc, *argv);
#            endif
        auto ret = PMPI_Init(argc, argv);
        timemory_MPI_Init(argc, argv);
        return ret;
    }

    //----------------------------------------------------------------------------------//

    int MPI_Init_thread(int* argc, char*** argv, int req, int* prov)
    {
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Init_thread!\n", __FUNCTION__,
                   __FILE__, __LINE__);
        }
#            if defined(TIMEMORY_USE_TAU)
        Tau_init(*argc, *argv);
#            endif
        auto ret = PMPI_Init_thread(argc, argv, req, prov);
        timemory_MPI_Init(argc, argv);
        return ret;
    }
    //
#        endif
    //
    //----------------------------------------------------------------------------------//
}

#    endif

//======================================================================================//

#endif  // defined(TIMEMORY_USE_EXTERN)
