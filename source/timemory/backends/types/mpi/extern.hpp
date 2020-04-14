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

/** \file extern/init.hpp
 * \headerfile extern/init.hpp "timemory/extern/init.hpp"
 * Provides extern initialization
 *
 */

#pragma once

#if defined(TIMEMORY_USE_MPI)

#    include "timemory/backends/mpi.hpp"
#    include "timemory/config.hpp"
#    include "timemory/manager/declaration.hpp"

#    if defined(TIMEMORY_USE_TAU)
#        define TAU_ENABLED
#        define TAU_DOT_H_LESS_HEADERS
#        include "TAU.h"
#    endif

//--------------------------------------------------------------------------------------//
//
#    if !defined(TAU_INIT)
#        if defined(TIMEMORY_USE_TAU)
#            define TAU_INIT(...) Tau_init(__VA_ARGS__)
#        else
#            define TAU_INIT(...)
#        endif
#    endif
//
//--------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------//
//
#    if defined(TIMEMORY_MPI_INIT_SOURCE)
//
#        define TIMEMORY_MPI_INIT_LINKAGE(...) __VA_ARGS__
//
#    else
//
#        if !defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_MPI_INIT_EXTERN)
#            define TIMEMORY_MPI_INIT_LINKAGE(...) inline __VA_ARGS__
#        else
#            define TIMEMORY_MPI_INIT_LINKAGE(...) extern __VA_ARGS__
#        endif
//
#    endif
//
//--------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------//
//
extern "C"
{
    //
    //----------------------------------------------------------------------------------//
    //
#    if defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_MPI_INIT_EXTERN) ||         \
        defined(TIMEMORY_MPI_INIT_SOURCE)
    extern ::tim::manager* timemory_manager_master_instance();
#    endif
    //
    //----------------------------------------------------------------------------------//
    //
#    if defined(TIMEMORY_USE_EXTERN) || defined(TIMEMORY_USE_MPI_INIT_EXTERN)
    //
    //----------------------------------------------------------------------------------//
    //
    extern void timemory_MPI_Comm_set_attr(void);
    extern void timemory_MPI_Init(int* argc, char*** argv);
    extern int  MPI_Init(int* argc, char*** argv);
#        if !defined(TIMEMORY_MPI_INIT) || (TIMEMORY_MPI_INIT > 0)
    extern int MPI_Init_thread(int* argc, char*** argv, int required, int* provided);
#        endif
#    else
    //
    //----------------------------------------------------------------------------------//
    //
    static int timemory_MPI_Finalize(MPI_Comm, int, void*, void*)
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
    //
    //----------------------------------------------------------------------------------//
    //
    TIMEMORY_MPI_INIT_LINKAGE(void) timemory_MPI_Comm_set_attr(void)
    {
        int comm_key = 0;
        MPI_Comm_create_keyval(MPI_NULL_COPY_FN, &timemory_MPI_Finalize, &comm_key, NULL);
        MPI_Comm_set_attr(MPI_COMM_SELF, comm_key, NULL);

        static auto _manager = timemory_manager_master_instance();
        tim::consume_parameters(_manager);
    }
    //
    //----------------------------------------------------------------------------------//
    //
    TIMEMORY_MPI_INIT_LINKAGE(void) timemory_MPI_Init(int* argc, char*** argv)
    {
        timemory_MPI_Comm_set_attr();
        ::tim::timemory_init(*argc, *argv);
    }
    //
    //----------------------------------------------------------------------------------//
    //
#        if !defined(TIMEMORY_MPI_INIT) || (TIMEMORY_MPI_INIT > 0)
    //
    //----------------------------------------------------------------------------------//
    //
    int MPI_Init(int* argc, char*** argv)
    {
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Init!\n", __FUNCTION__, __FILE__,
                   __LINE__);
        }
        TAU_INIT(argc, argv);
        auto ret = PMPI_Init(argc, argv);
        timemory_MPI_Init(argc, argv);
        return ret;
    }
    //
    //----------------------------------------------------------------------------------//
    //
    int MPI_Init_thread(int* argc, char*** argv, int req, int* prov)
    {
        if(tim::settings::debug())
        {
            printf("[%s@%s:%i]> timemory intercepted MPI_Init_thread!\n", __FUNCTION__,
                   __FILE__, __LINE__);
        }
        TAU_INIT(argc, argv);
        auto ret = PMPI_Init_thread(argc, argv, req, prov);
        timemory_MPI_Init(argc, argv);
        return ret;
    }
    //
    //----------------------------------------------------------------------------------//
    //
#        endif
#    endif
    //
    //----------------------------------------------------------------------------------//
    //
}
//
//--------------------------------------------------------------------------------------//
//
#endif  // defined(TIMEMORY_USE_MPI)
