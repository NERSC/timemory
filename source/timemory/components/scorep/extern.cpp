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

#include "timemory/components/scorep/extern.hpp"
#include <cstddef>

#include <scorep/SCOREP_MetricPlugins.h>
#include <scorep/SCOREP_MetricTypes.h>
#include <scorep/SCOREP_PublicHandles.h>
#include <scorep/SCOREP_PublicTypes.h>
//#include <scorep/SCOREP_Score_Estimator.hpp>
#include <scorep/SCOREP_Score_Event.hpp>
#include <scorep/SCOREP_Score_Group.hpp>
//#include <scorep/SCOREP_Score_Profile.hpp>
#include <scorep/SCOREP_Score_Types.hpp>
#include <scorep/SCOREP_SubstrateEvents.h>
#include <scorep/SCOREP_SubstratePlugins.h>
#include <scorep/SCOREP_User_Functions.h>
#include <scorep/SCOREP_User.h>
#include <scorep/SCOREP_User_Types.h>
#include <scorep/SCOREP_User_Variables.h>

extern const struct SCOREP_Subsystem SCOREP_Subsystem_Substrates;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_TaskStack;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_MetricService;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_UnwindingService;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_SamplingService;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_Topologies;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_PlatformTopology;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_UserAdapter;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_PthreadAdapter;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_ThreadCreateWait;
#if defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_INIT)
extern const struct SCOREP_Subsystem SCOREP_Subsystem_IoManagement;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_MpiAdapter;
/*
extern const struct SCOREP_Subsystem SCOREP_AllocMetric_New;
extern const struct SCOREP_Subsystem SCOREP_AllocMetric_GetAllocationSizeAttribute;
extern const struct SCOREP_Subsystem SCOREP_AllocMetric_GetDeallocationSizeAttribute;
extern const struct SCOREP_Subsystem SCOREP_AllocMetric_ReportLeaked;
extern const struct SCOREP_Subsystem SCOREP_AllocMetric_Destroy;
extern const struct SCOREP_Subsystem SCOREP_AllocMetric_AcquireAlloc;
extern const struct SCOREP_Subsystem SCOREP_AllocMetric_HandleFree;
extern const struct SCOREP_Subsystem SCOREP_AllocMetric_HandleAlloc;
*/
#endif

const struct SCOREP_Subsystem* scorep_subsystems[] = {
    &SCOREP_Subsystem_Substrates,       
    &SCOREP_Subsystem_TaskStack,
    &SCOREP_Subsystem_MetricService,    
    &SCOREP_Subsystem_UnwindingService,
    &SCOREP_Subsystem_SamplingService,  
    &SCOREP_Subsystem_Topologies,
    &SCOREP_Subsystem_PlatformTopology, 
    &SCOREP_Subsystem_UserAdapter,
    &SCOREP_Subsystem_PthreadAdapter,   
    &SCOREP_Subsystem_ThreadCreateWait,
#if defined(TIMEMORY_USE_MPI) && defined(TIMEMORY_USE_MPI_INIT)
    &SCOREP_Subsystem_IoManagement,     
    &SCOREP_Subsystem_MpiAdapter,
    /*    &SCOREP_AllocMetric_New,
    &SCOREP_AllocMetric_GetAllocationSizeAttribute,
    &SCOREP_AllocMetric_GetDeallocationSizeAttribute,
    &SCOREP_AllocMetric_ReportLeaked,
    &SCOREP_AllocMetric_Destroy,
    &SCOREP_AllocMetric_AcquireAlloc,
    &SCOREP_AllocMetric_HandleFree,
    &SCOREP_AllocMetric_HandleAlloc,*/
#endif
};

const size_t scorep_number_of_subsystems =
    sizeof(scorep_subsystems) / sizeof(scorep_subsystems[0]);
