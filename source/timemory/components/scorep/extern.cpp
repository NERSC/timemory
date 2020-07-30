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
#include <stddef.h>

extern const struct SCOREP_Subsystem SCOREP_Subsystem_Substrates;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_TaskStack;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_MetricService;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_UnwindingService;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_SamplingService;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_Topologies;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_PlatformTopology;
extern const struct SCOREP_Subsystem SCOREP_Subsystem_UserAdapter;

const struct SCOREP_Subsystem* scorep_subsystems[] = {
    &SCOREP_Subsystem_Substrates,       &SCOREP_Subsystem_TaskStack,
    &SCOREP_Subsystem_MetricService,    &SCOREP_Subsystem_UnwindingService,
    &SCOREP_Subsystem_SamplingService,  &SCOREP_Subsystem_Topologies,
    &SCOREP_Subsystem_PlatformTopology, &SCOREP_Subsystem_UserAdapter
};

const size_t scorep_number_of_subsystems =
    sizeof(scorep_subsystems) / sizeof(scorep_subsystems[0]);
