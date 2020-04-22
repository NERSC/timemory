
#pragma once

#include "timemory/timemory.hpp"

//--------------------------------------------------------------------------------------//

using namespace tim::component;

using empty_t   = tim::component_tuple<>;
using counter_t = tim::component_tuple<trip_count>;

//--------------------------------------------------------------------------------------//

struct SpaceHandle
{
    char name[64];
};

//--------------------------------------------------------------------------------------//
