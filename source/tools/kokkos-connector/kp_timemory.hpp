
#pragma once

#include "timemory/timemory.hpp"
#include <random>

//--------------------------------------------------------------------------------------//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
/// \class rand_intercept
/// \brief this component can be used to replace the C implementation of rand()
///
struct rand_intercept : public base<rand_intercept, void>
{
    using random_engine_t = std::default_random_engine;
    using distribution_t  = std::uniform_int_distribution<int>;
    using limits_t        = std::numeric_limits<int>;

    static void message()
    {
        // deliver one-time message
        static int num = (puts("intercepting rand..."), 0);
        consume_parameters(num);
    }

    static random_engine_t& get_generator()
    {
        static random_engine_t generator(tim::get_env("RANDOM_SEED", time(NULL)));
        return generator;
    }

    static int get_rand()
    {
        static auto&          generator = get_generator();
        static distribution_t dist(0, limits_t::max());
        return dist(generator);
    }

    rand_intercept() { message(); }

    // replaces 'void srand(unsigned)' with STL implementation
    void operator()(unsigned rseed) { get_generator() = random_engine_t(rseed); }

    // replaces 'int rand(void)' with STL implementation
    int operator()() { return get_rand(); }
};

//--------------------------------------------------------------------------------------//

}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//

using namespace tim::component;

using empty_t       = tim::component_tuple<>;
using counter_t     = tim::component_tuple<trip_count>;
using rand_gotcha_t = tim::component::gotcha<2, empty_t, rand_intercept>;
using misc_gotcha_t = tim::component::gotcha<8, counter_t, void>;

//--------------------------------------------------------------------------------------//

using gotcha_tools_t = tim::component_tuple<rand_gotcha_t, misc_gotcha_t>;

//--------------------------------------------------------------------------------------//
