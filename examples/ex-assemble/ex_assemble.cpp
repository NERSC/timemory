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

#include "timemory/timemory.hpp"

#include <cstdio>
#include <cstdlib>

//
// shorthand
//
using namespace tim::component;
using tim::type_list;
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DECLARE_COMPONENT(assembled_cpu_util)
TIMEMORY_STATISTICS_TYPE(component::assembled_cpu_util, double)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_percent_units, component::assembled_cpu_util,
                               true_type)

//--------------------------------------------------------------------------------------//

template <typename Tp>
struct remove_pointers;

template <template <typename...> class Tuple, typename... Tp>
struct remove_pointers<Tuple<Tp...>>
{
    using type = Tuple<std::remove_pointer_t<Tp>...>;
};

template <typename Tp>
using remove_pointers_t = typename remove_pointers<Tp>::type;

namespace tim
{
namespace component
{
struct assembled_cpu_util : public base<assembled_cpu_util, double>
{
    using ratio_t    = std::nano;
    using value_type = double;
    using this_type  = assembled_cpu_util;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "assembled_cpu_util"; }
    static std::string description() { return "cpu utilization (assembled)"; }

    struct pair_match;
    struct triplet_match;
    struct no_match;

    template <typename Tp>
    struct assembled_match
    {
        using TupleT = remove_pointers_t<Tp>;

        static constexpr bool pair_v =
            (is_one_of<wall_clock, TupleT>::value && is_one_of<cpu_clock, TupleT>::value);

        static constexpr bool triplet_v = (is_one_of<wall_clock, TupleT>::value &&
                                           is_one_of<user_clock, TupleT>::value &&
                                           is_one_of<system_clock, TupleT>::value);
        using type                      = conditional_t<(pair_v), pair_match,
                                   conditional_t<(triplet_v), triplet_match, no_match>>;
    };

    template <typename Tp>
    using assembled_match_t = typename assembled_match<Tp>::type;

    double get() const
    {
        return (is_transient) ? accum : value;
    }

    double get_display() const { return get(); }

    void start() {}
    void stop() {}

    template <template <typename...> class Tuple, typename T, typename... Tail,
              typename MatchT = assembled_match_t<Tuple<T, Tail...>>,
              enable_if_t<(std::is_same<MatchT, pair_match>::value), int> = 0>
    void dismantle(const Tuple<T, Tail...>& wrapper)
    {
        compute(wrapper.template get<wall_clock>(), wrapper.template get<cpu_clock>());
    }

    template <template <typename...> class Tuple, typename T, typename... Tail,
              typename MatchT = assembled_match_t<Tuple<T, Tail...>>,
              enable_if_t<(std::is_same<MatchT, triplet_match>::value), int> = 0>
    void dismantle(const Tuple<T, Tail...>& wrapper)
    {
        compute(wrapper.template get<wall_clock>(), wrapper.template get<user_clock>(),
                wrapper.template get<system_clock>());
    }

    void compute(const wall_clock* wc, const cpu_clock* cc)
    {
        if(wc && cc)
        {
            value = 100.0 * (cc->get() / wc->get());
            accum += value;
        }
    }

    void compute(const wall_clock* wc, const user_clock* uc, const system_clock* sc)
    {
        if(wc && uc && sc)
        {
            value = 100.0 * ((uc->get() + sc->get()) / wc->get());
            accum += value;
        }
    }
};
}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
// bundle of tools
//
using pair_tuple_t = tim::auto_tuple<wall_clock, cpu_clock, assembled_cpu_util>;
using pair_list_t  = tim::auto_list<wall_clock, cpu_clock, peak_rss, assembled_cpu_util>;

using triplet_tuple_t =
    tim::auto_tuple<wall_clock, user_clock, system_clock, peak_rss, assembled_cpu_util>;
using triplet_list_t =
    tim::auto_list<wall_clock, user_clock, system_clock, assembled_cpu_util>;

//--------------------------------------------------------------------------------------//

long
fib(long n)
{
    return (n < 2) ? n : (fib(n - 1) + fib(n - 2));
}

//--------------------------------------------------------------------------------------//

int
main(int argc, char** argv)
{
    tim::timemory_init(argc, argv);
    tim::settings::destructor_report() = true;
    tim::settings::width() = 12;

    long nfib = (argc > 1) ? atol(argv[1]) : 40;
    int  nitr = (argc > 2) ? atoi(argv[2]) : 10;

    // initially, provide all necessary info to compute
    pair_list_t::get_initializer() = [&](pair_list_t& pl) {
        pl.initialize<wall_clock, cpu_clock, peak_rss, assembled_cpu_util>();
    };

    // initially, don't provide system_clock
    triplet_list_t::get_initializer() = [&](triplet_list_t& tl) {
        tl.initialize<wall_clock, user_clock, assembled_cpu_util>();
    };

    for(int i = 0; i < nitr; ++i)
    {
        // should always report assembled_cpu_util
        TIMEMORY_BLANK_MARKER(pair_tuple_t, "pair_tuple     ");
        long ans = fib(nfib);

        // should report for first half of iterations
        TIMEMORY_BLANK_MARKER(pair_list_t, "pair_list", i, "     ");
        ans += fib(nfib + (i % 3));

        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        {
            // should always report assembled_cpu_util
            TIMEMORY_BLANK_MARKER(triplet_tuple_t, "triplet_tuple/", i);
            ans += fib(nfib - 1);
        }

        {
            // should report for second half of iterations
            TIMEMORY_BLANK_MARKER(triplet_list_t, "triplet_list/", i, " ");
            ans += fib(nfib - 1);

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        printf("\nAnswer = %li\n\n", ans);
        if(i == (nitr - 1) / 2)
        {
            // add system_clock to initialization
            triplet_list_t::get_initializer() = [&](triplet_list_t& tl) {
                tl.initialize<wall_clock, user_clock, system_clock, assembled_cpu_util>();
            };

            // remove assembled_cpu_util from initialization
            pair_list_t::get_initializer() = [&](pair_list_t& pl) {
                pl.initialize<wall_clock, cpu_clock, peak_rss>();
            };
        }
    }

    tim::timemory_finalize();
    return EXIT_SUCCESS;
}

//--------------------------------------------------------------------------------------//
