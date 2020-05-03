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

#pragma once

#define TIMEM_DEBUG
#define TIMEMORY_DISABLE_BANNER
#define TIMEMORY_DISABLE_PROPERTIES
#define TIMEMORY_DISABLE_COMPONENT_STORAGE_INIT

#include "timemory/components/rusage/types.hpp"

#if defined(_MACOS)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::read_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::written_bytes, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::virtual_memory, false_type)
#endif

#undef TIMEMORY_DISABLE_PROPERTIES

#include "timemory/sampling/sampler.hpp"
#include "timemory/timemory.hpp"

// C includes
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>

#if defined(TIMEMORY_USE_LIBEXPLAIN)
#    include <libexplain/execvp.h>
#endif

#if defined(_UNIX)
#    include <unistd.h>
extern "C"
{
    extern char** environ;
}
#endif

// C++ includes
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

template <typename Tp>
using vector_t = std::vector<Tp>;
using string_t = std::string;

template <typename Tp>
using vector_t = std::vector<Tp>;
using string_t = std::string;

using namespace tim::component;

//--------------------------------------------------------------------------------------//

#define TIMEM_SIGNAL SIGALRM
#define TIMEM_ITIMER ITIMER_REAL

//--------------------------------------------------------------------------------------//
// create a custom component tuple printer
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
namespace trait
{
//
//--------------------------------------------------------------------------------------//
//
template <>
struct custom_label_printing<papi_array_t> : true_type
{};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace trait
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct custom_print
{
    using value_type = typename Tp::value_type;
    using base_type  = component::base<Tp, value_type>;

    custom_print(std::size_t N, std::size_t /*_Ntot*/, base_type& obj, std::ostream& os,
                 bool /*endline*/)
    {
        std::stringstream ss;
        if(N == 0)
            ss << std::endl;
        ss << "    " << obj << std::endl;
        os << ss.str();
    }
};
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, bool _Sample = ::tim::trait::sampler<Tp>::value>
struct timem_sample;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct timem_sample<Tp, true>
{
    explicit timem_sample(Tp& obj) { obj.measure(); }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct timem_sample<Tp, false>
{
    explicit timem_sample(Tp&) {}
};
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_PAPI)
//
template <>
struct sample<component::papi_array_t>
{
    using EmptyT                 = std::tuple<>;
    using type                   = papi_array_t;
    using value_type             = typename type::value_type;
    using base_type              = typename type::base_type;
    using this_type              = sample<type>;
    static constexpr bool enable = trait::sampler<type>::value;
    using data_type = conditional_t<enable, decltype(std::declval<type>().get()), EmptyT>;

    TIMEMORY_DEFAULT_OBJECT(sample)

    template <typename Up, typename... Args,
              enable_if_t<(std::is_same<Up, this_type>::value), int> = 0>
    explicit sample(base_type& obj, Up, Args&&...)
    {
        obj.value        = type::record();
        obj.is_transient = false;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct start<component::papi_array_t>
{
    using type       = papi_array_t;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    template <typename... Args>
    explicit start(base_type&, Args&&...)
    {
        type::configure();
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct stop<component::papi_array_t>
{
    using type       = papi_array_t;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;

    template <typename... Args>
    explicit stop(base_type&, Args&&...)
    {}
};
//
#endif
//
//--------------------------------------------------------------------------------------//
//
template <>
struct base_printer<component::read_bytes>
{
    using type       = component::read_bytes;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;
    using widths_t   = std::vector<int64_t>;

    template <typename Up                                        = value_type,
              enable_if_t<!(std::is_same<Up, void>::value), int> = 0>
    explicit base_printer(std::ostream& _os, const type& _obj)
    {
        std::stringstream ss, ssv, ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = _obj.get_display_unit();
        auto              _val   = _obj.get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << " read\n    " << ssr.str() << " read";
        _os << ss.str();
    }

    template <typename Up                                       = value_type,
              enable_if_t<(std::is_same<Up, void>::value), int> = 0>
    explicit base_printer(std::ostream&, const type&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct base_printer<component::written_bytes>
{
    using type       = component::written_bytes;
    using value_type = typename type::value_type;
    using base_type  = typename type::base_type;
    using widths_t   = std::vector<int64_t>;

    template <typename Up                                        = value_type,
              enable_if_t<!(std::is_same<Up, void>::value), int> = 0>
    explicit base_printer(std::ostream& _os, const type& _obj)
    {
        std::stringstream ss, ssv, ssr;
        auto              _prec  = base_type::get_precision();
        auto              _width = base_type::get_width();
        auto              _flags = base_type::get_format_flags();
        auto              _disp  = _obj.get_display_unit();
        auto              _val   = _obj.get();

        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << std::get<0>(_val);
        if(!std::get<0>(_disp).empty())
            ssv << " " << std::get<0>(_disp);

        ssr.setf(_flags);
        ssr << std::setw(_width) << std::setprecision(_prec) << std::get<1>(_val);
        if(!std::get<1>(_disp).empty())
            ssr << " " << std::get<1>(_disp);

        ss << ssv.str() << " written\n    " << ssr.str() << " written";
        _os << ss.str();
    }

    template <typename Up                                       = value_type,
              enable_if_t<(std::is_same<Up, void>::value), int> = 0>
    explicit base_printer(std::ostream&, const type&)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
//
//--------------------------------------------------------------------------------------//
//
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_PAPI)
//
template <>
std::string
papi_array_t::get_display() const
{
    if(events.size() == 0)
        return "";
    auto val          = (is_transient) ? accum : value;
    auto _get_display = [&](std::ostream& os, size_type idx) {
        auto     _obj_value = val[idx];
        auto     _evt_type  = events[idx];
        string_t _label     = papi::get_event_info(_evt_type).short_descr;
        string_t _disp      = papi::get_event_info(_evt_type).units;
        auto     _prec      = base_type::get_precision();
        auto     _width     = base_type::get_width();
        auto     _flags     = base_type::get_format_flags();

        std::stringstream ss, ssv, ssi;
        ssv.setf(_flags);
        ssv << std::setw(_width) << std::setprecision(_prec) << _obj_value;
        if(!_disp.empty())
            ssv << " " << _disp;
        if(!_label.empty())
            ssi << " " << _label;
        ss << ssv.str() << ssi.str();
        if(idx > 0)
            os << "    ";
        os << ss.str();
    };

    std::stringstream ss;
    for(size_type i = 0; i < events.size(); ++i)
    {
        _get_display(ss, i);
        if(i + 1 < events.size())
            ss << '\n';
    }

    return ss.str();
}
//
#endif
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
class timem_tuple : public lightweight_tuple<Types...>
{
public:
    using base_type = lightweight_tuple<Types...>;
    using apply_v   = typename base_type::apply_v;
    using impl_type = typename base_type::impl_type;

    template <template <typename> class Op, typename Tuple = impl_type>
    using custom_operation_t =
        typename base_type::template custom_operation<Op, Tuple>::type;

    template <typename... T>
    struct opsample;

    template <template <typename...> class Tuple, typename... T>
    struct opsample<Tuple<T...>>
    {
        using type = Tuple<operation::timem_sample<T, trait::sampler<T>::value>...>;
    };

    template <typename T>
    using opsample_t = typename opsample<T>::type;

public:
    timem_tuple()
    : base_type()
    {}

    explicit timem_tuple(const string_t& key)
    : base_type(key)
    {}

    ~timem_tuple() {}

    //
    //----------------------------------------------------------------------------------//
    //
    void start() { base_type::start(); }
    void stop() { base_type::stop(); }
    void reset() { base_type::reset(); }
    auto get() { return base_type::get(); }
    auto get_labeled() { return base_type::get_labeled(); }
    void sample()
    {
        base_type::sample();
        apply<void>::access<opsample_t<impl_type>>(this->m_data);
    }

    //
    //----------------------------------------------------------------------------------//
    //
    friend std::ostream& operator<<(std::ostream& os, const timem_tuple<Types...>& obj)
    {
        std::stringstream ssp;
        std::stringstream ssd;
        auto&&            data  = obj.m_data;
        auto&&            key   = obj.key();
        auto&&            width = obj.output_width();

        using print_t = custom_operation_t<custom_print, impl_type>;
        apply<void>::access_with_indices<print_t>(data, std::ref(ssd), false);

        ssp << std::setw(width) << std::left << key;
        os << ssp.str() << ssd.str();

        return os;
    }

protected:
    using base_type::m_data;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
using timem_tuple_t = convert_t<available_tuple<type_list<Types...>>, timem_tuple<>>;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEM_BUNDLER)
#    define TIMEM_BUNDLER                                                                \
        tim::timem_tuple_t<wall_clock, user_clock, system_clock, cpu_clock, cpu_util,    \
                           peak_rss, page_rss, virtual_memory, num_major_page_faults,    \
                           num_minor_page_faults, priority_context_switch,               \
                           voluntary_context_switch, read_bytes, written_bytes,          \
                           papi_array_t>
#endif

using comp_tuple_t = TIMEM_BUNDLER;
using sampler_t    = tim::sampling::sampler<TIMEM_BUNDLER, 1>;

//--------------------------------------------------------------------------------------//

inline sampler_t*&
get_sampler()
{
    static sampler_t* _instance = nullptr;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline comp_tuple_t*
get_measure()
{
    return get_sampler()->get_samplers().at(0)->get_last();
}

//--------------------------------------------------------------------------------------//

struct timem_config
{
    bool        use_shell    = tim::get_env("TIMEM_USE_SHELL", false);
    bool        debug        = tim::get_env("TIMEM_DEBUG", false);
    int         verbose      = tim::get_env("TIMEM_VERBOSE", 0);
    std::string shell        = tim::get_env<std::string>("SHELL", getusershell());
    std::string shell_flags  = tim::get_env<std::string>("TIMEM_USE_SHELL_FLAGS", "-i");
    double      sample_freq  = tim::get_env<double>("TIMEM_SAMPLE_FREQ", 2.0);
    double      sample_delay = tim::get_env<double>("TIMEM_SAMPLE_DELAY", 0.001);
    pid_t       master_pid   = getpid();
    pid_t       worker_pid   = getpid();
    std::string command      = "";
};

//--------------------------------------------------------------------------------------//

inline timem_config&
get_config()
{
    static timem_config _instance;
    return _instance;
}

//--------------------------------------------------------------------------------------//

inline bool&
use_shell()
{
    return get_config().use_shell;
}

//--------------------------------------------------------------------------------------//

inline bool&
debug()
{
    return get_config().debug;
}

//--------------------------------------------------------------------------------------//

inline int&
verbose()
{
    return get_config().verbose;
}

//--------------------------------------------------------------------------------------//

inline std::string&
command()
{
    return get_config().command;
}

//--------------------------------------------------------------------------------------//

inline pid_t&
master_pid()
{
    return get_config().master_pid;
}

//--------------------------------------------------------------------------------------//

inline pid_t&
worker_pid()
{
    return get_config().worker_pid;
}

//--------------------------------------------------------------------------------------//

inline void
explain(int ret, const char* pathname, char** argv)
{
    if(ret < 0)
    {
#if defined(TIMEMORY_USE_LIBEXPLAIN)
        fprintf(stderr, "%s\n", explain_execvp(pathname, argv));
#else
        fprintf(stderr, "Return code: %i : %s\n", ret, pathname);
        int n = 0;
        std::cerr << "Command: ";
        while(argv[n] != nullptr)
            std::cerr << argv[n++] << " ";
        std::cerr << std::endl;
#endif
    }
    else if(debug() || verbose() > 0)
    {
        int n = 0;
        std::cerr << "Command: ";
        while(argv[n] != nullptr)
            std::cerr << argv[n++] << " ";
        std::cerr << std::endl;
    }
}

//--------------------------------------------------------------------------------------//
