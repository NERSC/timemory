//  MIT License
//
//  Copyright (c) 2019, The Regents of the University of California,
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

#pragma once

#include <cstdint>
#include <type_traits>

//======================================================================================//
//
namespace tim
{
//======================================================================================//
// generic
//
namespace component
{
// define this short-hand from C++14 for C++11
template <bool B, typename T>
using enable_if_t = typename std::enable_if<B, T>::type;

}  // component

//======================================================================================//
//  components that provide implementations (i.e. HOW to record a component)
//
namespace component
{
// generic static polymorphic base class
template <typename _Tp, typename value_type = int64_t, typename... _Policies>
struct base;

// timing
struct real_clock;
struct system_clock;
struct user_clock;
struct cpu_clock;
struct monotonic_clock;
struct monotonic_raw_clock;
struct thread_cpu_clock;
struct process_cpu_clock;
struct cpu_util;
struct process_cpu_util;
struct thread_cpu_util;

// resource usage
struct peak_rss;
struct current_rss;
struct stack_rss;
struct data_rss;
struct num_swap;
struct num_io_in;
struct num_io_out;
struct num_minor_page_faults;
struct num_major_page_faults;
struct num_msg_sent;
struct num_msg_recv;
struct num_signals;
struct voluntary_context_switch;
struct priority_context_switch;

// cuda
struct cuda_event;

// papi
template <int EventSet, int... EventTypes>
struct papi_tuple;

template <int EventSet, std::size_t NumEvent>
struct papi_array;

template <int... EventTypes>
struct cpu_roofline;

}  // component

//======================================================================================//
//  components that provide the invocation (i.e. WHAT the components need to do)
//
namespace component
{
// operators
template <typename _Tp>
struct construct;

template <typename _Tp>
struct live_count;

template <typename _Tp>
struct set_prefix;

template <typename _Tp>
struct insert_node;

template <typename _Tp>
struct pop_node;

template <typename _Tp>
struct record;

template <typename _Tp>
struct reset;

template <typename _Tp>
struct measure;

template <typename _Tp>
struct start;

template <typename _Tp>
struct stop;

template <typename _Tp>
struct conditional_start;

template <typename _Tp>
struct conditional_stop;

template <typename _Tp>
struct minus;

template <typename _Tp>
struct plus;

template <typename _Tp>
struct multiply;

template <typename _Tp>
struct divide;

template <typename _Tp>
struct print;

template <typename _Tp>
struct print_storage;

template <typename _Tp, typename _Archive>
struct serialization;

template <typename _Tp, typename _Op>
struct pointer_operator;

template <typename _Tp>
struct pointer_deleter;

}  // component

//--------------------------------------------------------------------------------------//
//
//  Language specification
//
//--------------------------------------------------------------------------------------//

class language
{
public:
    enum class type : int64_t
    {
        C       = 1,
        CXX     = 2,
        PYTHON  = 3,
        UNKNOWN = 4
    };

    friend std::ostream& operator<<(std::ostream& os, const language& lang)
    {
        os << language::as_string(lang);
        return os;
    }

    static std::string as_string(const language& _lang)
    {
        switch(_lang.m_type)
        {
            case type::C: return "[_c_]";
            case type::CXX: return "[cxx]";
            case type::PYTHON: return "[pyc]";
            case type::UNKNOWN:
            default: return _lang.m_descript;
        }
    }

    operator int64_t() const { return static_cast<int64_t>(m_type); }
    operator uint64_t() const { return static_cast<uint64_t>(m_type); }

    constexpr explicit language(const type& _type)
    : m_type(_type)
    {}

    language(const char* m_lang)
    : m_type(type::UNKNOWN)
    , m_descript(m_lang)
    {}

    constexpr static language c() { return language(type::C); }
    constexpr static language cxx() { return language(type::CXX); }
    constexpr static language pyc() { return language(type::PYTHON); }

private:
    type              m_type;
    const char* const m_descript = nullptr;
};

//--------------------------------------------------------------------------------------//
//
//  Some common types
//
//--------------------------------------------------------------------------------------//

template <typename... Types>
class component_tuple;

//--------------------------------------------------------------------------------------//
//  category configurations
//
using rusage_components_t = component_tuple<
    component::current_rss, component::peak_rss, component::stack_rss,
    component::data_rss, component::num_swap, component::num_io_in, component::num_io_out,
    component::num_minor_page_faults, component::num_major_page_faults,
    component::num_msg_sent, component::num_msg_recv, component::num_signals,
    component::voluntary_context_switch, component::priority_context_switch>;

using timing_components_t =
    component_tuple<component::real_clock, component::system_clock, component::user_clock,
                    component::cpu_clock, component::monotonic_clock,
                    component::monotonic_raw_clock, component::thread_cpu_clock,
                    component::process_cpu_clock, component::cpu_util,
                    component::thread_cpu_util, component::process_cpu_util>;

//--------------------------------------------------------------------------------------//
//  standard configurations
//
using standard_rusage_t =
    component_tuple<component::current_rss, component::peak_rss, component::num_io_in,
                    component::num_io_out, component::num_minor_page_faults,
                    component::num_major_page_faults, component::priority_context_switch,
                    component::voluntary_context_switch>;

using standard_timing_t =
    component_tuple<component::real_clock, component::user_clock, component::system_clock,
                    component::cpu_clock, component::cpu_util>;

}  // tim

//======================================================================================//
