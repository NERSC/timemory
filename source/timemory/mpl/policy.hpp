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

/** \file policy.hpp
 * \headerfile policy.hpp "timemory/mpl/policy.hpp"
 * Provides the template meta-programming policy types
 *
 */

#pragma once

#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/runtime/configure.hpp"
#include "timemory/runtime/enumerate.hpp"

namespace tim
{
namespace policy
{
//======================================================================================//

template <typename _Comp, typename _Tp>
struct record_statistics
{
    using type            = _Tp;
    using this_type       = record_statistics<_Comp, type>;
    using policy_type     = this_type;
    using statistics_type = statistics<type>;

    static void apply(statistics<type>&, const _Comp&);
    static void apply(type&, const _Comp&) {}
};

//--------------------------------------------------------------------------------------//
//
template <typename _Comp>
struct record_statistics<_Comp, void>
{
    using type            = void;
    using this_type       = record_statistics<_Comp, type>;
    using policy_type     = this_type;
    using statistics_type = statistics<type>;

    template <typename... _Args>
    static void apply(_Args&&...)
    {}
};

//--------------------------------------------------------------------------------------//
//
template <typename _Comp>
struct record_statistics<_Comp, std::tuple<>>
{
    using type            = std::tuple<>;
    using this_type       = record_statistics<_Comp, type>;
    using policy_type     = this_type;
    using statistics_type = statistics<type>;

    template <typename... _Args>
    static void apply(_Args&&...)
    {}
};

//======================================================================================//

template <typename _Tp>
struct instance_tracker<_Tp, true>
{
public:
    using type                           = _Tp;
    using int_type                       = int64_t;
    static constexpr bool thread_support = true;

    instance_tracker()                        = default;
    ~instance_tracker()                       = default;
    instance_tracker(const instance_tracker&) = default;
    instance_tracker(instance_tracker&&)      = default;
    instance_tracker& operator=(const instance_tracker&) = default;
    instance_tracker& operator=(instance_tracker&&) = default;

public:
    //----------------------------------------------------------------------------------//
    //
    static int_type get_started_count() { return get_started().load(); }

    //----------------------------------------------------------------------------------//
    //
    static int_type get_thread_started_count() { return get_thread_started(); }

protected:
    //----------------------------------------------------------------------------------//
    //
    static std::atomic<int_type>& get_started()
    {
        static std::atomic<int_type> _instance(0);
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //
    static int_type& get_thread_started()
    {
        static thread_local int_type _instance = 0;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //
    void start()
    {
        m_tot = get_started()++;
        m_thr = get_thread_started()++;
    }

    //----------------------------------------------------------------------------------//
    //
    void stop()
    {
        m_tot = --get_started();
        m_thr = --get_thread_started();
    }

protected:
    int_type m_tot = 0;
    int_type m_thr = 0;
};

//======================================================================================//

template <typename _Tp>
struct instance_tracker<_Tp, false>
{
public:
    using type                           = _Tp;
    using int_type                       = int64_t;
    static constexpr bool thread_support = false;

    instance_tracker()                        = default;
    ~instance_tracker()                       = default;
    instance_tracker(const instance_tracker&) = default;
    instance_tracker(instance_tracker&&)      = default;
    instance_tracker& operator=(const instance_tracker&) = default;
    instance_tracker& operator=(instance_tracker&&) = default;

public:
    //----------------------------------------------------------------------------------//
    //
    static int_type get_started_count() { return get_started().load(); }

protected:
    //----------------------------------------------------------------------------------//
    //
    static std::atomic<int_type>& get_started()
    {
        static std::atomic<int_type> _instance(0);
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //
    void start() { m_tot = get_started()++; }

    //----------------------------------------------------------------------------------//
    //
    void stop() { m_tot = --get_started(); }

protected:
    int_type m_tot = 0;
};

//======================================================================================//

template <typename T, typename Toolset>
struct omp_tools
{
    using type               = Toolset;
    using api_type           = T;
    using function_type      = std::function<void()>;
    using user_ompt_bundle_t = component::user_ompt_bundle;

    //----------------------------------------------------------------------------------//
    //  the default initalizer for OpenMP tools when user_ompt_bundle is included
    //
    template <typename Bundle = user_ompt_bundle_t, typename Tuple = Toolset,
              enable_if_t<(is_one_of<Bundle, Tuple>::value), int> = 0>
    static function_type& get_initializer()
    {
        static function_type _instance = []() {
            std::string components = "wall_clock";
            auto        env_var    = tim::get_env("TIMEMORY_OMPT_COMPONENTS", components);
            std::transform(
                env_var.begin(), env_var.end(), env_var.begin(),
                [](unsigned char c) -> unsigned char { return std::tolower(c); });
            ::tim::configure<Bundle>(enumerate_components(delimit(env_var)));
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //  this functin calls the initializer for the
    //
    template <typename Bundle = user_ompt_bundle_t, typename Tuple = Toolset,
              enable_if_t<!(is_one_of<Bundle, Tuple>::value), int> = 0>
    static function_type& get_initializer()
    {
        static function_type _instance = []() {};
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    //  this functin calls the initializer for the
    //
    static void configure() { get_initializer()(); }
};

//======================================================================================//

}  // namespace policy
}  // namespace tim
