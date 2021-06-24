//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
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

/**
 * \file timemory/components/gotcha/components.hpp
 * \brief Implementation of the gotcha component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/gotcha/backends.hpp"
#include "timemory/components/gotcha/types.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/variadic/types.hpp"

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//======================================================================================//
//
// template params:
//      Nt             ==  max number of GOTCHA wrappers
//      BundleT     ==  {auto,component}_{tuple,list,bundle}
//      DiffT ==  extra param to differentiate when Nt and BundleT are same
//
//  TODO: filter any gotcha components out of BundleT
//
/// \struct tim::component::gotcha
/// \tparam Nt Max number of functions which will wrapped by this component
/// \tparam BundleT Component bundle to wrap around the function(s)
/// \tparam DiffT Differentiator type to distinguish different sets of wrappers with
/// identical values of `Nt` and `BundleT` (or provide function call operator if replacing
/// functions instead of wrapping functions)
///
/// \brief The gotcha component rewrites the global offset table such that calling the
/// wrapped function actually invokes either a function which is wrapped by timemory
/// instrumentation or is replaced by a timemory component with an function call operator
/// (`operator()`) whose return value and arguments exactly match the original function.
/// This component is only available on Linux and can only by applied to external,
/// dynamically-linked functions (i.e. functions defined in a shared library).
/// If the `BundleT` template parameter is a non-empty component bundle, this component
/// will surround the original function call with:
///
/// \code{.cpp}
/// bundle_type _obj{ "<NAME-OF-ORIGINAL-FUNCTION>" };
/// _obj.construct(_args...);
/// _obj.start();
/// _obj.audit("<NAME-OF-ORIGINAL-FUNCTION>", _args...);
///
/// Ret _ret = <CALL-ORIGINAL-FUNCTION>
///
/// _obj.audit("<NAME-OF-ORIGINAL-FUNCTION>", _ret);
/// _obj.stop();
/// \endcode
///
/// If the `BundleT` template parameter is an empty variadic class, e.g. `std::tuple<>`,
/// `tim::component_tuple<>`, etc., and the `DiffT` template parameter is a timemory
/// component, the assumption is that the `DiffT` component has a function call operator
/// which should replace the original function call, e.g. `void* malloc(size_t)` can be
/// replaced with a component with `void* operator()(size_t)`, e.g.:
///
/// \code{.cpp}
/// // replace 'double exp(double)'
/// struct exp_replace : base<exp_replace, void>
/// {
///     double operator()(double value)
///     {
///         float result = expf(static_cast<float>(value));
///         return static_cast<double>(result);
///     }
/// };
/// \endcode
///
/// Example usage:
///
/// \code{.cpp}
/// #include <timemory/timemory.hpp>
///
/// #include <cassert>
/// #include <cmath>
/// #include <tuple>
///
/// using empty_tuple_t = std::tuple<>;
/// using base_bundle_t = tim::component_tuple<wall_clock, cpu_clock>;
/// using gotcha_wrap_t = tim::component::gotcha<2, base_bundle_t, void>;
/// using gotcha_repl_t = tim::component::gotcha<2, empty_tuple_t, exp_replace>;
/// using impl_bundle_t = tim::mpl::append_type_t<base_bundle_t,
///                                               tim::type_list<gotcha_wrap_t,
///                                                              gotcha_repl_t>>;
///
/// void init_wrappers()
/// {
///     // wraps the sin and cos math functions
///     gotcha_wrap_t::get_initializer() = []()
///     {
///         TIMEMORY_C_GOTCHA(gotcha_wrap_t, 0, sin);   // index 0 replaces sin
///         TIMEMORY_C_GOTCHA(gotcha_wrap_t, 1, cos);   // index 1 replace cos
///     };
///
///     // replaces the 'exp' function which may be 'exp' in symbols table
///     // or '__exp_finite' in symbols table (use `nm <bindary>` to determine)
///     gotcha_repl_t::get_initializer() = []()
///     {
///         TIMEMORY_C_GOTCHA(gotcha_repl_t, 0, exp);
///         TIMEMORY_DERIVED_GOTCHA(gotcha_repl_t, 1, exp, "__exp_finite");
///     };
/// }
///
/// // the following is useful to avoid having to call 'init_wrappers()' explicitly:
/// // use comma operator to call 'init_wrappers' and return true
/// static auto called_init_at_load = (init_wrappers(), true);
///
/// int main()
/// {
///     assert(called_init_at_load == true);
///
///     double angle = 45.0 * (M_PI / 180.0);
///
///     impl_bundle_t _obj{ "main" };
///
///     // gotcha wrappers not activated yet
///     printf("cos(%f) = %f\n", angle, cos(angle));
///     printf("sin(%f) = %f\n", angle, sin(angle));
///     printf("exp(%f) = %f\n", angle, exp(angle));
///
///     // gotcha wrappers are reference counted according to start/stop
///     _obj.start();
///
///     printf("cos(%f) = %f\n", angle, cos(angle));
///     printf("sin(%f) = %f\n", angle, sin(angle));
///     printf("exp(%f) = %f\n", angle, exp(angle));
///
///     _obj.stop();
///
///     // gotcha wrappers will be deactivated
///     printf("cos(%f) = %f\n", angle, cos(angle));
///     printf("sin(%f) = %f\n", angle, sin(angle));
///     printf("exp(%f) = %f\n", angle, exp(angle));
///
///     return 0;
/// }
/// \endcode
template <size_t Nt, typename BundleT, typename DiffT>
struct gotcha
: public base<gotcha<Nt, BundleT, DiffT>, void>
, public concepts::external_function_wrapper
{
    static_assert(concepts::has_gotcha<BundleT>::value == false,
                  "Error! {auto,component}_{list,tuple,bundle} in a GOTCHA specification "
                  "cannot include another gotcha_component");

    using value_type   = void;
    using this_type    = gotcha<Nt, BundleT, DiffT>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using tuple_type   = concepts::tuple_type_t<BundleT>;
    using bundle_type  = concepts::component_type_t<BundleT>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

    template <typename Tp>
    using array_t = std::array<Tp, Nt>;

    using binding_t     = backend::gotcha::binding_t;
    using wrappee_t     = backend::gotcha::wrappee_t;
    using wrappid_t     = backend::gotcha::string_t;
    using error_t       = backend::gotcha::error_t;
    using constructor_t = std::function<void()>;
    using destructor_t  = std::function<void()>;
    using atomic_bool_t = std::atomic<bool>;

    using select_list_t = std::set<std::string>;

    using config_t          = void;
    using get_initializer_t = std::function<config_t()>;
    using get_select_list_t = std::function<select_list_t()>;

    static constexpr size_t components_size = mpl::get_tuple_size<tuple_type>::value;
    static constexpr bool   differ_is_component =
        (is_one_of<DiffT, tuple_type>::value ||
         (components_size == 0 && concepts::is_component<DiffT>::value));
    // backwards-compat
    static constexpr bool differentiator_is_component = differ_is_component;

    using operator_type =
        typename std::conditional<differ_is_component, DiffT, void>::type;

    static std::string label() { return "gotcha"; }
    static std::string description()
    {
        return "Generates GOTCHA wrappers which can be used to wrap or replace "
               "dynamically linked function calls";
    }
    static value_type record() {}

    //----------------------------------------------------------------------------------//

    static get_initializer_t& get_initializer()
    {
        return get_persistent_data().m_initializer;
    }

    //----------------------------------------------------------------------------------//
    /// when a permit list is provided, only these functions are wrapped by GOTCHA
    static get_select_list_t& get_permit_list()
    {
        return get_persistent_data().m_permit_list;
    }

    //----------------------------------------------------------------------------------//
    /// reject listed functions are never wrapped by GOTCHA
    static get_select_list_t& get_reject_list()
    {
        return get_persistent_data().m_reject_list;
    }

    //----------------------------------------------------------------------------------//

    static bool& get_default_ready()
    {
        static bool _instance = false;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// add function names at runtime to suppress wrappers
    static void add_global_suppression(const std::string& func)
    {
        get_suppresses().insert(func);
    }

    //----------------------------------------------------------------------------------//
    /// get an array of whether the wrappers are filled and ready
    static auto get_ready()
    {
        std::array<std::pair<bool, bool>, Nt> _ready;
        for(size_t i = 0; i < Nt; ++i)
            _ready.at(i) = { get_data().at(i).filled, get_data().at(i).ready };
        return _ready;
    }

    //----------------------------------------------------------------------------------//
    /// set filled wrappers to array of ready values
    static auto set_ready(bool val)
    {
        for(size_t i = 0; i < Nt; ++i)
        {
            if(get_data().at(i).filled)
                get_data().at(i).ready = val;
        }
        return get_ready();
    }

    //----------------------------------------------------------------------------------//
    /// set filled wrappers to array of ready values
    static auto set_ready(const std::array<bool, Nt>& values)
    {
        for(size_t i = 0; i < Nt; ++i)
        {
            if(get_data().at(i).filled)
                get_data().at(i).ready = values.at(i);
        }
        return get_ready();
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static bool construct(const std::string& _func, int _priority = 0,
                          const std::string& _tool = "")
    {
        gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());

        init_storage<bundle_type>(0);

        static_assert(N < Nt, "Error! N must be less than Nt!");
        auto& _data = get_data()[N];

        if(!is_permitted<N, Ret, Args...>(_func))
            return false;

        if(_data.debug == nullptr)
            _data.debug = &settings::debug();

        if(!_data.filled)
        {
            auto _label = demangle(_func);
            if(_tool.length() > 0 && _label.find(_tool + "/") != 0)
            {
                _label = _tool + "/" + _label;
                while(_label.find("//") != std::string::npos)
                    _label.erase(_label.find("//"), 1);
            }

            // ensure the hash to string pairing is stored
            storage_type::instance()->add_hash_id(_label);

            _data.filled   = true;
            _data.priority = _priority;
            _data.tool_id  = _label;
            _data.wrap_id  = _func;
            _data.ready    = get_default_ready();

            if(get_suppresses().find(_func) != get_suppresses().end())
            {
                _data.suppression = &gotcha_suppression::get();
                _data.ready       = false;
            }

            _data.constructor = [_func, _priority, _tool]() {
                this_type::construct<N, Ret, Args...>(_func, _priority, _tool);
            };
            _data.destructor = []() { this_type::revert<N>(); };
            _data.binding = std::move(construct_binder<N, Ret, Args...>(_data.wrap_id));
            error_t ret_wrap = backend::gotcha::wrap(_data.binding, _data.tool_id);
            check_error<N>(ret_wrap, "binding");
        }

        if(!_data.is_active)
        {
            _data.is_active = true;
            error_t ret_prio =
                backend::gotcha::set_priority(_data.tool_id, _data.priority);
            check_error<N>(ret_prio, "set priority");
        }

        if(!_data.ready)
            revert<N>();

        return _data.filled;
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static auto configure(const std::string& _func, int _priority = 0,
                          const std::string& _tool = "")
    {
        return construct<N, Ret, Args...>(_func, _priority, _tool);
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static auto configure(const std::vector<std::string>& _funcs, int _priority = 0,
                          const std::string& _tool = "")
    {
        auto itr = _funcs.begin();
        auto ret = false;
        while(!ret && itr != _funcs.end())
        {
            ret = construct<N, Ret, Args...>(*itr, _priority, _tool);
            ++itr;
        }
    }

    //----------------------------------------------------------------------------------//

    template <size_t N>
    static bool revert()
    {
        gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());

        static_assert(N < Nt, "Error! N must be less than Nt!");
        auto& _data = get_data()[N];

        if(_data.filled && _data.is_active)
        {
            _data.is_active = false;

            error_t ret_prio = backend::gotcha::set_priority(_data.tool_id, -1);
            check_error<N>(ret_prio, "get priority");

            if(get_suppresses().find(_data.tool_id) != get_suppresses().end())
            {
                _data.ready = false;
            }
            else
            {
                _data.ready = get_default_ready();
            }
        }

        return _data.filled;
    }

    //----------------------------------------------------------------------------------//

    static bool& is_configured() { return get_persistent_data().m_is_configured; }

    //----------------------------------------------------------------------------------//

    static std::mutex& get_mutex() { return get_persistent_data().m_mutex; }

    //----------------------------------------------------------------------------------//

    static auto get_info()
    {
        std::array<size_t, 5> _info{};
        _info.fill(0);
        for(auto& itr : get_data())
        {
            _info.at(0) += (itr.ready) ? 1 : 0;
            _info.at(1) += (itr.filled) ? 1 : 0;
            _info.at(2) += (itr.is_active) ? 1 : 0;
            _info.at(3) += (itr.is_finalized) ? 1 : 0;
            _info.at(4) += (itr.suppression && !(*itr.suppression)) ? 1 : 0;
        }
        return _info;
    }

    //----------------------------------------------------------------------------------//

    static void configure()
    {
        std::unique_lock<std::mutex> lk(get_mutex(), std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();

        if(!is_configured())
        {
            is_configured() = true;
            lk.unlock();
            auto& _init = get_initializer();
            _init();
        }
    }

    //----------------------------------------------------------------------------------//

    static void enable() { configure(); }

    //----------------------------------------------------------------------------------//

    static void disable()
    {
        std::unique_lock<std::mutex> lk(get_mutex(), std::defer_lock);
        if(!lk.owns_lock())
            lk.lock();

        if(is_configured())
        {
            is_configured() = false;
            lk.unlock();
            for(auto& itr : get_data())
            {
                if(!itr.is_finalized)
                {
                    itr.is_finalized = true;
                    itr.destructor();
                }
            }
        }
    }

    //----------------------------------------------------------------------------------//

    static void global_finalize()
    {
        while(get_started() > 0)
            --get_started();
        while(get_thread_started() > 0)
            --get_thread_started();
        disable();
    }

    static void thread_init()
    {
        auto& _data = get_data();
        for(size_t i = 0; i < Nt; ++i)
            _data[i].ready = (_data[i].filled && get_default_ready());
    }

public:
    //----------------------------------------------------------------------------------//

    void start()
    {
        if(storage_type::is_finalizing())
            return;

        auto _n = get_started()++;
        auto _t = get_thread_started()++;

#if defined(DEBUG)
        if(settings::debug())
        {
            static std::atomic<int64_t> _tcount(0);
            static thread_local int64_t _tid = _tcount++;
            std::stringstream           ss;
            ss << "[T" << _tid << "]> n = " << _n << ", t = " << _t << "...\n";
            std::cout << ss.str() << std::flush;
        }
#endif

        // this ensures that if started from multiple threads, all threads synchronize
        // before
        if(_t == 0 && !is_configured())
            configure();

        if(_n == 0)
        {
            configure();
            for(auto& itr : get_data())
            {
                if(!itr.is_finalized)
                    itr.constructor();
            }
        }

        if(_t == 0)
        {
            auto& _data = get_data();
            for(size_t i = 0; i < Nt; ++i)
                _data[i].ready = _data[i].filled;
        }
    }

    void stop()
    {
        auto _n = --get_started();
        auto _t = --get_thread_started();

#if defined(DEBUG)
        if(settings::debug())
        {
            static std::atomic<int64_t> _tcount(0);
            static thread_local int64_t _tid = _tcount++;
            std::stringstream           ss;
            ss << "[T" << _tid << "]> n = " << _n << ", t = " << _t << "...\n";
            std::cout << ss.str() << std::flush;
        }
#endif

        if(_t == 0)
        {
            auto& _data = get_data();
            for(size_t i = 0; i < Nt; ++i)
                _data[i].ready = false;
        }

        if(_n == 0)
        {
            for(auto& itr : get_data())
            {
                if(!itr.is_finalized)
                    itr.destructor();
            }
        }
    }

public:
    //----------------------------------------------------------------------------------//
    //  secondary method
    //
    template <size_t N, typename Ret, typename... Args>
    struct instrument
    {
        static void generate(const std::string& _func, const std::string& _tool = "",
                             int _priority = 0)
        {
            this_type::configure<N, Ret, Args...>(_func, _priority, _tool);
        }
    };

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    struct instrument<N, Ret, std::tuple<Args...>> : instrument<N, Ret, Args...>
    {};

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static void gotcha_factory(const std::string& _func, const std::string& _tool = "",
                               int _priority = 0)
    {
        instrument<N, Ret, Args...>::generate(_func, _tool, _priority);
    }

    //----------------------------------------------------------------------------------//

private:
    //----------------------------------------------------------------------------------//
    //
    struct persistent_data
    {
        persistent_data()
        {
            for(auto& itr : m_data)
                itr.ready = get_default_ready();
        }

        ~persistent_data() = default;
        TIMEMORY_DELETE_COPY_MOVE_OBJECT(persistent_data)

        bool                  m_is_configured = false;
        std::atomic<int64_t>  m_started{ 0 };
        array_t<gotcha_data>  m_data;
        std::mutex            m_mutex;
        std::set<std::string> m_suppress    = { "malloc", "calloc", "free" };
        get_initializer_t     m_initializer = []() {
            for(const auto& itr : get_data())
                itr.constructor();
        };
        get_select_list_t m_reject_list = []() { return select_list_t{}; };
        get_select_list_t m_permit_list = []() { return select_list_t{}; };
    };

    //----------------------------------------------------------------------------------//
    //
    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// \fn array_t<gotcha_data>& get_data()
    /// \brief Gotcha wrapper data
    static array_t<gotcha_data>& get_data() { return get_persistent_data().m_data; }

    //----------------------------------------------------------------------------------//
    /// \fn std::atomic<int64_t>& get_started()
    /// \brief Global counter of active gotchas started
    static std::atomic<int64_t>& get_started() { return get_persistent_data().m_started; }

    //----------------------------------------------------------------------------------//
    /// \fn int64_t& get_thread_started()
    /// \brief Thread-local counter of activate gotchas
    static int64_t& get_thread_started()
    {
        static thread_local int64_t _instance = 0;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// \fn std::set<std::string>& get_suppresses()
    /// \brief global suppression when being used
    static std::set<std::string>& get_suppresses()
    {
        return get_persistent_data().m_suppress;
    }

    //----------------------------------------------------------------------------------//
    /// \brief bool is_permitted()
    /// Check the permit list and reject list for whether the component is permitted
    /// to be wrapped.
    template <size_t N, typename Ret, typename... Args>
    static bool is_permitted(const std::string& _func)
    {
        // if instruments are being used, we need to restrict using GOTCHAs around
        // certain MPI functions which can cause deadlocks. However, allow
        // these GOTCHA components which serve as function replacements to
        // wrap these functions
        if(std::is_same<operator_type, void>::value &&
           (_func.find("MPI_") != std::string::npos ||
            _func.find("mpi_") != std::string::npos))
        {
            static auto mpi_reject_list = { "MPI_Init",           "MPI_Finalize",
                                            "MPI_Pcontrol",       "MPI_Init_thread",
                                            "MPI_Initialized",    "MPI_Comm_rank",
                                            "MPI_Comm_size",      "MPI_T_init_thread",
                                            "MPI_Comm_split",     "MPI_Abort",
                                            "MPI_Comm_split_type" };

            auto tofortran = [](std::string _fort) {
                for(auto& itr : _fort)
                    itr = tolower(itr);
                if(_fort[_fort.length() - 1] != '_')
                    _fort += "_";
                return _fort;
            };

            // if function matches a reject_listed entry, do not construct wrapper
            for(const auto& itr : mpi_reject_list)
            {
                if(_func == itr || _func == tofortran(itr))
                {
                    if(settings::debug())
                    {
                        printf("[gotcha]> Skipping gotcha binding for %s...\n",
                               _func.c_str());
                    }
                    return false;
                }
            }
        }

        const select_list_t& _permit_list = get_permit_list()();
        const select_list_t& _reject_list = get_reject_list()();

        // if function matches a reject_listed entry, do not construct wrapper
        if(_reject_list.count(_func) > 0)
        {
            if(settings::debug())
            {
                printf(
                    "[gotcha]> GOTCHA binding for function '%s' is in reject list...\n",
                    _func.c_str());
            }
            return false;
        }

        // if a permit_list was provided, then do not construct wrapper if not in permit
        // list
        if(!_permit_list.empty())
        {
            if(_permit_list.count(_func) == 0)
            {
                if(settings::debug())
                {
                    printf("[gotcha]> GOTCHA binding for function '%s' is not in permit "
                           "list...\n",
                           _func.c_str());
                }
                return false;
            }
        }

        return true;
    }

    //----------------------------------------------------------------------------------//

    template <size_t N>
    static void check_error(error_t _ret, const std::string& _prefix)
    {
        if(_ret != GOTCHA_SUCCESS)
        {
            auto&             _data = get_data()[N];
            std::stringstream msg;
            msg << _prefix << " at index '" << N << "' for function '" << _data.wrap_id
                << "' returned error code " << static_cast<int>(_ret) << ": "
                << backend::gotcha::get_error(_ret) << "\n";
            std::cerr << msg.str() << std::endl;
        }
        else if(settings::verbose() > 1 || settings::debug())
        {
#if defined(TIMEMORY_USE_GOTCHA)
            auto&             _data = get_data()[N];
            std::stringstream msg;
            msg << "[gotcha::" << __FUNCTION__ << "]> " << _prefix << " :: "
                << "wrapped: " << _data.wrap_id << ", label: " << _data.tool_id;
            /*
            if((void*) _data.binding != nullptr)
            {
                msg << ", wrapped pointer: " << _data.binding.wrapper_pointer
                    << ", function_handle: " << _data.binding.function_handle
                    << ", name: " << _data.binding.name;
            }
            */
            std::cout << msg.str() << std::endl;
#endif
        }
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args, typename This = this_type,
              typename std::enable_if<(This::components_size != 0), int>::type    = 0,
              typename std::enable_if<!std::is_same<Ret, void>::value, int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[N];
        _data.wrapper = (void*) this_type::wrap<N, Ret, Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args, typename This = this_type,
              typename std::enable_if<(This::components_size != 0), int>::type   = 0,
              typename std::enable_if<std::is_same<Ret, void>::value, int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[N];
        _data.wrapper = (void*) this_type::wrap_void<N, Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args, typename This = this_type,
              typename std::enable_if<This::components_size == 0, int>::type      = 0,
              typename std::enable_if<!std::is_same<Ret, void>::value, int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[N];
        _data.wrapper = (void*) this_type::replace_func<N, Ret, Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args, typename This = this_type,
              typename std::enable_if<This::components_size == 0, int>::type     = 0,
              typename std::enable_if<std::is_same<Ret, void>::value, int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[N];
        _data.wrapper = (void*) this_type::replace_void_func<N, Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<This::differ_is_component, int>       = 0,
              enable_if_t<!std::is_same<Ret, void>::value, int> = 0>
    static Ret invoke(Comp& _comp, Ret (*_func)(Args...), Args&&... _args)
    {
        using Type    = DiffT;
        using Invoker = gotcha_invoker<Type, Ret>;
        Type& _obj    = *_comp.template get<Type>();
        return Invoker::invoke(_obj, _func, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<!This::differ_is_component, int>      = 0,
              enable_if_t<!std::is_same<Ret, void>::value, int> = 0>
    static Ret invoke(Comp&, Ret (*_func)(Args...), Args&&... _args)
    {
        return _func(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<This::differ_is_component, int>      = 0,
              enable_if_t<std::is_same<Ret, void>::value, int> = 0>
    static void invoke(Comp& _comp, Ret (*_func)(Args...), Args&&... _args)
    {
        using Type    = DiffT;
        using Invoker = gotcha_invoker<Type, Ret>;
        Type& _obj    = *_comp.template get<Type>();
        Invoker::invoke(_obj, _func, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<!This::differ_is_component, int>     = 0,
              enable_if_t<std::is_same<Ret, void>::value, int> = 0>
    static void invoke(Comp&, Ret (*_func)(Args...), Args&&... _args)
    {
        _func(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static TIMEMORY_NOINLINE Ret wrap(Args... _args)
    {
        static_assert(N < Nt, "Error! N must be less than Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        auto& _data = get_data()[N];

        // PRINT_HERE("%s", _data.tool_id.c_str());

        static constexpr bool void_operator = std::is_same<operator_type, void>::value;
        static_assert(void_operator, "operator_type should be void!");

        typedef Ret (*func_t)(Args...);
        func_t _orig = (func_t)(gotcha_get_wrappee(_data.wrappee));

        if(_data.is_finalized)
            return (_orig) ? (*_orig)(_args...) : Ret{};

        auto _suppress =
            gotcha_suppression::get() || (_data.suppression && *_data.suppression);
        if(!_data.ready || _suppress)
        {
            static thread_local bool _recursive = false;
            if(!_recursive && _data.debug && *_data.debug)
            {
                _recursive = true;
                auto _tid  = threading::get_id();
                fprintf(stderr,
                        "[T%i][%s]> %s is either not ready (ready=%s) or is globally "
                        "suppressed (suppressed=%s)\n",
                        (int) _tid, __FUNCTION__, _data.tool_id.c_str(),
                        (_data.ready) ? "true" : "false", (_suppress) ? "true" : "false");
                fflush(stderr);
                _recursive = false;
            }
            return (_orig) ? (*_orig)(_args...) : Ret{};
        }

        bool did_data_toggle = false;
        bool did_glob_toggle = false;

        auto toggle_suppress_on = [](bool* _bsuppress, bool& _did) {
            if(_bsuppress && *_bsuppress == false)
            {
                *(_bsuppress) = true;
                _did          = true;
            }
        };

        auto toggle_suppress_off = [](bool* _bsuppress, bool& _did) {
            if(_bsuppress && _did == true && *_bsuppress == true)
            {
                *(_bsuppress) = false;
                _did          = false;
            }
        };

        if(_orig)
        {
            // make sure the function is not recursively entered
            // (important for allocation-based wrappers)
            _data.ready = false;
            toggle_suppress_on(_data.suppression, did_data_toggle);

            // bundle_type is always: component_{tuple,list,bundle}
            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);
            //
            bundle_type _obj{ _data.tool_id };
            _obj.construct(_args...);
            _obj.start();
            _obj.audit(_data, audit::incoming{}, _args...);
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);

            _data.ready = true;
            Ret _ret    = invoke<bundle_type>(_obj, _orig, std::forward<Args>(_args)...);
            _data.ready = false;

            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);
            _obj.audit(_data, audit::outgoing{}, _ret);
            _obj.stop();
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);

            // allow re-entrance into wrapper
            toggle_suppress_off(_data.suppression, did_data_toggle);
            _data.ready = true;

            return _ret;
        }

        if(_data.debug && *_data.debug)
            PRINT_HERE("%s", "nullptr to original function!");
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
#endif
        return Ret{};
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename... Args>
    static TIMEMORY_NOINLINE void wrap_void(Args... _args)
    {
        static_assert(N < Nt, "Error! N must be less than Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        auto& _data = get_data()[N];

        // PRINT_HERE("%s", _data.tool_id.c_str());

        static constexpr bool void_operator = std::is_same<operator_type, void>::value;
        static_assert(void_operator, "operator_type should be void!");

        auto _orig = (void (*)(Args...)) gotcha_get_wrappee(_data.wrappee);

        if(_data.is_finalized)
        {
            if(_orig)
                (*_orig)(_args...);
            return;
        }

        auto _suppress =
            gotcha_suppression::get() || (_data.suppression && *_data.suppression);
        if(!_data.ready || _suppress)
        {
            static thread_local bool _recursive = false;
            if(!_recursive && _data.debug && *_data.debug)
            {
                _recursive = true;
                auto _tid  = threading::get_id();
                fprintf(stderr,
                        "[T%i][%s]> %s is either not ready (ready=%s) or is globally "
                        "suppressed (suppressed=%s)\n",
                        (int) _tid, __FUNCTION__, _data.tool_id.c_str(),
                        (_data.ready) ? "true" : "false", (_suppress) ? "true" : "false");
                fflush(stderr);
                _recursive = false;
            }
            if(_orig)
                (*_orig)(_args...);
            return;
        }

        bool did_data_toggle = false;
        bool did_glob_toggle = false;

        auto toggle_suppress_on = [](bool* _bsuppress, bool& _did) {
            if(_bsuppress && *_bsuppress == false)
            {
                *(_bsuppress) = true;
                _did          = true;
            }
        };

        auto toggle_suppress_off = [](bool* _bsuppress, bool& _did) {
            if(_bsuppress && _did == true && *_bsuppress == true)
            {
                *(_bsuppress) = false;
                _did          = false;
            }
        };

        // make sure the function is not recursively entered
        // (important for allocation-based wrappers)
        _data.ready = false;
        toggle_suppress_on(_data.suppression, did_data_toggle);
        toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);

        if(_orig)
        {
            //
            bundle_type _obj{ _data.tool_id };
            _obj.construct(_args...);
            _obj.start();
            _obj.audit(_data, audit::incoming{}, _args...);
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);

            _data.ready = true;
            invoke<bundle_type>(_obj, _orig, std::forward<Args>(_args)...);
            _data.ready = false;

            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);
            _obj.audit(_data, audit::outgoing{});
            _obj.stop();
        }
        else if(_data.debug && *_data.debug)
        {
            PRINT_HERE("%s", "nullptr to original function!");
        }

        // allow re-entrance into wrapper
        toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);
        toggle_suppress_off(_data.suppression, did_data_toggle);
        _data.ready = true;

#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
#endif
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static TIMEMORY_NOINLINE Ret replace_func(Args... _args)
    {
        static_assert(N < Nt, "Error! N must be less than Nt!");
        static_assert(components_size == 0, "Error! Number of components must be zero!");

#if defined(TIMEMORY_USE_GOTCHA)
        static auto& _data = get_data()[N];

        // PRINT_HERE("%s", _data.tool_id.c_str());

        typedef Ret (*func_t)(Args...);
        using wrap_type = tim::component_tuple<operator_type>;

        static constexpr bool void_operator = std::is_same<operator_type, void>::value;
        static_assert(!void_operator, "operator_type cannot be void!");

        auto _orig = (func_t) gotcha_get_wrappee(_data.wrappee);
        if(!_data.ready)
            return (*_orig)(_args...);

        _data.ready = false;
        static wrap_type _obj{ _data.tool_id };
        Ret              _ret = invoke(_obj, _orig, std::forward<Args>(_args)...);
        _data.ready           = true;
        return _ret;
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
        return Ret{};
#endif
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename... Args>
    static TIMEMORY_NOINLINE void replace_void_func(Args... _args)
    {
        static_assert(N < Nt, "Error! N must be less than Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        static auto& _data = get_data()[N];

        // PRINT_HERE("%s", _data.tool_id.c_str());

        typedef void (*func_t)(Args...);
        using wrap_type = tim::component_tuple<operator_type>;

        static constexpr bool void_operator = std::is_same<operator_type, void>::value;
        static_assert(!void_operator, "operator_type cannot be void!");

        auto _orig = (func_t) gotcha_get_wrappee(_data.wrappee);
        if(!_data.ready)
            (*_orig)(_args...);
        else
        {
            _data.ready = false;
            static wrap_type _obj{ _data.tool_id };
            invoke(_obj, _orig, std::forward<Args>(_args)...);
            _data.ready = true;
        }
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
#endif
    }

private:
    template <typename Tp>
    static auto init_storage(int) -> decltype(Tp::init_storage())
    {
        return Tp::init_storage();
    }

    template <typename Tp>
    static auto init_storage(long)
    {}
};
//
}  // namespace component
}  // namespace tim
