// MIT License
//
// Copyright (c) 2019, The Regents of the University of California,
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

#include "timemory/backends/gotcha.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/details/settings.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/mangler.hpp"

#include <cassert>

//======================================================================================//

namespace tim
{
namespace component
{
using size_type = std::size_t;

//
// template params:
//      _Nt             ==  max number of GOTCHA wrappers
//      _Components     ==  {auto,component}_{tuple,list,hybrid}
//      _Differentiator ==  extra param to differentiate when _Nt and _Components are same
//
//  TODO: filter any gotcha components out of _Components
//
template <size_type _Nt, typename _Components, typename _Differentiator>
struct gotcha
: public base<gotcha<_Nt, _Components, _Differentiator>, void, policy::global_init,
              policy::global_finalize>
{
    static_assert(_Components::contains_gotcha == false,
                  "Error! {auto,component}_{list,tuple,hybrid} in a GOTCHA specification "
                  "cannot include another gotcha_component");

    using value_type = void;
    using this_type  = gotcha<_Nt, _Components, _Differentiator>;
    using base_type =
        base<this_type, value_type, policy::global_init, policy::global_finalize>;
    using storage_type = typename base_type::storage_type;

    template <typename _Tp>
    using array_t = std::array<_Tp, _Nt>;

    using binding_t     = ::tim::gotcha::binding_t;
    using wrappee_t     = ::tim::gotcha::wrappee_t;
    using wrappid_t     = ::tim::gotcha::string_t;
    using error_t       = ::tim::gotcha::error_t;
    using destructor_t  = std::function<void()>;
    using constructor_t = std::function<void()>;
    using atomic_bool_t = std::atomic<bool>;

    using blacklist_t = std::set<std::string>;

    // using config_t = std::tuple<binding_t, wrappee_t, wrappid_t>;
    using config_t          = void;
    using get_initializer_t = std::function<config_t()>;
    using get_blacklist_t   = std::function<blacklist_t()>;

    static std::string label() { return "gotcha"; }
    static std::string description() { return "GOTCHA wrapper"; }
    static value_type  record() { return; }

    //----------------------------------------------------------------------------------//

    static get_initializer_t& get_initializer()
    {
        static get_initializer_t _instance = []() {
            for(const auto& itr : get_constructors())
                itr();
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static get_blacklist_t& get_blacklist()
    {
        static get_blacklist_t _instance = []() { return blacklist_t{}; };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args>
    static void construct(const std::string& _func, int _priority = 0,
                          const std::string& _tool = "")
    {
        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
        auto& _fill_ids = get_filled();

        if(_func.find("MPI_") != std::string::npos ||
           _func.find("mpi_") != std::string::npos)
        {
            static auto mpi_blacklist = { "MPI_Init",       "MPI_Initialized",
                                          "MPI_Pcontrol",   "MPI_Init_thread",
                                          "MPI_Finalize",   "MPI_Comm_rank",
                                          "MPI_Comm_size",  "MPI_T_init_thread",
                                          "MPI_Comm_split", "MPI_Abort" };

            auto tofortran = [](std::string _fort) {
                for(auto& itr : _fort)
                    itr = tolower(itr);
                if(_fort[_fort.length() - 1] != '_')
                    _fort += "_";
                return _fort;
            };

            // if function matches a blacklisted entry, do not construct wrapper
            for(const auto& itr : mpi_blacklist)
                if(_func == itr || _func == tofortran(itr))
                {
                    if(settings::debug())
                        printf("[gotcha]> Skipping gotcha binding for %s...\n",
                               _func.c_str());
                    return;
                }
        }

        // if function matches a blacklisted entry, do not construct wrapper
        for(const auto& itr : get_blacklist()())
        {
            if(_func == itr)
            {
                if(settings::debug())
                    printf("[gotcha]> Skipping gotcha binding for %s...\n",
                           _func.c_str());
                return;
            }
        }

        if(!_fill_ids[_N])
        {
            auto& _bindings     = get_bindings();
            auto& _wrap_ids     = get_wrap_ids();
            auto& _tool_ids     = get_tool_ids();
            auto& _constructors = get_constructors();
            auto& _destructors  = get_destructors();

            // static int _incr = _priority;
            // _priority        = _incr++;

            auto _label = demangle(_func);
            if(_tool.length() > 0 && _label.find(_tool + "/") != 0)
            {
                _label = _tool + "/" + _label;
                while(_label.find("//") != std::string::npos)
                    _label.erase(_label.find("//"), 1);
            }

            _tool_ids[_N] = _label;
            _fill_ids[_N] = true;
            _wrap_ids[_N] = _func;

            error_t ret_prio = ::tim::gotcha::set_priority(_label, _priority);
            check_error<_N>(ret_prio, "set priority");

            _bindings[_N] = std::move(construct_binder<_N, _Ret, _Args...>(_func));

            error_t ret_wrap = ::tim::gotcha::wrap(_bindings[_N], _tool_ids[_N]);
            check_error<_N>(ret_wrap, "binding");

            if(ret_wrap == GOTCHA_SUCCESS)
            {
                _constructors[_N] = [=]() {
                    this_type::configure<_N, _Ret, _Args...>(_wrap_ids[_N], _priority,
                                                             _tool);
                };

                _destructors[_N] = [=]() {
                    this_type::revert<_N, _Ret, _Args...>(_wrap_ids[_N]);
                };

                if(settings::verbose() > 1 || settings::debug())
                {
                    std::cout << "[gotcha::" << __FUNCTION__ << "]> "
                              << "wrapped: " << get_wrap_ids()[_N]
                              << ", wrapped pointer: " << _bindings[_N].wrapper_pointer
                              << ", function_handle: " << _bindings[_N].function_handle
                              << ", name: " << _bindings[_N].name << std::endl;
                }
            }
        }
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args>
    static void configure(const std::string& _func, int _priority = 0,
                          const std::string& _tool = "")
    {
        construct<_N, _Ret, _Args...>(_func, _priority, _tool);
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args>
    static void revert(std::string _func = "")
    {
        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        auto& _fill_ids = get_filled();
        auto& _wrappids = get_wrap_ids();

        if(_fill_ids[_N] && (_func.empty() || _wrappids[_N] == _func))
        {
            auto& _wrappees = get_wrappees();
            auto& _bindings = get_bindings();

            if(_func.empty())
                _func = _wrappids[_N];

            get_filled()[_N] = false;
            auto      _orig  = gotcha_get_wrappee(_wrappees[_N]);
            wrappee_t _dummy = 0x0;
            _bindings[_N]    = { _func.c_str(), _orig, &_dummy };
            error_t ret_wrap = ::tim::gotcha::wrap(_bindings[_N], _wrappids[_N]);
            check_error<_N>(ret_wrap, "unwrap binding");
        }
#else
        consume_parameters(_func);
#endif
        get_destructors()[_N] = []() {};
    }

    //----------------------------------------------------------------------------------//

    static bool& is_configured()
    {
        static bool _instance = false;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void configure()
    {
        static std::mutex            _mtx;
        std::unique_lock<std::mutex> lk(_mtx, std::defer_lock);
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

    static void invoke_global_init(storage_type*)
    {
        // configure();
    }

    static void invoke_global_finalize(storage_type*)
    {
        while(get_started() > 0)
            --get_started();
        while(get_thread_started() > 0)
            --get_thread_started();
        for(auto& itr : get_destructors())
            itr();
    }

    double get_display() const { return 0; }

    double get() const { return 0; }

    void start()
    {
        auto _n = get_started()++;
        auto _t = get_thread_started()++;

        if(_n == 0)
        {
            configure();
        }

        if(_t == 0)
        {
            for(size_type i = 0; i < _Nt; ++i)
                get_ready_flags()[i] = get_filled()[i];
        }
    }

    void stop()
    {
        auto _n = --get_started();
        auto _t = --get_thread_started();

        if(_t == 0)
        {
            for(size_type i = 0; i < _Nt; ++i)
                get_ready_flags()[i] = false;
        }

        if(_n == 0)
        {
            for(auto& itr : get_destructors())
                itr();
        }
    }

public:
    //----------------------------------------------------------------------------------//
    //  secondary method
    //
    template <size_t _N, typename _Ret, typename... _Args>
    struct instrument
    {
        static void generate(const std::string& _func, const std::string& _tool = "",
                             int _priority = 0)
        {
            this_type::configure<_N, _Ret, _Args...>(_func, _priority, _tool);
        }
    };

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args>
    struct instrument<_N, _Ret, std::tuple<_Args...>>
    {
        static void generate(const std::string& _func, const std::string& _tool = "",
                             int _priority = 0)
        {
            this_type::configure<_N, _Ret, _Args...>(_func, _priority, _tool);
        }
    };

    //----------------------------------------------------------------------------------//

private:
    //----------------------------------------------------------------------------------//

    static std::atomic<int64_t>& get_started()
    {
        static std::atomic<int64_t> _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static int64_t& get_thread_started()
    {
        static thread_local int64_t _instance = 0;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static array_t<bool>& get_filled()
    {
        static auto _get = []() {
            array_t<bool> _arr;
            apply<void>::set_value(_arr, false);
            return _arr;
        };

        static array_t<bool> _instance = _get();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static array_t<binding_t>& get_bindings()
    {
        static array_t<binding_t> _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static array_t<wrappee_t>& get_wrappees()
    {
        static array_t<wrappee_t> _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static array_t<wrappid_t>& get_wrap_ids()
    {
        static array_t<wrappid_t> _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static array_t<wrappid_t>& get_tool_ids()
    {
        static array_t<wrappid_t> _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static array_t<destructor_t>& get_destructors()
    {
        static auto _get = []() {
            array_t<destructor_t> _arr;
            for(auto& itr : _arr)
                itr = []() {};
            return _arr;
        };
        static array_t<destructor_t> _instance = _get();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static array_t<constructor_t>& get_constructors()
    {
        static auto _get = []() {
            array_t<constructor_t> _arr;
            for(auto& itr : _arr)
                itr = []() {};
            return _arr;
        };
        static array_t<constructor_t> _instance = _get();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

private:
    //----------------------------------------------------------------------------------//

    static array_t<bool>& get_ready_flags()
    {
        static auto _get = []() {
            array_t<bool> _arr;
            for(auto& itr : _arr)
                itr = false;
            return _arr;
        };
        static thread_local array_t<bool> _instance = _get();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N>
    static void check_error(error_t _ret, const std::string& _prefix)
    {
        if(_ret != GOTCHA_SUCCESS)
        {
            std::stringstream msg;
            msg << _prefix << " at index '" << _N << "' for function '"
                << get_wrap_ids()[_N] << "' returned error code "
                << static_cast<int>(_ret) << ": " << ::tim::gotcha::get_error(_ret)
                << "\n";
            std::cerr << msg.str() << std::endl;
        }
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args,
              typename std::enable_if<!(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _wrappees = get_wrappees();
        return binding_t{ _func.c_str(), (void*) this_type::wrap<_N, _Ret, _Args...>,
                          &_wrappees[_N] };
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args,
              typename std::enable_if<(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _wrappees = get_wrappees();
        return binding_t{ _func.c_str(), (void*) this_type::wrap_void<_N, _Args...>,
                          &_wrappees[_N] };
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args>
    static _Ret wrap(_Args... _args)
    {
        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        typedef _Ret (*func_t)(_Args...);
        func_t _orig = (func_t)(gotcha_get_wrappee(get_wrappees()[_N]));

        if(!get_ready_flags()[_N])
            return (_orig) ? (*_orig)(_args...) : _Ret{};

        // make sure the function is not recursively entered (important for
        // allocation-based wrappers)
        get_ready_flags()[_N] = false;

#    if defined(DEBUG)
        if(settings::verbose() > 2 || settings::debug())
        {
            static std::atomic<int32_t> _count;
            if(_count++ < 50)
            {
                auto _atype =
                    apply<std::string>::join(", ", demangle(typeid(_args).name())...);
                auto _rtype = demangle(typeid(_Ret).name());
                printf("\n");
                printf("[%s]>   wrappee: %s\n", __FUNCTION__,
                       demangle(typeid(_orig).name()).c_str());
                printf("[%s]> signature: %s (*)(%s)\n", __FUNCTION__, _rtype.c_str(),
                       _atype.c_str());
            }
        }
#    endif

        if(_orig)
        {
            static constexpr bool _is_component_type = (_Components::is_component_tuple ||
                                                        _Components::is_component_list ||
                                                        _Components::is_component_hybrid);
            bool _boolA = (_is_component_type || settings::flat_profile());
            bool _boolB = (_is_component_type && settings::flat_profile()) || (!_is_component_type && settings::destructor_report());
            _Components _obj(get_tool_ids()[_N], _boolA, _boolB);
            _obj.customize(get_tool_ids()[_N], _args...);
            _obj.start();

            get_ready_flags()[_N] = true;
            _Ret _ret             = (*_orig)(_args...);
            get_ready_flags()[_N] = false;

            _obj.customize(get_tool_ids()[_N], _ret);
            _obj.stop();
#    if defined(DEBUG)
            if(settings::verbose() > 2 || settings::debug())
            {
                static std::atomic<int32_t> _count;
                if(_count++ < 50)
                {
                    auto _sargs = apply<std::string>::join(", ", _args...);
                    std::cout << "[" << __FUNCTION__ << "]>      args: (" << _sargs
                              << ") "
                              << "result: " << _ret << "\n"
                              << std::endl;
                }
            }
#    endif

            // allow re-entrance into wrapper
            get_ready_flags()[_N] = true;

            return _ret;
        }
        if(settings::debug())
            PRINT_HERE("nullptr to original function!");

        // allow re-entrance into wrapper
        get_ready_flags()[_N] = true;
#else
        consume_parameters(_args...);
        PRINT_HERE("should not be here!");
#endif
        return _Ret{};
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename... _Args>
    static void wrap_void(_Args... _args)
    {
        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        auto _orig = (void (*)(_Args...)) gotcha_get_wrappee(get_wrappees()[_N]);

        if(!get_ready_flags()[_N])
        {
            if(_orig)
                (*_orig)(_args...);
            return;
        }

        // make sure the function is not recursively entered (important for
        // allocation-based wrappers)
        get_ready_flags()[_N] = false;

        if(_orig)
        {
            _Components _obj(get_tool_ids()[_N], true);
            _obj.customize(get_tool_ids()[_N], _args...);
            _obj.start();

            get_ready_flags()[_N] = true;
            (*_orig)(_args...);
            get_ready_flags()[_N] = false;

            _obj.stop();
        } else if(settings::debug())
        {
            PRINT_HERE("nullptr to original function!");
        }

        // allow re-entrance into wrapper
        get_ready_flags()[_N] = true;
#else
        consume_parameters(_args...);
        PRINT_HERE("should not be here!");
#endif
    }

    //----------------------------------------------------------------------------------//
};

}  // namespace component

}  // namespace tim

//--------------------------------------------------------------------------------------//

///
/// attempt to generate a GOTCHA wrapper for a C function (unmangled)
///
#define TIMEMORY_C_GOTCHA(type, idx, func)                                               \
    type::instrument<idx, ::tim::function_traits<decltype(func)>::result_type,           \
                     ::tim::function_traits<decltype(func)>::call_type>::                \
        generate(TIMEMORY_STRINGIZE(func))

///
/// generate a GOTCHA wrapper for function with identical args but different name
///
#define TIMEMORY_DERIVED_GOTCHA(type, idx, func, fort_func)                              \
    type::instrument<                                                                    \
        idx, ::tim::function_traits<decltype(func)>::result_type,                        \
        ::tim::function_traits<decltype(func)>::call_type>::generate(fort_func)

///
/// attempt to generate a GOTCHA wrapper for a C++ function by mangling the function name
/// in general, mangling template function is not supported
///
#define TIMEMORY_CXX_GOTCHA(type, idx, func)                                             \
    type::instrument<idx, ::tim::function_traits<decltype(func)>::result_type,           \
                     ::tim::function_traits<decltype(func)>::call_type>::                \
        generate(::tim::mangle<decltype(func)>(TIMEMORY_STRINGIZE(func)))

///
/// attempt to generate a GOTCHA wrapper for a C++ function by mangling the function name
/// in general, mangling template function is not supported
///
#define TIMEMORY_CXX_MEMFUN_GOTCHA(type, idx, func)                                      \
    type::instrument<idx, ::tim::function_traits<decltype(&func)>::result_type,          \
                     ::tim::function_traits<decltype(&func)>::call_type>::               \
        generate(::tim::mangle<decltype(&func)>(TIMEMORY_STRINGIZE(func)))

///
/// TIMEMORY_C_GOTCHA + ability to pass priority and tool name
///
#define TIMEMORY_C_GOTCHA_TOOL(type, idx, func, ...)                                     \
    type::instrument<idx, ::tim::function_traits<decltype(func)>::result_type,           \
                     ::tim::function_traits<decltype(func)>::call_type>::                \
        generate(TIMEMORY_STRINGIZE(func), __VA_ARGS__)

///
/// TIMEMORY_CXX_GOTCHA + ability to pass priority and tool name
///
#define TIMEMORY_CXX_GOTCHA_TOOL(type, idx, func, ...)                                   \
    type::instrument<idx, ::tim::function_traits<decltype(func)>::result_type,           \
                     ::tim::function_traits<decltype(func)>::call_type>::                \
        generate(::tim::mangle<decltype(func)>(TIMEMORY_STRINGIZE(func)), __VA_ARGS__)
