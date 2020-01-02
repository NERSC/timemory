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

/** \file timemory/components/gotcha.hpp
 * \headerfile timemory/components/gotcha.hpp "timemory/components/gotcha.hpp"
 * Defines GOTCHA component
 *
 */

#pragma once

#include "timemory/backends/gotcha.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/settings.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/mangler.hpp"
#include "timemory/variadic/types.hpp"

#include <cassert>

//======================================================================================//

namespace tim
{
namespace component
{
using size_type = std::size_t;

//======================================================================================//
//
class gotcha_suppression
{
private:
    template <size_type _Nt, typename _Components, typename _Differentiator>
    friend struct gotcha;

    template <typename _Tp, typename _Ret>
    struct gotcha_invoker;

    static bool& get()
    {
        static thread_local bool _instance = false;
        return _instance;
    }

public:
    struct auto_toggle
    {
        explicit auto_toggle(bool& _value, bool _if_equal = false)
        : m_value(_value)
        , m_if_equal(_if_equal)
        {
            if(m_value == m_if_equal)
                m_value = !m_value;
        }

        ~auto_toggle()
        {
            if(m_value != m_if_equal)
                m_value = !m_value;
        }

        auto_toggle(const auto_toggle&) = delete;
        auto_toggle(auto_toggle&&)      = delete;
        auto_toggle& operator=(const auto_toggle&) = delete;
        auto_toggle& operator=(auto_toggle&&) = delete;

    private:
        bool& m_value;
        bool  m_if_equal;
    };
};

//======================================================================================//
///
/// \class component::gotcha_invoker
///
///
template <typename _Tp, typename _Ret>
struct gotcha_invoker
{
    using Type       = _Tp;
    using value_type = typename Type::value_type;
    using base_type  = typename Type::base_type;

    template <typename... _Args>
    static _Ret invoke(_Tp& _obj, bool& _ready, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        return invoke_sfinae(_obj, _ready, _func, std::forward<_Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  Call:
    //
    //      _Ret Type::operator()(_Args...)
    //
    //  instead of gotcha_wrappee
    //
    template <typename... _Args>
    static auto invoke_sfinae_impl(_Tp& _obj, int, bool& _ready, _Ret (*)(_Args...),
                                   _Args&&... _args)
        -> decltype(_obj(std::forward<_Args>(_args)...), _Ret())
    {
        gotcha_suppression::auto_toggle suppress_lock(_ready);
        return _obj(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Call the original gotcha_wrappee
    //
    template <typename... _Args>
    static auto invoke_sfinae_impl(_Tp&, long, bool& _ready, _Ret (*_func)(_Args...),
                                   _Args&&... _args)
        -> decltype(_func(std::forward<_Args>(_args)...), _Ret())
    {
        gotcha_suppression::auto_toggle suppress_lock(_ready);
        return _func(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename... _Args>
    static auto invoke_sfinae(_Tp& _obj, bool& _ready, _Ret (*_func)(_Args...),
                              _Args&&... _args)
        -> decltype(invoke_sfinae_impl(_obj, 0, _ready, _func,
                                       std::forward<_Args>(_args)...),
                    _Ret())
    {
        return invoke_sfinae_impl(_obj, 0, _ready, _func, std::forward<_Args>(_args)...);
    }

    //==================================================================================//
public:
    template <typename... _Args>
    static _Ret invoke(_Tp& _obj, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        return invoke_sfinae(_obj, _func, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
private:
    template <typename... _Args>
    static auto invoke_sfinae_impl(_Tp& _obj, int, _Ret (*)(_Args...), _Args&&... _args)
        -> decltype(_obj(std::forward<_Args>(_args)...), _Ret())
    {
        return _obj(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Call the original gotcha_wrappee
    //
    template <typename... _Args>
    static auto invoke_sfinae_impl(_Tp&, long, _Ret (*_func)(_Args...), _Args&&... _args)
        -> decltype(_func(std::forward<_Args>(_args)...), _Ret())
    {
        return _func(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename... _Args>
    static auto invoke_sfinae(_Tp& _obj, _Ret (*_func)(_Args...), _Args&&... _args)
        -> decltype(invoke_sfinae_impl(_obj, 0, _func, std::forward<_Args>(_args)...),
                    _Ret())
    {
        return invoke_sfinae_impl(_obj, 0, _func, std::forward<_Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//
};

template <typename _Tp>
struct gotcha_invoker<_Tp, void>
{
    using Type        = _Tp;
    using _Ret        = void;
    using return_type = void;
    using value_type  = typename Type::value_type;
    using base_type   = typename Type::base_type;

    template <typename... _Args>
    static _Ret invoke(_Tp& _obj, bool& _ready, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        invoke_sfinae(_obj, _ready, _func, std::forward<_Args>(_args)...);
    }

private:
    //----------------------------------------------------------------------------------//
    //  Call:
    //
    //      _Ret Type::operator()(_Args...)
    //
    //  instead of gotcha_wrappee
    //
    template <typename... _Args>
    static auto invoke_sfinae_impl(_Tp& _obj, int, bool& _ready, _Ret (*)(_Args...),
                                   _Args&&... _args)
        -> decltype(_obj(std::forward<_Args>(_args)...), _Ret())
    {
        gotcha_suppression::auto_toggle suppress_lock(_ready);
        _obj(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Call the original gotcha_wrappee
    //
    template <typename... _Args>
    static auto invoke_sfinae_impl(_Tp&, long, bool& _ready, _Ret (*_func)(_Args...),
                                   _Args&&... _args)
        -> decltype(_func(std::forward<_Args>(_args)...), _Ret())
    {
        gotcha_suppression::auto_toggle suppress_lock(_ready);
        _func(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename... _Args>
    static auto invoke_sfinae(_Tp& _obj, bool& _ready, _Ret (*_func)(_Args...),
                              _Args&&... _args)
        -> decltype(invoke_sfinae_impl(_obj, 0, _ready, _func,
                                       std::forward<_Args>(_args)...),
                    _Ret())
    {
        invoke_sfinae_impl(_obj, 0, _ready, _func, std::forward<_Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//

    //==================================================================================//
public:
    template <typename... _Args>
    static _Ret invoke(_Tp& _obj, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        invoke_sfinae(_obj, _func, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
private:
    template <typename... _Args>
    static auto invoke_sfinae_impl(_Tp& _obj, int, _Ret (*)(_Args...), _Args&&... _args)
        -> decltype(_obj(std::forward<_Args>(_args)...), _Ret())
    {
        _obj(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Call the original gotcha_wrappee
    //
    template <typename... _Args>
    static auto invoke_sfinae_impl(_Tp&, long, _Ret (*_func)(_Args...), _Args&&... _args)
        -> decltype(_func(std::forward<_Args>(_args)...), _Ret())
    {
        _func(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  Wrapper that calls one of two above
    //
    template <typename... _Args>
    static auto invoke_sfinae(_Tp& _obj, _Ret (*_func)(_Args...), _Args&&... _args)
        -> decltype(invoke_sfinae_impl(_obj, 0, _func, std::forward<_Args>(_args)...),
                    _Ret())
    {
        invoke_sfinae_impl(_obj, 0, _func, std::forward<_Args>(_args)...);
    }
    //
    //----------------------------------------------------------------------------------//
};

//======================================================================================//

template <typename _Tp>
struct gotcha_differentiator
{
    template <typename _Up>
    static constexpr decltype(_Up::is_component, bool()) test_is_component(int)
    {
        return true;
    }

    template <typename _Up>
    static constexpr bool test_is_component(...)
    {
        return false;
    }

    static constexpr bool value        = test_is_component<_Tp>(int());
    static constexpr bool is_component = test_is_component<_Tp>(int());
};

//======================================================================================//

template <typename... _Types>
struct gotcha_components_size
{
    static constexpr size_t value = sizeof...(_Types);
};

//--------------------------------------------------------------------------------------//

template <typename... _Types, template <typename...> class _Tuple>
struct gotcha_components_size<_Tuple<_Types...>>
{
    static constexpr size_t value = sizeof...(_Types);
};

//======================================================================================//
//
// template params:
//      _Nt             ==  max number of GOTCHA wrappers
//      _Components     ==  {auto,component}_{tuple,list,hybrid}
//      _Differentiator ==  extra param to differentiate when _Nt and _Components are same
//
//  TODO: filter any gotcha components out of _Components
//
template <size_type _Nt, typename _Components, typename _Differentiator>
struct gotcha : public base<gotcha<_Nt, _Components, _Differentiator>, void>
{
    static_assert(_Components::contains_gotcha == false,
                  "Error! {auto,component}_{list,tuple,hybrid} in a GOTCHA specification "
                  "cannot include another gotcha_component");

    using value_type     = void;
    using this_type      = gotcha<_Nt, _Components, _Differentiator>;
    using base_type      = base<this_type, value_type>;
    using storage_type   = typename base_type::storage_type;
    using component_type = typename _Components::component_type;
    using type_tuple     = typename _Components::type_tuple;

    template <typename _Tp>
    using array_t = std::array<_Tp, _Nt>;

    using binding_t     = backend::gotcha::binding_t;
    using wrappee_t     = backend::gotcha::wrappee_t;
    using wrappid_t     = backend::gotcha::string_t;
    using error_t       = backend::gotcha::error_t;
    using destructor_t  = std::function<void()>;
    using constructor_t = std::function<void()>;
    using atomic_bool_t = std::atomic<bool>;

    using select_list_t = std::set<std::string>;

    // using config_t = std::tuple<binding_t, wrappee_t, wrappid_t>;
    using config_t          = void;
    using get_initializer_t = std::function<config_t()>;
    using get_select_list_t = std::function<select_list_t()>;

    static constexpr size_t components_size = component_type::size();
    static constexpr bool   differentiator_is_component =
        (is_one_of<_Differentiator, type_tuple>::value ||
         (components_size == 0 && gotcha_differentiator<_Differentiator>::is_component));

    using operator_type = typename std::conditional<(differentiator_is_component),
                                                    _Differentiator, void>::type;

    static std::string label() { return "gotcha"; }
    static std::string description() { return "GOTCHA wrapper"; }
    static value_type  record() { return; }

    //----------------------------------------------------------------------------------//

    static get_initializer_t& get_initializer()
    {
        static get_initializer_t _instance = []() {
            for(const auto& itr : get_data())
                itr.constructor();
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// reject listed functions are never wrapped by GOTCHA
    static get_select_list_t& get_reject_list()
    {
        static get_select_list_t _instance = []() { return select_list_t{}; };
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// when a permit list is provided, only these functions are wrapped by GOTCHA
    static get_select_list_t& get_permit_list()
    {
        static get_select_list_t _instance = []() { return select_list_t{}; };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static bool& get_default_ready()
    {
        static bool _instance = false;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args>
    static void construct(const std::string& _func, int _priority = 0,
                          const std::string& _tool = "")
    {
        gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());

        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
        auto& _data = get_data()[_N];

        if(!is_permitted<_N, _Ret, _Args...>(_func))
            return;

        if(!_data.filled)
        {
            // static int _incr = _priority;
            // _priority        = _incr++;

            auto _label = demangle(_func);
            if(_tool.length() > 0 && _label.find(_tool + "/") != 0)
            {
                _label = _tool + "/" + _label;
                while(_label.find("//") != std::string::npos)
                    _label.erase(_label.find("//"), 1);
            }

            // ensure the hash to string pairing is stored
            storage_type::instance()->add_hash_id(_label);

            _data.filled      = true;
            _data.priority    = _priority;
            _data.tool_id     = _label;
            _data.wrap_id     = _func;
            _data.ready       = get_default_ready();
            _data.constructor = [=]() {
                this_type::construct<_N, _Ret, _Args...>(_data.wrap_id);
            };
            _data.destructor = [=]() { this_type::revert<_N, _Ret, _Args...>(); };
            _data.binding =
                std::move(construct_binder<_N, _Ret, _Args...>(_data.wrap_id));
            error_t ret_wrap = backend::gotcha::wrap(_data.binding, _data.tool_id);
            check_error<_N>(ret_wrap, "binding");
        }

        if(!_data.is_active)
        {
            _data.is_active = true;
            error_t ret_prio =
                backend::gotcha::set_priority(_data.tool_id, _data.priority);
            check_error<_N>(ret_prio, "set priority");
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
    static void revert()
    {
        gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());

        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
        auto& _data = get_data()[_N];

        if(_data.filled && _data.is_active)
        {
            _data.is_active = false;

            error_t ret_prio = backend::gotcha::set_priority(_data.tool_id, -1);
            check_error<_N>(ret_prio, "get priority");

            /*
            _data.wrapper = 0x0;
            _data.binding = std::move(revert_binder<_N, _Ret, _Args...>(_data.wrap_id));

            error_t ret_wrap = backend::gotcha::wrap(_data.binding, _data.wrap_id);
            check_error<_N>(ret_wrap, "unwrap binding");
            */

            _data.ready = get_default_ready();
        }
    }

    //----------------------------------------------------------------------------------//

    static bool& is_configured()
    {
        static bool _instance = false;
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static std::mutex& get_mutex()
    {
        static std::mutex _mtx;
        return _mtx;
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

    static void global_init(storage_type*)
    {
        // if(get_default_ready())
        //     configure();
    }

    static void global_finalize(storage_type*)
    {
        while(get_started() > 0)
            --get_started();
        while(get_thread_started() > 0)
            --get_thread_started();
        disable();
    }

    static void thread_init(storage_type*)
    {
        auto& _data = get_data();
        for(size_type i = 0; i < _Nt; ++i)
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

        if(_n == 0 && !storage_type::is_finalizing())
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
            for(size_type i = 0; i < _Nt; ++i)
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
            for(size_type i = 0; i < _Nt; ++i)
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
    struct instrument<_N, _Ret, std::tuple<_Args...>> : instrument<_N, _Ret, _Args...>
    {};

    //----------------------------------------------------------------------------------//

private:
    //----------------------------------------------------------------------------------//
    /// \brief gotcha_data
    /// Holds the properties for wrapping and unwrapping a binding
    struct gotcha_data
    {
        gotcha_data() = default;

        bool          ready        = get_default_ready();  /// ready to be used
        bool          filled       = false;                /// structure is populated
        bool          is_active    = false;                /// is currently wrapping
        bool          is_finalized = false;                /// no more wrapping is allowed
        int           priority     = 0;                    /// current priority
        binding_t     binding      = binding_t{};          /// hold the binder set
        wrappee_t     wrapper      = 0x0;      /// the func pointer doing wrapping
        wrappee_t     wrappee      = 0x0;      /// the func pointer being wrapped
        wrappid_t     wrap_id      = "";       /// the function name (possibly mangled)
        wrappid_t     tool_id      = "";       /// the function name (unmangled)
        constructor_t constructor  = []() {};  /// wrap the function
        destructor_t  destructor   = []() {};  /// unwrap the function
    };

    //----------------------------------------------------------------------------------//

    static array_t<gotcha_data>& get_data()
    {
        static auto _get = []() {
            array_t<gotcha_data> _arr;
            return _arr;
        };
        static array_t<gotcha_data> _instance = _get();
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// \brief get_started()
    /// Global counter of active gotchas started
    static std::atomic<int64_t>& get_started()
    {
        static std::atomic<int64_t> _instance(0);
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// \brief get_thread_started()
    /// Thread-local counter of activate gotchas
    static int64_t& get_thread_started()
    {
        static thread_local int64_t _instance = 0;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// \brief is_permitted()
    /// Check the permit list and reject list for whether the component is permitted
    /// to be wrapped.
    template <size_t _N, typename _Ret, typename... _Args>
    static bool is_permitted(const std::string& _func)
    {
        if(_func.find("MPI_") != std::string::npos ||
           _func.find("mpi_") != std::string::npos)
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
                if(_func == itr || _func == tofortran(itr))
                {
                    if(settings::debug())
                        printf("[gotcha]> Skipping gotcha binding for %s...\n",
                               _func.c_str());
                    return false;
                }
        }

        const select_list_t& _permit_list = get_permit_list()();
        const select_list_t& _reject_list = get_reject_list()();

        // if function matches a reject_listed entry, do not construct wrapper
        if(_reject_list.count(_func) > 0)
        {
            if(settings::debug())
                printf(
                    "[gotcha]> GOTCHA binding for function '%s' is in reject list...\n",
                    _func.c_str());
            return false;
        }

        // if a permit_list was provided, then do not construct wrapper if not in permit
        // list
        if(_permit_list.size() > 0)
        {
            if(_permit_list.count(_func) == 0)
            {
                if(settings::debug())
                    printf("[gotcha]> GOTCHA binding for function '%s' is not in permit "
                           "list...\n",
                           _func.c_str());
                return false;
            }
        }

        return true;
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N>
    static void check_error(error_t _ret, const std::string& _prefix)
    {
        if(_ret != GOTCHA_SUCCESS)
        {
            auto&             _data = get_data()[_N];
            std::stringstream msg;
            msg << _prefix << " at index '" << _N << "' for function '" << _data.wrap_id
                << "' returned error code " << static_cast<int>(_ret) << ": "
                << backend::gotcha::get_error(_ret) << "\n";
            std::cerr << msg.str() << std::endl;
        }
        else if(settings::verbose() > 1 || settings::debug())
        {
#if defined(TIMEMORY_USE_GOTCHA)
            auto&             _data = get_data()[_N];
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

    template <size_t _N, typename _Ret, typename... _Args, typename _This = this_type,
              typename std::enable_if<(_This::components_size != 0), int>::type      = 0,
              typename std::enable_if<!(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[_N];
        _data.wrapper = (void*) this_type::wrap<_N, _Ret, _Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args, typename _This = this_type,
              typename std::enable_if<(_This::components_size != 0), int>::type     = 0,
              typename std::enable_if<(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[_N];
        _data.wrapper = (void*) this_type::wrap_void<_N, _Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args, typename _This = this_type,
              typename std::enable_if<(_This::components_size == 0), int>::type      = 0,
              typename std::enable_if<!(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[_N];
        _data.wrapper = (void*) this_type::wrap_op<_N, _Ret, _Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args, typename _This = this_type,
              typename std::enable_if<(_This::components_size == 0), int>::type     = 0,
              typename std::enable_if<(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[_N];
        _data.wrapper = (void*) this_type::wrap_void_op<_N, _Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args, typename _This = this_type,
              typename std::enable_if<(_This::components_size != 0), int>::type      = 0,
              typename std::enable_if<!(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t revert_binder(const std::string& _func)
    {
        auto& _data = get_data()[_N];
        return binding_t{ _func.c_str(), _data.wrappee, &_data.wrapper };
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args, typename _This = this_type,
              typename std::enable_if<(_This::components_size != 0), int>::type     = 0,
              typename std::enable_if<(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t revert_binder(const std::string& _func)
    {
        auto& _data = get_data()[_N];
        return binding_t{ _func.c_str(), _data.wrappee, &_data.wrapper };
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args, typename _This = this_type,
              typename std::enable_if<(_This::components_size == 0), int>::type      = 0,
              typename std::enable_if<!(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t revert_binder(const std::string& _func)
    {
        auto& _data = get_data()[_N];
        void* _orig = gotcha_get_wrappee(_data.wrappee);
        return binding_t{ _func.c_str(), _data.wrappee, &_orig };
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args, typename _This = this_type,
              typename std::enable_if<(_This::components_size == 0), int>::type     = 0,
              typename std::enable_if<(std::is_same<_Ret, void>::value), int>::type = 0>
    static binding_t revert_binder(const std::string& _func)
    {
        auto& _data = get_data()[_N];
        void* _orig = gotcha_get_wrappee(_data.wrappee);
        return binding_t{ _func.c_str(), _data.wrappee, &_orig };
    }

    //----------------------------------------------------------------------------------//

    template <typename _Comp, typename _Ret, typename... _Args,
              typename _This                                         = this_type,
              enable_if_t<(_This::differentiator_is_component), int> = 0,
              enable_if_t<!(std::is_same<_Ret, void>::value), int>   = 0>
    static _Ret invoke(_Comp& _comp, bool& _ready, _Ret (*_func)(_Args...),
                       _Args&&... _args)
    {
        using _Type    = _Differentiator;
        using _Invoker = gotcha_invoker<_Type, _Ret>;
        _Type& _obj    = _comp.template get<_Type>();
        return _Invoker::invoke(_obj, _ready, _func, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Comp, typename _Ret, typename... _Args,
              typename _This                                          = this_type,
              enable_if_t<!(_This::differentiator_is_component), int> = 0,
              enable_if_t<!(std::is_same<_Ret, void>::value), int>    = 0>
    static _Ret invoke(_Comp&, bool& _ready, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        gotcha_suppression::auto_toggle suppress_lock(_ready);
        // _ready    = true;
        _Ret _ret = _func(std::forward<_Args>(_args)...);
        // _ready    = false;
        return _ret;
    }

    //----------------------------------------------------------------------------------//

    template <typename _Comp, typename _Ret, typename... _Args,
              typename _This                                         = this_type,
              enable_if_t<(_This::differentiator_is_component), int> = 0,
              enable_if_t<(std::is_same<_Ret, void>::value), int>    = 0>
    static void invoke(_Comp& _comp, bool& _ready, _Ret (*_func)(_Args...),
                       _Args&&... _args)
    {
        using _Type    = _Differentiator;
        using _Invoker = gotcha_invoker<_Type, _Ret>;
        _Type& _obj    = _comp.template get<_Type>();
        _Invoker::invoke(_obj, _ready, _func, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Comp, typename _Ret, typename... _Args,
              typename _This                                          = this_type,
              enable_if_t<!(_This::differentiator_is_component), int> = 0,
              enable_if_t<(std::is_same<_Ret, void>::value), int>     = 0>
    static void invoke(_Comp&, bool& _ready, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        gotcha_suppression::auto_toggle suppress_lock(_ready);
        // _ready    = true;
        _func(std::forward<_Args>(_args)...);
        // _ready    = false;
    }

    //----------------------------------------------------------------------------------//

    template <typename _Comp, typename _Ret, typename... _Args,
              typename _This                                         = this_type,
              enable_if_t<(_This::differentiator_is_component), int> = 0,
              enable_if_t<!(std::is_same<_Ret, void>::value), int>   = 0>
    static _Ret invoke(_Comp& _comp, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        using _Type    = _Differentiator;
        using _Invoker = gotcha_invoker<_Type, _Ret>;
        _Type& _obj    = _comp.template get<_Type>();
        return _Invoker::invoke(_obj, _func, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Comp, typename _Ret, typename... _Args,
              typename _This                                          = this_type,
              enable_if_t<!(_This::differentiator_is_component), int> = 0,
              enable_if_t<!(std::is_same<_Ret, void>::value), int>    = 0>
    static _Ret invoke(_Comp&, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        return _func(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Comp, typename _Ret, typename... _Args,
              typename _This                                         = this_type,
              enable_if_t<(_This::differentiator_is_component), int> = 0,
              enable_if_t<(std::is_same<_Ret, void>::value), int>    = 0>
    static void invoke(_Comp& _comp, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        using _Type    = _Differentiator;
        using _Invoker = gotcha_invoker<_Type, _Ret>;
        _Type& _obj    = _comp.template get<_Type>();
        _Invoker::invoke(_obj, _func, std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename _Comp, typename _Ret, typename... _Args,
              typename _This                                          = this_type,
              enable_if_t<!(_This::differentiator_is_component), int> = 0,
              enable_if_t<(std::is_same<_Ret, void>::value), int>     = 0>
    static void invoke(_Comp&, _Ret (*_func)(_Args...), _Args&&... _args)
    {
        _func(std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args>
    static _Ret wrap(_Args... _args)
    {
        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        auto& _data = get_data()[_N];

        typedef _Ret (*func_t)(_Args...);
        func_t _orig = (func_t)(gotcha_get_wrappee(_data.wrappee));

        auto& _global_suppress = gotcha_suppression::get();
        if(!_data.ready || _global_suppress)
        {
            if(settings::debug())
            {
                static std::atomic<int64_t> _tcount(0);
                static thread_local int64_t _tid = _tcount++;
                std::stringstream           ss;
                ss << "[T" << _tid << "]> is either not ready (" << std::boolalpha
                   << _data.ready << ") or is globally suppressed (" << _global_suppress
                   << ")...\n";
                std::cout << ss.str() << std::flush;
            }
            return (_orig) ? (*_orig)(_args...) : _Ret{};
        }

        // make sure the function is not recursively entered (important for
        // allocation-based wrappers)
        _data.ready = false;
        gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());

        if(_orig)
        {
            // component_type is always: component_{tuple,list,hybrid}
            component_type _obj(_data.tool_id, true, settings::flat_profile());
            _obj.start();
            _obj.audit(_data.tool_id, _args...);
            _Ret _ret = invoke<component_type>(_obj, _data.ready, _orig,
                                               std::forward<_Args>(_args)...);
            _obj.audit(_data.tool_id, _ret);
            _obj.stop();

            // allow re-entrance into wrapper
            _data.ready = true;

            return _ret;
        }
        if(settings::debug())
            PRINT_HERE("%s", "nullptr to original function!");

        // allow re-entrance into wrapper
        _data.ready = true;
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
#endif
        return _Ret{};
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename... _Args>
    static void wrap_void(_Args... _args)
    {
        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        auto& _data = get_data()[_N];

        auto _orig = (void (*)(_Args...)) gotcha_get_wrappee(_data.wrappee);

        auto& _global_suppress = gotcha_suppression::get();
        if(!_data.ready || _global_suppress)
        {
            if(settings::debug())
            {
                static std::atomic<int64_t> _tcount(0);
                static thread_local int64_t _tid = _tcount++;
                std::stringstream           ss;
                ss << "[T" << _tid << "]> is either not ready (" << std::boolalpha
                   << _data.ready << ") or is globally suppressed (" << _global_suppress
                   << ")...\n";
                std::cout << ss.str() << std::flush;
            }
            if(_orig)
                (*_orig)(_args...);
            return;
        }

        // make sure the function is not recursively entered (important for
        // allocation-based wrappers)
        _data.ready = false;
        gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());

        if(_orig)
        {
            component_type _obj(_data.tool_id, true, settings::flat_profile());
            _obj.start();
            _obj.audit(_data.tool_id, _args...);
            invoke<component_type>(_obj, _data.ready, _orig,
                                   std::forward<_Args>(_args)...);
            _obj.audit(_data.tool_id);
            _obj.stop();
        }
        else if(settings::debug())
        {
            PRINT_HERE("%s", "nullptr to original function!");
        }

        // allow re-entrance into wrapper
        _data.ready = true;
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
#endif
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Ret, typename... _Args>
    static _Ret wrap_op(_Args... _args)
    {
        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
        static_assert(components_size == 0, "Error! Number of components must be zero!");

#if defined(TIMEMORY_USE_GOTCHA)
        static auto& _data = get_data()[_N];
        typedef _Ret (*func_t)(_Args...);
        using wrap_type = tim::component_tuple<operator_type>;

        auto _orig = (func_t) gotcha_get_wrappee(_data.wrappee);
        if(!_data.ready)
            return (*_orig)(_args...);

        _data.ready = false;
        static thread_local wrap_type _obj(_data.tool_id, false);
        _Ret _ret   = invoke(_obj, _orig, std::forward<_Args>(_args)...);
        _data.ready = true;
        return _ret;
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
        return _Ret{};
#endif
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename... _Args>
    static void wrap_void_op(_Args... _args)
    {
        static_assert(_N < _Nt, "Error! _N must be less than _Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        static auto& _data = get_data()[_N];
        typedef void (*func_t)(_Args...);
        auto _orig      = (func_t) gotcha_get_wrappee(_data.wrappee);
        using wrap_type = tim::component_tuple<operator_type>;

        if(!_data.ready)
            (*_orig)(_args...);
        else
        {
            _data.ready = false;
            static thread_local wrap_type _obj(_data.tool_id, false);
            invoke(_obj, _orig, std::forward<_Args>(_args)...);
            _data.ready = true;
        }
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
#endif
    }

    //----------------------------------------------------------------------------------//
};  // namespace tim

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
/// -- useful for C++ template function where the mangled name is determined
///    via `nm --dynamic <EXE>`
///
#define TIMEMORY_DERIVED_GOTCHA(type, idx, func, deriv_name)                             \
    type::instrument<                                                                    \
        idx, ::tim::function_traits<decltype(func)>::result_type,                        \
        ::tim::function_traits<decltype(func)>::call_type>::generate(deriv_name)

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
