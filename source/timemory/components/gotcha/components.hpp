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
#include "timemory/macros.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"

#include "timemory/components/gotcha/backends.hpp"
#include "timemory/components/gotcha/types.hpp"

#include "timemory/mpl/function_traits.hpp"

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
//      Components     ==  {auto,component}_{tuple,list,hybrid}
//      Differentiator ==  extra param to differentiate when Nt and Components are same
//
//  TODO: filter any gotcha components out of Components
//
template <size_t Nt, typename Components, typename Differentiator>
struct gotcha : public base<gotcha<Nt, Components, Differentiator>, void>
{
    static_assert(Components::has_gotcha_v == false,
                  "Error! {auto,component}_{list,tuple,hybrid} in a GOTCHA specification "
                  "cannot include another gotcha_component");

    using value_type     = void;
    using this_type      = gotcha<Nt, Components, Differentiator>;
    using base_type      = base<this_type, value_type>;
    using storage_type   = typename base_type::storage_type;
    using component_type = typename Components::component_type;
    using type_tuple     = typename Components::type_tuple;

    template <typename Tp>
    using array_t = std::array<Tp, Nt>;

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
        (is_one_of<Differentiator, type_tuple>::value ||
         (components_size == 0 && gotcha_differentiator<Differentiator>::is_component));

    using operator_type = typename std::conditional<(differentiator_is_component),
                                                    Differentiator, void>::type;

    static std::string label() { return "gotcha"; }
    static std::string description() { return "GOTCHA wrapper"; }
    static value_type  record() { return; }

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

    static void add_global_suppression(const std::string& func)
    {
        get_suppresses().insert(func);
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static void construct(const std::string& _func, int _priority = 0,
                          const std::string& _tool = "")
    {
        gotcha_suppression::auto_toggle suppress_lock(gotcha_suppression::get());

        component_type::init_storage();

        static_assert(N < Nt, "Error! N must be less than Nt!");
        auto& _data = get_data()[N];

        if(!is_permitted<N, Ret, Args...>(_func))
            return;

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
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static void configure(const std::string& _func, int _priority = 0,
                          const std::string& _tool = "")
    {
        construct<N, Ret, Args...>(_func, _priority, _tool);
    }

    //----------------------------------------------------------------------------------//

    template <size_t N>
    static void revert()
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
                _data.ready = false;
            else
                _data.ready = get_default_ready();
        }
    }

    //----------------------------------------------------------------------------------//

    static bool& is_configured() { return get_persistent_data().m_is_configured; }

    //----------------------------------------------------------------------------------//

    static std::mutex& get_mutex() { return get_persistent_data().m_mutex; }

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
    /// \brief gotcha_data
    /// Holds the properties for wrapping and unwrapping a binding
    struct gotcha_data
    {
        gotcha_data()  = default;
        ~gotcha_data() = default;

        gotcha_data(const gotcha_data&) = delete;
        gotcha_data(gotcha_data&&)      = delete;
        gotcha_data& operator=(const gotcha_data&) = delete;
        gotcha_data& operator=(gotcha_data&&) = delete;

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
        bool*         suppression  = nullptr;  /// turn on/off some suppression variable
    };

    //----------------------------------------------------------------------------------//
    //
    struct persistent_data
    {
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
    /// \fn get_data()
    /// \brief Gotcha wrapper data
    static array_t<gotcha_data>& get_data() { return get_persistent_data().m_data; }

    //----------------------------------------------------------------------------------//
    /// \fn get_started()
    /// \brief Global counter of active gotchas started
    static std::atomic<int64_t>& get_started() { return get_persistent_data().m_started; }

    //----------------------------------------------------------------------------------//
    /// \fn get_thread_started()
    /// \brief Thread-local counter of activate gotchas
    static int64_t& get_thread_started()
    {
        static thread_local int64_t _instance = 0;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// \fn get_suppresses()
    /// \brief global suppression when being used
    static std::set<std::string>& get_suppresses()
    {
        return get_persistent_data().m_suppress;
    }

    //----------------------------------------------------------------------------------//
    /// \brief is_permitted()
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
              typename std::enable_if<(This::components_size != 0), int>::type      = 0,
              typename std::enable_if<!(std::is_same<Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[N];
        _data.wrapper = (void*) this_type::wrap<N, Ret, Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args, typename This = this_type,
              typename std::enable_if<(This::components_size != 0), int>::type     = 0,
              typename std::enable_if<(std::is_same<Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[N];
        _data.wrapper = (void*) this_type::wrap_void<N, Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args, typename This = this_type,
              typename std::enable_if<(This::components_size == 0), int>::type      = 0,
              typename std::enable_if<!(std::is_same<Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[N];
        _data.wrapper = (void*) this_type::wrap_op<N, Ret, Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args, typename This = this_type,
              typename std::enable_if<(This::components_size == 0), int>::type     = 0,
              typename std::enable_if<(std::is_same<Ret, void>::value), int>::type = 0>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data   = get_data()[N];
        _data.wrapper = (void*) this_type::wrap_void_op<N, Args...>;
        return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<(This::differentiator_is_component), int> = 0,
              enable_if_t<!(std::is_same<Ret, void>::value), int>   = 0>
    static Ret invoke(Comp& _comp, bool& _ready, Ret (*_func)(Args...), Args&&... _args)
    {
        using Type    = Differentiator;
        using Invoker = gotcha_invoker<Type, Ret>;
        Type& _obj    = *_comp.template get<Type>();
        return Invoker::invoke(_obj, _ready, _func, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<!(This::differentiator_is_component), int> = 0,
              enable_if_t<!(std::is_same<Ret, void>::value), int>    = 0>
    static Ret invoke(Comp&, bool&, Ret (*_func)(Args...), Args&&... _args)
    {
        // gotcha_suppression::auto_toggle suppress_lock(_ready);
        Ret _ret = _func(std::forward<Args>(_args)...);
        return _ret;
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<(This::differentiator_is_component), int> = 0,
              enable_if_t<(std::is_same<Ret, void>::value), int>    = 0>
    static void invoke(Comp& _comp, bool& _ready, Ret (*_func)(Args...), Args&&... _args)
    {
        using Type    = Differentiator;
        using Invoker = gotcha_invoker<Type, Ret>;
        Type& _obj    = *_comp.template get<Type>();
        Invoker::invoke(_obj, _ready, _func, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<!(This::differentiator_is_component), int> = 0,
              enable_if_t<(std::is_same<Ret, void>::value), int>     = 0>
    static void invoke(Comp&, bool&, Ret (*_func)(Args...), Args&&... _args)
    {
        // gotcha_suppression::auto_toggle suppress_lock(_ready);
        _func(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<(This::differentiator_is_component), int> = 0,
              enable_if_t<!(std::is_same<Ret, void>::value), int>   = 0>
    static Ret invoke(Comp& _comp, Ret (*_func)(Args...), Args&&... _args)
    {
        using Type    = Differentiator;
        using Invoker = gotcha_invoker<Type, Ret>;
        Type& _obj    = *_comp.template get<Type>();
        return Invoker::invoke(_obj, _func, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<!(This::differentiator_is_component), int> = 0,
              enable_if_t<!(std::is_same<Ret, void>::value), int>    = 0>
    static Ret invoke(Comp&, Ret (*_func)(Args...), Args&&... _args)
    {
        return _func(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<(This::differentiator_is_component), int> = 0,
              enable_if_t<(std::is_same<Ret, void>::value), int>    = 0>
    static void invoke(Comp& _comp, Ret (*_func)(Args...), Args&&... _args)
    {
        using Type    = Differentiator;
        using Invoker = gotcha_invoker<Type, Ret>;
        Type& _obj    = *_comp.template get<Type>();
        Invoker::invoke(_obj, _func, std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args, typename This = this_type,
              enable_if_t<!(This::differentiator_is_component), int> = 0,
              enable_if_t<(std::is_same<Ret, void>::value), int>     = 0>
    static void invoke(Comp&, Ret (*_func)(Args...), Args&&... _args)
    {
        _func(std::forward<Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static Ret wrap(Args... _args)
    {
        static_assert(N < Nt, "Error! N must be less than Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        auto& _data = get_data()[N];

        typedef Ret (*func_t)(Args...);
        func_t _orig = (func_t)(gotcha_get_wrappee(_data.wrappee));

        auto& _global_suppress = gotcha_suppression::get();
        if(!_data.ready || _global_suppress)
        {
            if(settings::debug())
            {
                static std::atomic<int64_t> _tcount(0);
                static thread_local int64_t _tid = _tcount++;
                std::stringstream           ss;
                ss << "[T" << _tid << "]> " << _data.tool_id << " is either not ready ("
                   << std::boolalpha << !_data.ready << ") or is globally suppressed ("
                   << _global_suppress << ")...\n";
                std::cout << ss.str() << std::flush;
            }
            return (_orig) ? (*_orig)(_args...) : Ret{};
        }

        bool did_data_toggle = false;
        bool did_glob_toggle = false;

        auto toggle_suppress_on = [](bool* _suppress, bool& _did) {
            if(_suppress && *_suppress == false)
            {
                *(_suppress) = true;
                _did         = true;
            }
        };

        auto toggle_suppress_off = [](bool* _suppress, bool& _did) {
            if(_suppress && _did == true && *_suppress == true)
            {
                *(_suppress) = false;
                _did         = false;
            }
        };

        if(_orig)
        {
            // make sure the function is not recursively entered
            // (important for allocation-based wrappers)
            _data.ready = false;
            toggle_suppress_on(_data.suppression, did_data_toggle);

            // component_type is always: component_{tuple,list,hybrid}
            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);
            component_type _obj(_data.tool_id, true);
            _obj.construct(_args...);
            _obj.start();
            _obj.audit(_data.tool_id, _args...);
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);

            _data.ready = true;
            Ret _ret    = invoke<component_type>(_obj, _data.ready, _orig,
                                              std::forward<Args>(_args)...);
            _data.ready = false;

            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);
            _obj.audit(_data.tool_id, _ret);
            _obj.stop();
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);

            // allow re-entrance into wrapper
            toggle_suppress_off(_data.suppression, did_data_toggle);
            _data.ready = true;

            return _ret;
        }

        if(settings::debug())
            PRINT_HERE("%s", "nullptr to original function!");
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
#endif
        return Ret{};
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename... Args>
    static void wrap_void(Args... _args)
    {
        static_assert(N < Nt, "Error! N must be less than Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        auto& _data = get_data()[N];

        auto _orig = (void (*)(Args...)) gotcha_get_wrappee(_data.wrappee);

        auto& _global_suppress = gotcha_suppression::get();
        if(!_data.ready || _global_suppress)
        {
            if(settings::debug())
            {
                static std::atomic<int64_t> _tcount(0);
                static thread_local int64_t _tid = _tcount++;
                std::stringstream           ss;
                ss << "[T" << _tid << "]> " << _data.tool_id << " is either not ready ("
                   << std::boolalpha << !_data.ready << ") or is globally suppressed ("
                   << _global_suppress << ")...\n";
                std::cout << ss.str() << std::flush;
            }
            if(_orig)
                (*_orig)(_args...);
            return;
        }

        bool did_data_toggle = false;
        bool did_glob_toggle = false;

        auto toggle_suppress_on = [](bool* _suppress, bool& _did) {
            if(_suppress && *_suppress == false)
            {
                *(_suppress) = true;
                _did         = true;
            }
        };

        auto toggle_suppress_off = [](bool* _suppress, bool& _did) {
            if(_suppress && _did == true && *_suppress == true)
            {
                *(_suppress) = false;
                _did         = false;
            }
        };

        // make sure the function is not recursively entered
        // (important for allocation-based wrappers)
        _data.ready = false;
        toggle_suppress_on(_data.suppression, did_data_toggle);
        toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);

        if(_orig)
        {
            component_type _obj(_data.tool_id, true);
            _obj.construct(_args...);
            _obj.start();
            _obj.audit(_data.tool_id, _args...);
            toggle_suppress_off(&gotcha_suppression::get(), did_glob_toggle);

            _data.ready = true;
            invoke<component_type>(_obj, _data.ready, _orig,
                                   std::forward<Args>(_args)...);
            _data.ready = false;

            toggle_suppress_on(&gotcha_suppression::get(), did_glob_toggle);
            _obj.audit(_data.tool_id);
            _obj.stop();
        }
        else if(settings::debug())
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
    static Ret wrap_op(Args... _args)
    {
        static_assert(N < Nt, "Error! N must be less than Nt!");
        static_assert(components_size == 0, "Error! Number of components must be zero!");

#if defined(TIMEMORY_USE_GOTCHA)
        static auto& _data = get_data()[N];
        typedef Ret (*func_t)(Args...);
        using wrap_type = tim::component_tuple<operator_type>;

        auto _orig = (func_t) gotcha_get_wrappee(_data.wrappee);
        if(!_data.ready)
            return (*_orig)(_args...);

        _data.ready = false;
        static thread_local wrap_type _obj(_data.tool_id, false);
        Ret _ret    = invoke(_obj, _orig, std::forward<Args>(_args)...);
        _data.ready = true;
        return _ret;
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
        return Ret{};
#endif
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename... Args>
    static void wrap_void_op(Args... _args)
    {
        static_assert(N < Nt, "Error! N must be less than Nt!");
#if defined(TIMEMORY_USE_GOTCHA)
        static auto& _data = get_data()[N];
        typedef void (*func_t)(Args...);
        auto _orig      = (func_t) gotcha_get_wrappee(_data.wrappee);
        using wrap_type = tim::component_tuple<operator_type>;

        if(!_data.ready)
            (*_orig)(_args...);
        else
        {
            _data.ready = false;
            static thread_local wrap_type _obj(_data.tool_id, false);
            invoke(_obj, _orig, std::forward<Args>(_args)...);
            _data.ready = true;
        }
#else
        consume_parameters(_args...);
        PRINT_HERE("%s", "should not be here!");
#endif
    }

    //----------------------------------------------------------------------------------//
};
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
//
#if defined(__GNUC__) && (__GNUC__ >= 6)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
//
namespace tim
{
namespace component
{
//======================================================================================//
//
struct malloc_gotcha : base<malloc_gotcha, double>
{
#if defined(TIMEMORY_USE_CUDA)
    static constexpr uintmax_t data_size = 5;
    static constexpr uintmax_t num_alloc = 3;
#else
    static constexpr uintmax_t data_size = 3;
    static constexpr uintmax_t num_alloc = 2;
#endif

    using value_type   = double;
    using this_type    = malloc_gotcha;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using string_hash  = std::hash<std::string>;

    // formatting
    static const short precision = 3;
    static const short width     = 12;

    // required static functions
    static std::string label() { return "malloc_gotcha"; }
    static std::string description() { return "GOTCHA wrapper for memory allocation"; }
    static std::string display_unit() { return "MB"; }
    static int64_t     unit() { return units::megabyte; }
    static value_type  record() { return value_type{ 0.0 }; }

    using base_type::accum;
    using base_type::is_transient;
    using base_type::set_started;
    using base_type::set_stopped;
    using base_type::value;

    template <typename Tp>
    using gotcha_component_type = push_back_t<Tp, this_type>;

    template <typename Tp>
    using gotcha_type = gotcha<data_size, push_back_t<Tp, this_type>, type_list<>>;

    template <typename Tp>
    using component_type = push_back_t<Tp, gotcha_type<Tp>>;

public:
    //----------------------------------------------------------------------------------//

    template <typename Tp, typename... Types>
    static void configure();

    //----------------------------------------------------------------------------------//

    static uintmax_t get_index(uintmax_t _hash)
    {
        uintmax_t idx = std::numeric_limits<uintmax_t>::max();
        for(uintmax_t i = 0; i < get_hash_array().size(); ++i)
        {
            if(_hash == get_hash_array()[i])
                idx = i;
        }
        return idx;
    }

public:
    //----------------------------------------------------------------------------------//

    malloc_gotcha(const std::string& _prefix)
    : prefix_hash(string_hash()(_prefix))
    , prefix_idx(get_index(prefix_hash))
    , prefix(_prefix)
    {
        value = 0.0;
        accum = 0.0;
    }

    malloc_gotcha()                 = default;
    ~malloc_gotcha()                = default;
    malloc_gotcha(const this_type&) = default;
    malloc_gotcha(this_type&&)      = default;
    malloc_gotcha& operator=(const this_type&) = default;
    malloc_gotcha& operator=(this_type&&) = default;

public:
    //----------------------------------------------------------------------------------//

    void start()
    {
        set_started();
        value = record();
    }

    void stop()
    {
        // value should be update via audit in-between start() and stop()
        auto tmp = record();
        accum += (value - tmp);
        value = std::move(std::max(value, tmp));
        set_stopped();
    }

    //----------------------------------------------------------------------------------//

    double get_display() const { return get(); }

    //----------------------------------------------------------------------------------//

    double get() const { return accum / base_type::get_unit(); }

    //----------------------------------------------------------------------------------//

    void audit(const std::string& fname, size_t nbytes)
    {
        DEBUG_PRINT_HERE("%s(%i)", fname.c_str(), (int) nbytes);

        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        DEBUG_PRINT_HERE("hash: %lu, index: %i", (unsigned long) _hash, (int) idx);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::get_label().c_str(),
                       fname.c_str());
            return;
        }

        if(_hash == prefix_hash)
        {
            // malloc
            value = (nbytes);
            accum += (nbytes);
            DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
        }
        else
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> skipped function '%s with hash %llu'\n",
                       this_type::get_label().c_str(), fname.c_str(),
                       (long long unsigned) _hash);
        }
    }

    //----------------------------------------------------------------------------------//

    void audit(const std::string& fname, size_t nmemb, size_t size)
    {
        DEBUG_PRINT_HERE("%s(%i, %i)", fname.c_str(), (int) nmemb, (int) size);

        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::get_label().c_str(),
                       fname.c_str());
            return;
        }

        if(_hash == prefix_hash)
        {
            // calloc
            value = (nmemb * size);
            accum += (nmemb * size);
            DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
        }
        else
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> skipped function '%s with hash %llu'\n",
                       this_type::get_label().c_str(), fname.c_str(),
                       (long long unsigned) _hash);
        }
    }

    //----------------------------------------------------------------------------------//

    void audit(const std::string& fname, void* ptr)
    {
        DEBUG_PRINT_HERE("%s(%p)", fname.c_str(), ptr);

        if(!ptr)
            return;

        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::get_label().c_str(),
                       fname.c_str());
            return;
        }

        // malloc
        if(idx < num_alloc)
        {
            get_allocation_map()[ptr] = value;
            DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
        }
        else
        {
            auto itr = get_allocation_map().find(ptr);
            if(itr != get_allocation_map().end())
            {
                value = itr->second;
                accum += itr->second;
                DEBUG_PRINT_HERE("value: %12.8f, accum: %12.8f", value, accum);
                get_allocation_map().erase(itr);
            }
            else
            {
                if(settings::verbose() > 1 || settings::debug())
                    printf("[%s]> free of unknown pointer size: %p\n",
                           this_type::get_label().c_str(), ptr);
            }
        }
    }

    //----------------------------------------------------------------------------------//

#if defined(TIMEMORY_USE_CUDA)

    //----------------------------------------------------------------------------------//

    void audit(const std::string& fname, void** devPtr, size_t size)
    {
        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::get_label().c_str(),
                       fname.c_str());
            return;
        }

        if(_hash == prefix_hash)
        {
            // malloc
            value = (size);
            accum += (size);
            m_last_addr = devPtr;
        }
        else
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> skipped function '%s with hash %llu'\n",
                       this_type::get_label().c_str(), fname.c_str(),
                       (long long unsigned) _hash);
        }
    }

    //----------------------------------------------------------------------------------//

    void audit(const std::string& fname, cuda::error_t)
    {
        auto _hash = string_hash()(fname);
        auto idx   = get_index(_hash);

        if(idx > get_hash_array().size())
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> unknown function: '%s'\n", this_type::get_label().c_str(),
                       fname.c_str());
            return;
        }

        if(_hash == prefix_hash && idx < num_alloc)
        {
            // cudaMalloc
            if(m_last_addr)
            {
                void* ptr                 = (void*) ((char**) (m_last_addr)[0]);
                get_allocation_map()[ptr] = value;
            }
        }
        else if(_hash == prefix_hash && idx >= num_alloc)
        {
            // cudaFree
        }
        else
        {
            if(settings::verbose() > 1 || settings::debug())
                printf("[%s]> skipped function '%s with hash %llu'\n",
                       this_type::get_label().c_str(), fname.c_str(),
                       (long long unsigned) _hash);
        }
    }

    //----------------------------------------------------------------------------------//

#endif

    //----------------------------------------------------------------------------------//

    void set_prefix(const std::string& _prefix)
    {
        prefix      = _prefix;
        prefix_hash = add_hash_id(prefix);
        for(uintmax_t i = 0; i < get_hash_array().size(); ++i)
        {
            if(prefix_hash == get_hash_array()[i])
                prefix_idx = i;
        }
    }

    //----------------------------------------------------------------------------------//

    this_type& operator+=(const this_type& rhs)
    {
        value += rhs.value;
        accum += rhs.accum;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

    //----------------------------------------------------------------------------------//

    this_type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return *this;
    }

private:
    using alloc_map_t  = std::unordered_map<void*, size_t>;
    using vaddr_map_t  = std::unordered_map<void**, size_t>;
    using hash_array_t = std::array<uintmax_t, data_size>;

    static alloc_map_t& get_allocation_map()
    {
        static thread_local alloc_map_t _instance;
        return _instance;
    }

    static vaddr_map_t& get_void_address_map()
    {
        static thread_local vaddr_map_t _instance;
        return _instance;
    }

    static hash_array_t& get_hash_array()
    {
        static hash_array_t _instance = []() {
#if defined(TIMEMORY_USE_CUDA)
            hash_array_t _tmp = {
                { string_hash()("malloc"),
                  string_hash()("calloc"),
                  string_hash()("cudaMalloc"),
                  string_hash()("free"),
                  string_hash()("cudaFree") }
            };
#else
            hash_array_t _tmp = { { string_hash()("malloc"), string_hash()("calloc"),
                                    string_hash()("free") } };
#endif

            return _tmp;
        }();
        return _instance;
    }

private:
    uintmax_t   prefix_hash = string_hash()("");
    uintmax_t   prefix_idx  = std::numeric_limits<uintmax_t>::max();
    std::string prefix      = "";
#if defined(TIMEMORY_USE_CUDA)
    void** m_last_addr = nullptr;
#endif
};
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_GOTCHA)
//
template <typename Type, typename... Types>
inline void
malloc_gotcha::configure()
{
    // static_assert(!std::is_same<Type, malloc_gotcha>::value,
    //              "Error! Cannot configure with self as the type!");

    using tuple_t           = component_tuple<Type, Types..., malloc_gotcha>;
    using local_gotcha_type = gotcha<data_size, tuple_t, type_list<>>;

    local_gotcha_type::get_default_ready() = false;
    local_gotcha_type::get_initializer()   = []() {
    //
#    if defined(TIMEMORY_USE_CUDA)
        TIMEMORY_C_GOTCHA(local_gotcha_type, 0, malloc);
        TIMEMORY_C_GOTCHA(local_gotcha_type, 1, calloc);
        // TIMEMORY_C_GOTCHA(local_gotcha_type, 2, cudaMalloc);
        local_gotcha_type::template configure<2, cudaError_t, void**, size_t>(
            "cudaMalloc");
        TIMEMORY_C_GOTCHA(local_gotcha_type, 3, free);
        TIMEMORY_C_GOTCHA(local_gotcha_type, 4, cudaFree);
#    else
        TIMEMORY_C_GOTCHA(local_gotcha_type, 0, malloc);
        TIMEMORY_C_GOTCHA(local_gotcha_type, 1, calloc);
        TIMEMORY_C_GOTCHA(local_gotcha_type, 2, free);
#    endif
        //
    };
}
//
#endif
//
}  // namespace component
}  // namespace tim
//
#if defined(__GNUC__) && (__GNUC__ >= 6)
#    pragma GCC diagnostic pop
#endif
