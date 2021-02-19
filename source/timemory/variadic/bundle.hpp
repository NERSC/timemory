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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include "timemory/general/source_location.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/transient_function.hpp"
#include "timemory/variadic/base_bundle.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>

namespace tim
{
//
template <typename...>
class bundle;
/// \class tim::bundle
/// \tparam Tag unique identifying type for the bundle which when \ref
/// tim::trait::is_available<Tag> is false at compile-time or \ref
/// tim::trait::runtime_enabled<Tag>() is false at runtime, then none of the components
/// will be collected
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper which combines the features of \ref
/// tim::component_tuple<T...> and \ref tim::component_list<U..>. The "T" types
/// (compile-time fixed, allocated on stack) should be specified as usual, the "U" types
/// (runtime-time optional, allocated on the heap) should be specified as a pointer.
/// Initialization of the optional types is similar to \ref tim::auto_list<U...> but no
/// environment variable is built-in since, ideally, this environment variable should be
/// customized based on the Tag template parameter.
///
/// See also: \ref tim::auto_bundle.
/// The primary difference b/t the "component_*" and "auto_*" is that the latter
/// used the constructor/destructor to call start and stop and is thus easier to
/// just copy-and-paste into different places. However, the former is better suited for
/// special configuration, data-access, etc.
///
template <typename Tag, typename BundleT, typename TupleT>
class bundle<Tag, BundleT, TupleT>
: public api_bundle<Tag, typename TupleT::available_type>
{
    template <typename U>
    using remove_pointer_decay_t = std::remove_pointer_t<decay_t<U>>;

    template <typename... Tp>
    friend class impl::base_bundle;

    template <typename... Tp>
    friend class auto_base_bundle;

    template <typename... Tp>
    friend class auto_bundle;

    template <typename... Tp>
    friend class auto_tuple;

    template <typename... Tp>
    friend class auto_list;

#if defined(TIMEMORY_USE_DEPRECATED)
    template <typename TupleC, typename ListC>
    friend class component_hybrid;

    template <typename TupleC, typename ListC>
    friend class auto_hybrid;
#endif

    using internal_tag = tim::variadic::impl::internal_tag;

protected:
    using apply_v        = apply<void>;
    using bundle_type    = api_bundle<Tag, typename TupleT::available_type>;
    using string_t       = typename bundle_type::string_t;
    using reference_type = typename TupleT::reference_type;

public:
    using captured_location_t = source_location::captured;
    using this_type           = typename TupleT::template this_type<BundleT>;
    using type                = typename TupleT::template type<BundleT>;
    using component_type      = typename TupleT::template component_type<BundleT>;
    using data_type           = typename TupleT::template data_type<Tag>;
    using type_list_type      = typename TupleT::type_list_type;
    using value_type          = data_type;
    using size_type           = typename bundle_type::size_type;
    using initializer_type    = std::function<void(this_type&)>;
    using transient_func_t    = utility::transient_function<void(this_type&)>;

public:
    static initializer_type& get_initializer();

    template <typename T, typename... U>
    using quirk_config = tim::variadic::impl::quirk_config<T, reference_type, U...>;

public:
    bundle();

    template <typename... T>
    explicit bundle(const string_t& _key, quirk::config<T...>,
                    transient_func_t = get_initializer());

    template <typename... T>
    explicit bundle(const captured_location_t& _loc, quirk::config<T...>,
                    transient_func_t = get_initializer());

    template <typename... T>
    explicit bundle(const string_t& _key, bool _store, quirk::config<T...>,
                    transient_func_t = get_initializer());

    template <typename... T>
    explicit bundle(const captured_location_t& _loc, bool _store, quirk::config<T...>,
                    transient_func_t = get_initializer());

    explicit bundle(size_t _hash, bool _store = true,
                    scope::config _scope = scope::get_default(),
                    transient_func_t     = get_initializer());

    explicit bundle(const string_t& _key, bool _store = true,
                    scope::config _scope = scope::get_default(),
                    transient_func_t     = get_initializer());

    explicit bundle(const captured_location_t& _loc, bool _store = true,
                    scope::config _scope = scope::get_default(),
                    transient_func_t     = get_initializer());

    explicit bundle(size_t _hash, scope::config _scope,
                    transient_func_t = get_initializer());

    explicit bundle(const string_t& _key, scope::config _scope,
                    transient_func_t = get_initializer());

    explicit bundle(const captured_location_t& _loc, scope::config _scope,
                    transient_func_t = get_initializer());

    ~bundle();
    bundle(const bundle& rhs);
    bundle(bundle&&) noexcept = default;

    // this_type operators
    bundle& operator=(const bundle& rhs);
    bundle& operator=(bundle&&) noexcept = default;

    bundle& operator-=(const bundle& rhs);
    bundle& operator+=(const bundle& rhs);

    // generic operators
    template <typename Op>
    bundle& operator-=(Op&& rhs);
    template <typename Op>
    bundle& operator+=(Op&& rhs);
    template <typename Op>
    bundle& operator*=(Op&& rhs);
    template <typename Op>
    bundle& operator/=(Op&& rhs);

    // friend operators
    friend bundle operator+(const bundle& lhs, const bundle& rhs)
    {
        return bundle{ lhs } += rhs;
    }

    friend bundle operator-(const bundle& lhs, const bundle& rhs)
    {
        return bundle{ lhs } -= rhs;
    }

    template <typename Op>
    friend bundle operator*(const bundle& lhs, Op&& rhs)
    {
        return bundle{ lhs } *= std::forward<Op>(rhs);
    }

    template <typename Op>
    friend bundle operator/(const bundle& lhs, Op&& rhs)
    {
        return bundle{ lhs } /= std::forward<Op>(rhs);
    }

    friend std::ostream& operator<<(std::ostream& os, const bundle& obj)
    {
        obj.print<true, true>(os);
        return os;
    }

    bundle clone(bool store, scope::config _scope = scope::get_default());

public:
    /// Query at compile-time whether a user_bundle exists in the set of components.
    /// user_bundle are more restricted versions of component bundlers but allow
    /// runtime insertion of components.
    static constexpr bool has_user_bundle() { return bundle_type::has_user_bundle_v; }

    /// Query at compile-time whether initialization can occur on the stack
    template <typename U>
    static constexpr bool can_stack_init()
    {
        using T = remove_pointer_decay_t<U>;
        return trait::is_available<T>::value && is_one_of<T, data_type>::value;
    }

    /// Query at compile-time whether initialization can occur on the heap
    template <typename U>
    static constexpr bool can_heap_init()
    {
        using T = remove_pointer_decay_t<U>;
        return trait::is_available<T>::value && is_one_of<T*, data_type>::value;
    }

    /// Query at compile-time whether initialization can occur via a placement new.
    /// Placement new init allows for the combination of optional initialization without
    /// a heap allocation. Not currently available.
    template <typename U>
    static constexpr bool can_placement_init()
    {
        return false;
    }

    /// Query at compile-time whether the specified type can be initialized
    template <typename U>
    static constexpr bool can_init()
    {
        using T = remove_pointer_decay_t<U>;
        return can_stack_init<T>() || can_heap_init<T>() || can_placement_init<T>() ||
               has_user_bundle();
    }

    /// Query at compile-time whether initialization will occur on the heap.
    /// `can_heap_init<T>() && !will_heap_init<T>()` will indicate that a
    /// stack-allocated instance of the same type exists.
    template <typename U>
    static constexpr bool will_heap_init()
    {
        using T = remove_pointer_decay_t<U>;
        return can_init<T>() && !can_stack_init<T>();
    }

    /// Query at compile-time whether initialization will happen with an opaque wrapper
    /// (i.e. via user bundle). In this situation, initialization arguments are ignored
    /// and the component will only be initialized with the \ref tim::scope::config of the
    /// bundle.
    template <typename U>
    static constexpr bool will_opaque_init()
    {
        using T = remove_pointer_decay_t<U>;
        return has_user_bundle() && !can_stack_init<T>() && !can_heap_init<T>();
    }

public:
    /// requests the component initialize their storage
    static void init_storage();

    /// tells each component to push itself into the call-stack hierarchy
    this_type& push();

    /// tells each component to pop itself off of the call-stack hierarchy
    this_type& pop();

    /// selective push
    template <typename... Tp>
    this_type& push(mpl::piecewise_select<Tp...>);

    /// selective push with scope configuration
    template <typename... Tp>
    this_type& push(mpl::piecewise_select<Tp...>, scope::config);

    /// selective pop
    template <typename... Tp>
    this_type& pop(mpl::piecewise_select<Tp...>);

    /// requests each component record a measurment
    template <typename... Args>
    this_type& measure(Args&&...);

    /// requests each component take a sample (if supported)
    template <typename... Args>
    this_type& sample(Args&&...);

    /// invokes start on all the components
    template <typename... Args>
    this_type& start(Args&&...);

    /// invokes stop on all the components
    template <typename... Args>
    this_type& stop(Args&&...);

    /// requests each component perform a measurement
    template <typename... Args>
    this_type& record(Args&&...);

    /// invokes reset member function on all the components
    template <typename... Args>
    this_type& reset(Args&&...);

    /// variant of start() which excludes push()
    template <typename... Args>
    this_type& start(mpl::lightweight, Args&&...);

    /// variant of stop() which excludes pop()
    template <typename... Args>
    this_type& stop(mpl::lightweight, Args&&...);

    /// variant of start() which only gets applied to Tp types
    template <typename... Tp, typename... Args>
    this_type& start(mpl::piecewise_select<Tp...>, Args&&...);

    /// variant of stop() which only gets applied to Tp types
    template <typename... Tp, typename... Args>
    this_type& stop(mpl::piecewise_select<Tp...>, Args&&...);

    using bundle_type::get_prefix;
    using bundle_type::get_scope;
    using bundle_type::get_store;
    using bundle_type::hash;
    using bundle_type::key;
    using bundle_type::laps;
    using bundle_type::prefix;
    using bundle_type::rekey;
    using bundle_type::size;
    using bundle_type::store;

    /// query the number of (compile-time) fixed components
    static constexpr uint64_t fixed_count();

    /// query the number of (run-time) optional components
    static constexpr uint64_t optional_count();

    /// number of objects that will be performing measurements
    uint64_t count();

    /// when chaining together operations, this function enables executing a function
    /// inside the chain
    template <typename FuncT, typename... Args>
    decltype(auto) execute(FuncT&& func, Args&&... args);

    /// construct the objects that have constructors with matching arguments
    template <typename... Args>
    this_type& construct(Args&&... _args);

    /// provide preliminary info to the objects with matching arguments. This is typically
    /// used to notify a component that it has been bundled alongside another component
    /// that it can extract data from.
    template <typename... Args>
    this_type& assemble(Args&&... _args);

    /// provide conclusive info to the objects with matching arguments. This is typically
    /// used by components to extract data from another component it has been bundled
    /// alongside, e.g. the cpu_util component can extract data from \ref
    /// tim::component::wall_clock and \ref tim::component::cpu_clock
    template <typename... Args>
    this_type& derive(Args&&... _args);

    /// mark an atomic event
    template <typename... Args>
    this_type& mark(Args&&... _args);

    /// mark a beginning position in the execution (typically used by asynchronous
    /// structures)
    template <typename... Args>
    this_type& mark_begin(Args&&... _args);

    /// mark a beginning position in the execution (typically used by asynchronous
    /// structures)
    template <typename... Args>
    this_type& mark_end(Args&&... _args);

    /// store a value
    template <typename... Args>
    this_type& store(Args&&... _args);

    /// allow the components to inspect the incoming arguments before start
    /// or out-going return value before returning (typically using in GOTCHA components)
    template <typename... Args>
    this_type& audit(Args&&... _args);

    /// perform an add_secondary operation. This operation allows components to add
    /// additional entries to storage which are their direct descendant
    template <typename... Args>
    this_type& add_secondary(Args&&... _args);

    /// generic member function for invoking user-provided operations
    /// \tparam OpT Operation struct
    template <template <typename> class OpT, typename... Args>
    this_type& invoke(Args&&... _args);

    /// generic member function for invoking user-provided operations on a specific
    /// set of component types
    /// \tparam OpT Operation struct
    template <template <typename> class OpT, typename... Tp, typename... Args>
    this_type& invoke(mpl::piecewise_select<Tp...>, Args&&... _args);

    template <bool PrintPrefix = true, bool PrintLaps = true>
    this_type& print(std::ostream& os, bool _endl = false) const;

    /// returns a tuple of invoking get() on all the components
    template <typename... Args>
    auto get(Args&&...) const;

    /// returns a tuple of the component label + invoking get() on all the components
    template <typename... Args>
    auto get_labeled(Args&&...) const;

    /// returns a reference to the underlying tuple of components
    data_type& data();

    /// returns a const reference to the underlying tuple of components
    const data_type& data() const;

    /// get a component from the bundle
    template <typename U>
    decltype(auto) get();

    /// get a component from the bundle
    template <typename U>
    decltype(auto) get() const;

    /// performs an opaque search. Opaque searches are generally provided by user_bundles
    /// with a functor such as this:
    ///
    /// \code{.cpp}
    /// auto _get = [](void* v_this, void*& ptr, size_t _hash) {
    /// {
    ///     if(!ptr && v_this && _hash == typeid_hash<Tp>())
    ///     {
    ///         Tp* _this = static_cast<Tp*>(v_this);
    ///         _this->get(ptr, _hash);
    ///     }
    ///     return ptr;
    /// };
    /// \endcode
    ///
    /// And the component provides this function:
    ///
    /// \code{.cpp}
    /// template <typename Tp, typename Value>
    /// void
    /// base<Tp, Value>::get(void*& ptr, size_t _hash) const
    /// {
    ///     if(!ptr && _hash == typeid_hash<Tp>())
    ///         ptr = reinterpret_cast<void*>(const_cast<base_type*>(this));
    /// }
    /// \endcode
    ///
    this_type& get(void*& ptr, size_t _hash) const;

    /// this is a simple alternative to get<T>() when used from SFINAE in operation
    /// namespace which has a struct get also templated. Usage there can cause error
    /// with older compilers
    template <typename U>
    auto get_component(
        enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                        is_one_of<remove_pointer_decay_t<U>, data_type>::value,
                    int> = 0);

    template <typename U>
    auto get_component(
        enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                        is_one_of<remove_pointer_decay_t<U>*, data_type>::value,
                    int> = 0);

    /// returns a reference from a stack component instead of a pointer
    template <typename U>
    auto& get_reference(
        enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                        is_one_of<remove_pointer_decay_t<U>, data_type>::value,
                    int> = 0);

    /// returns a reference from a heap component instead of a pointer
    template <typename U>
    auto& get_reference(
        enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                        is_one_of<remove_pointer_decay_t<U>*, data_type>::value,
                    int> = 0);

    /// create an optional type that is in variadic list AND is available AND
    /// accepts arguments
    template <typename U, typename T = remove_pointer_decay_t<U>, typename... Args>
    enable_if_t<will_heap_init<T>() && !will_opaque_init<T>(), bool> init(
        Args&&... _args, enable_if_t<std::is_constructible<T, Args...>::value, int> = 0)
    {
        T*& _obj = std::get<index_of<T*, data_type>::value>(m_data);
        if(!_obj)
        {
            if(settings::debug())
            {
                printf("[bundle::init]> initializing type '%s'...\n",
                       demangle<T>().c_str());
            }
            _obj = new T(std::forward<Args>(_args)...);
            set_prefix(_obj, internal_tag{});
            set_scope(_obj, internal_tag{});
            return true;
        }

        static std::atomic<int> _count(0);
        if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
        {
            std::string _id = demangle<T>();
            fprintf(stderr,
                    "[bundle::init]> skipping re-initialization of type"
                    " \"%s\"...\n",
                    _id.c_str());
        }

        return false;
    }

    /// create an optional type that is in variadic list AND is available but is not
    /// constructible with provided arguments
    template <typename U, typename T = remove_pointer_decay_t<U>, typename... Args>
    enable_if_t<will_heap_init<T>() && !will_opaque_init<T>(), bool> init(
        Args&&..., enable_if_t<!std::is_constructible<T, Args...>::value &&
                                   std::is_default_constructible<T>::value,
                               long> = 0)
    {
        T*& _obj = std::get<index_of<T*, data_type>::value>(m_data);
        if(!_obj)
        {
            if(settings::debug())
            {
                fprintf(stderr, "[bundle::init]> initializing type '%s'...\n",
                        demangle<T>().c_str());
            }
            _obj = new T{};
            set_prefix(_obj, internal_tag{});
            set_scope(_obj, internal_tag{});
            return true;
        }

        static std::atomic<int> _count(0);
        if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
        {
            fprintf(stderr,
                    "[bundle::init]> skipping re-initialization of type "
                    "\"%s\"...\n",
                    demangle<T>().c_str());
        }

        return false;
    }

    /// try to re-create a stack object with provided arguments
    template <typename U, typename T = remove_pointer_decay_t<U>, typename... Args>
    enable_if_t<can_stack_init<T>(), bool> init(Args&&... _args)
    {
        IF_CONSTEXPR(can_heap_init<T>())
        {
            static std::atomic<int> _count(0);
            if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
            {
                fprintf(stderr,
                        "[bundle::init]> type exists as a heap-allocated instance "
                        "and stack-allocated instance: \"%s\"...\n",
                        demangle<T>().c_str());
            }
        }
        T& _obj = std::get<index_of<T, data_type>::value>(m_data);
        operation::construct<T>{ _obj, std::forward<Args>(_args)... };
        set_prefix(&_obj, internal_tag{});
        set_scope(&_obj, internal_tag{});
        return true;
    }

    /// if a type is not in variadic list but a \ref tim::component::user_bundle is
    /// available, add it in there
    template <typename U, typename T = remove_pointer_decay_t<U>, typename... Args>
    enable_if_t<will_opaque_init<T>(), bool> init(Args&&...)
    {
        using bundle_t = decay_t<decltype(
            std::get<0>(std::declval<typename bundle_type::user_bundle_types>()))>;
        static_assert(trait::is_user_bundle<bundle_t>::value, "Error! Not a user_bundle");
        this->init<bundle_t>();
        auto* _bundle = this->get<bundle_t>();
        if(_bundle)
        {
            _bundle->insert(component::factory::get_opaque<T>(m_scope),
                            component::factory::get_typeids<T>());
            return true;
        }
        return false;
    }

    /// do nothing if type not available, not one of the variadic types, and there
    /// is no user bundle available
    template <typename U, typename T = remove_pointer_decay_t<U>, typename... Args>
    enable_if_t<!trait::is_available<T>::value ||
                    !(is_one_of<T*, data_type>::value || is_one_of<T, data_type>::value ||
                      bundle_type::has_user_bundle_v),
                bool>
    init(Args&&...)
    {
        return false;
    }

    /// \brief variadic initialization
    /// \tparam T components to initialize
    /// \tparam Args arguments to pass to the construction of the component
    template <typename... T, typename... Args>
    std::array<bool, sizeof...(T)> initialize(Args&&... args);

    /// delete any optional types currently allocated
    template <typename... Tail>
    this_type& disable();

    /// apply a member function to a stack type that is in variadic list AND is available
    template <typename T, typename Func, typename... Args,
              enable_if_t<is_one_of<T, data_type>::value, int> = 0>
    this_type& type_apply(Func&& _func, Args&&... _args)
    {
        auto* _obj = get<T>();
        ((*_obj).*(_func))(std::forward<Args>(_args)...);
        return get_this_type();
    }

    /// apply a member function to either a heap type or a type that is in a user_bundle
    template <typename T, typename Func, typename... Args,
              enable_if_t<trait::is_available<T>::value, int> = 0>
    this_type& type_apply(Func&& _func, Args&&... _args);

    /// ignore applying a member function because the type is not present
    template <typename T, typename Func, typename... Args,
              enable_if_t<!trait::is_available<T>::value, int> = 0>
    this_type& type_apply(Func&&, Args&&...);

    void       set_prefix(const string_t&) const;
    this_type& set_prefix(size_t) const;
    this_type& set_prefix(captured_location_t) const;
    this_type& set_scope(scope::config);

    const data_type& get_data() const { return m_data; }

protected:
    template <typename T>
    void set_scope(T* obj, internal_tag) const;

    template <typename T>
    void set_prefix(T* obj, internal_tag) const;

protected:
    // objects
    using bundle_type::m_config;
    using bundle_type::m_enabled;
    using bundle_type::m_hash;
    using bundle_type::m_is_active;
    using bundle_type::m_is_pushed;
    using bundle_type::m_laps;
    using bundle_type::m_scope;
    using bundle_type::m_store;
    mutable data_type m_data{};

private:
    this_type& get_this_type() { return static_cast<this_type&>(*this); }
    this_type& get_this_type() const
    {
        return const_cast<this_type&>(static_cast<const this_type&>(*this));
    }

    static this_type*& get_last_instance()
    {
        static thread_local this_type* _instance = nullptr;
        return _instance;
    }

    static void update_last_instance(this_type*  _new_instance,
                                     this_type*& _old_instance = get_last_instance(),
                                     bool        _stop_last    = false)
    {
        if(_stop_last && _old_instance && _old_instance != _new_instance)
            _old_instance->stop();
        _old_instance = _new_instance;
    }

public:
    // archive serialization
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int);
};
//
//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
constexpr uint64_t
bundle<Tag, BundleT, TupleT>::fixed_count()
{
    return (size() -
            mpl::get_tuple_size<
                typename get_true_types<std::is_pointer, data_type>::type>::value);
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
constexpr uint64_t
bundle<Tag, BundleT, TupleT>::optional_count()
{
    return mpl::get_tuple_size<
        typename get_true_types<std::is_pointer, data_type>::type>::value;
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename FuncT, typename... Args>
decltype(auto)
bundle<Tag, BundleT, TupleT>::execute(FuncT&& func, Args&&... args)
{
    return mpl::execute(get_this_type(),
                        std::forward<FuncT>(func)(std::forward<Args>(args)...));
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
decltype(auto)
bundle<Tag, BundleT, TupleT>::get()
{
    return tim::variadic::impl::get<U, Tag>(m_data);
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
decltype(auto)
bundle<Tag, BundleT, TupleT>::get() const
{
    return tim::variadic::impl::get<U, Tag>(m_data);
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
auto
bundle<Tag, BundleT, TupleT>::get_component(
    enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                    is_one_of<remove_pointer_decay_t<U>, data_type>::value,
                int>)
{
    return get<remove_pointer_decay_t<U>>();
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
auto
bundle<Tag, BundleT, TupleT>::get_component(
    enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                    is_one_of<remove_pointer_decay_t<U>*, data_type>::value,
                int>)
{
    return get<remove_pointer_decay_t<U>>();
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
auto&
bundle<Tag, BundleT, TupleT>::get_reference(
    enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                    is_one_of<remove_pointer_decay_t<U>, data_type>::value,
                int>)
{
    return std::get<index_of<remove_pointer_decay_t<U>, data_type>::value>(m_data);
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
auto&
bundle<Tag, BundleT, TupleT>::get_reference(
    enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                    is_one_of<remove_pointer_decay_t<U>*, data_type>::value,
                int>)
{
    return std::get<index_of<remove_pointer_decay_t<U>*, data_type>::value>(m_data);
}
//
//----------------------------------------------------------------------------------//
//
//
//----------------------------------------------------------------------------------//
//
/*
template <typename Tag, typename BundleT, typename TupleT
          >
class bundle<Tag, BundleT, TransformT<>, std::tuple<Types...>>
: public bundle<Tag, BundleT, TupleT>
{
    TIMEMORY_DEFAULT_OBJECT(bundle)

    template <typename... Args>
    bundle(Args&&... args)
    : bundle<Tag, BundleT, TupleT>{ std::forward<Args>(args)... }
    {}
};
//
//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT
          >
class bundle<Tag, BundleT, TransformT<>, type_list<Types...>>
: public bundle<Tag, BundleT, TupleT>
{
    TIMEMORY_DEFAULT_OBJECT(bundle)

    template <typename... Args>
    bundle(Args&&... args)
    : bundle<Tag, BundleT, TupleT>{ std::forward<Args>(args)... }
    {}
};
*/
}  // namespace tim

#include "timemory/variadic/bundle.cpp"
