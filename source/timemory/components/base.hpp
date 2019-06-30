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

#include "timemory/components/types.hpp"
#include "timemory/macros.hpp"
#include "timemory/policy.hpp"
#include "timemory/serializer.hpp"
#include "timemory/storage.hpp"

//======================================================================================//

namespace tim
{
namespace component
{
template <typename _Tp, typename value_type, typename... _Policies>
struct base : public tim::counted_object<_Tp>
{
    //
    friend class storage<_Tp>;

    friend struct construct<_Tp>;
    friend struct live_count<_Tp>;
    friend struct set_prefix<_Tp>;
    friend struct insert_node<_Tp>;
    friend struct pop_node<_Tp>;
    friend struct record<_Tp>;
    friend struct reset<_Tp>;
    friend struct measure<_Tp>;
    friend struct start<_Tp>;
    friend struct stop<_Tp>;
    friend struct conditional_start<_Tp>;
    friend struct conditional_stop<_Tp>;
    friend struct minus<_Tp>;
    friend struct plus<_Tp>;
    friend struct multiply<_Tp>;
    friend struct divide<_Tp>;
    friend struct print<_Tp>;
    friend struct print_storage<_Tp>;

    template <typename _Up, typename Archive>
    friend struct serialization;

    // template <typename _Up, typename _Op>
    // friend struct pointer_operator<_Up, _Op>;
    // friend struct pointer_deleter<_Tp>;

    static_assert(std::is_pointer<_Tp>::value == false, "Error pointer base type");

public:
    using Type           = _Tp;
    using policy_type    = policy::wrapper<_Policies...>;
    using this_type      = base<_Tp, value_type, _Policies...>;
    using storage_type   = storage<Type>;
    using graph_iterator = typename storage_type::iterator;

    base()                          = default;
    ~base()                         = default;
    explicit base(const this_type&) = default;
    explicit base(this_type&&)      = default;
    base& operator=(const this_type&) = default;
    base& operator=(this_type&&) = default;

private:
    static void serialization_policy() { policy_type::template invoke_serialize<_Tp>(); }
    static void initialize_policy() { policy_type::template invoke_initialize<_Tp>(); }
    static void finalize_policy() { policy_type::template invoke_finalize<_Tp>(); }

public:
    //----------------------------------------------------------------------------------//
    // function operator
    //
    value_type operator()() { return Type::record(); }

    //----------------------------------------------------------------------------------//
    // set the graph node prefix
    //
    void set_prefix(const string_t& _prefix)
    {
        storage_type::instance()->set_prefix(_prefix);
    }

    //----------------------------------------------------------------------------------//
    // insert the node into the graph
    //
    void insert_node(bool& exists, const int64_t& _hashid)
    {
        hashid    = _hashid;
        Type& obj = static_cast<Type&>(*this);
        graph_itr = storage_type::instance()->insert(hashid, obj, exists);
    }

    void insert_node(const string_t& _prefix, const int64_t& _hashid)
    {
        hashid    = _hashid;
        Type& obj = static_cast<Type&>(*this);
        graph_itr = storage_type::instance()->insert(hashid, obj, _prefix);
    }

    //----------------------------------------------------------------------------------//
    // pop the node off the graph
    //
    template <typename U = value_type, enable_if_t<(!std::is_class<U>::value), int> = 0>
    void pop_node()
    {
        Type& obj = graph_itr->obj();
        obj.accum += accum;
        obj.value += value;
        obj.is_transient = is_transient;
        obj.is_running   = false;
        obj.laps += laps;
        graph_itr = storage_type::instance()->pop();
    }

    //----------------------------------------------------------------------------------//
    // pop the node off the graph
    //
    template <typename U = value_type, enable_if_t<(std::is_class<U>::value), int> = 0>
    void pop_node()
    {
        Type& obj = graph_itr->obj();
        Type& rhs = static_cast<Type&>(*this);
        obj += rhs;
        obj.laps += rhs.laps;
        storage_type::instance()->pop();
    }

    //----------------------------------------------------------------------------------//
    // reset the values
    //
    void reset()
    {
        is_running   = false;
        is_transient = false;
        laps         = 0;
        value        = value_type();
        accum        = value_type();
    }

    //----------------------------------------------------------------------------------//
    // just record a measurment
    //
    void measure()
    {
        is_running   = false;
        is_transient = false;
        value        = Type::record();
    }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        ++laps;
        static_cast<Type&>(*this).start();
        set_started();
    }

    //----------------------------------------------------------------------------------//
    // stop
    //
    void stop()
    {
        static_cast<Type&>(*this).stop();
        set_stopped();
    }

    //----------------------------------------------------------------------------------//
    // set the firsts notify that start has been called
    //
    void set_started()
    {
        is_running   = true;
        is_transient = true;
    }

    //----------------------------------------------------------------------------------//
    // set the firsts notify that stop has been called
    //
    void set_stopped()
    {
        is_running   = false;
        is_transient = true;
    }

    //----------------------------------------------------------------------------------//
    // conditional start if not running
    //
    bool conditional_start()
    {
        if(!is_running)
        {
            set_started();
            static_cast<Type&>(*this).start();
            return true;
        }
        return false;
    }

    //----------------------------------------------------------------------------------//
    // conditional stop if running
    //
    bool conditional_stop()
    {
        if(is_running)
        {
            static_cast<Type&>(*this).stop();
            set_stopped();
            return true;
        }
        return false;
    }

    CREATE_STATIC_VARIABLE_ACCESSOR(short, get_precision, precision)
    CREATE_STATIC_VARIABLE_ACCESSOR(short, get_width, width)
    CREATE_STATIC_VARIABLE_ACCESSOR(std::ios_base::fmtflags, get_format_flags,
                                    format_flags)
    CREATE_STATIC_FUNCTION_ACCESSOR(int64_t, get_unit, unit)
    CREATE_STATIC_FUNCTION_ACCESSOR(std::string, get_label, label)
    CREATE_STATIC_FUNCTION_ACCESSOR(std::string, get_description, descript)
    CREATE_STATIC_FUNCTION_ACCESSOR(std::string, get_display_unit, display_unit)

    //----------------------------------------------------------------------------------//
    // comparison operators
    //
    bool operator==(const base<Type>& rhs) const { return (value == rhs.value); }
    bool operator<(const base<Type>& rhs) const { return (value < rhs.value); }
    bool operator>(const base<Type>& rhs) const { return (value > rhs.value); }
    bool operator!=(const base<Type>& rhs) const { return !(*this == rhs); }
    bool operator<=(const base<Type>& rhs) const { return !(*this > rhs); }
    bool operator>=(const base<Type>& rhs) const { return !(*this < rhs); }

    //----------------------------------------------------------------------------------//
    // this_type operators (plain-old data)
    //
    template <typename U = value_type, enable_if_t<(!std::is_class<U>::value), int> = 0>
    Type& operator+=(const this_type& rhs)
    {
        value += rhs.value;
        accum += rhs.accum;
        laps += rhs.laps;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(!std::is_class<U>::value), int> = 0>
    Type& operator-=(const this_type& rhs)
    {
        value -= rhs.value;
        accum -= rhs.accum;
        laps -= rhs.laps;
        if(rhs.is_transient)
            is_transient = rhs.is_transient;
        return static_cast<Type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    // this_type operators (complex data)
    //
    template <typename U = value_type, enable_if_t<(std::is_class<U>::value), int> = 0>
    Type& operator+=(const this_type& rhs)
    {
        laps += rhs.laps;
        return static_cast<Type&>(*this).operator+=(static_cast<const Type&>(rhs));
    }

    template <typename U = value_type, enable_if_t<(std::is_class<U>::value), int> = 0>
    Type& operator-=(const this_type& rhs)
    {
        laps -= rhs.laps;
        return static_cast<Type&>(*this).operator-=(static_cast<const Type&>(rhs));
    }

    //----------------------------------------------------------------------------------//
    // value type operators (plain-old data)
    //
    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value), int> = 0>
    Type& operator+=(const value_type& rhs)
    {
        value += rhs;
        accum += rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value), int> = 0>
    Type& operator-=(const value_type& rhs)
    {
        value -= rhs;
        accum -= rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value), int> = 0>
    Type& operator*=(const value_type& rhs)
    {
        value *= rhs;
        accum *= rhs;
        return static_cast<Type&>(*this);
    }

    template <typename U = value_type, enable_if_t<(std::is_pod<U>::value), int> = 0>
    Type& operator/=(const value_type& rhs)
    {
        value /= rhs;
        accum /= rhs;
        return static_cast<Type&>(*this);
    }

    //----------------------------------------------------------------------------------//
    // value type operators (complex data)
    //
    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value), int> = 0>
    Type& operator+=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator+=(rhs);
    }

    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value), int> = 0>
    Type& operator-=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator-=(rhs);
    }

    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value), int> = 0>
    Type& operator*=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator*=(rhs);
    }

    template <typename U = value_type, enable_if_t<(!std::is_pod<U>::value), int> = 0>
    Type& operator/=(const value_type& rhs)
    {
        return static_cast<Type&>(*this).operator/=(rhs);
    }

    //----------------------------------------------------------------------------------//
    // friend operators
    //
    friend Type operator+(const this_type& lhs, const this_type& rhs)
    {
        return base<Type>(lhs) += rhs;
    }

    friend Type operator-(const this_type& lhs, const this_type& rhs)
    {
        return base<Type>(lhs) -= rhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        auto _value = static_cast<const Type&>(obj).compute_display();
        auto _label = this_type::get_label();
        auto _disp  = this_type::get_display_unit();
        auto _prec  = this_type::get_precision();
        auto _width = this_type::get_width();
        auto _flags = this_type::get_format_flags();

        std::stringstream ss_value;
        std::stringstream ss_extra;
        ss_value.setf(_flags);
        ss_value << std::setw(_width) << std::setprecision(_prec) << _value;
        if(!_disp.empty())
            ss_extra << " " << _disp;
        if(!_label.empty())
            ss_extra << " " << _label;
        os << ss_value.str() << ss_extra.str();

        return os;
    }

    //----------------------------------------------------------------------------------//
    // serialization
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        auto _disp = static_cast<const Type&>(*this).compute_display();
        ar(serializer::make_nvp("is_transient", is_transient),
           serializer::make_nvp("laps", laps), serializer::make_nvp("value", value),
           serializer::make_nvp("accum", accum), serializer::make_nvp("display", _disp));
    }

    const int64_t&    nlaps() const { return laps; }
    const value_type& get_value() const { return value; }
    const value_type& get_accum() const { return accum; }

protected:
    bool           is_running   = false;
    bool           is_transient = false;
    value_type     value        = value_type();
    value_type     accum        = value_type();
    int64_t        hashid       = 0;
    int64_t        laps         = 0;
    graph_iterator graph_itr;
};

//--------------------------------------------------------------------------------------//

}  // component
}  // tim

//======================================================================================//
