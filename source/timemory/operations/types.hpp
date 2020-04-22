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

/**
 * \file timemory/operations/types.hpp
 * \brief Declare the operations types
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/variadic/types.hpp"

#include <functional>
#include <iosfwd>

namespace std
{
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
TSTAG(struct)
tuple_size<::tim::type_list<Types...>>
{
public:
    static constexpr size_t value = sizeof...(Types);
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace std

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct remove_pointers;
//
template <template <typename...> class Tuple, typename... Tp>
struct remove_pointers<Tuple<Tp...>>
{
    using type = Tuple<std::remove_pointer_t<Tp>...>;
};
//
template <typename Tp>
using remove_pointers_t = typename remove_pointers<Tp>::type;
//
//--------------------------------------------------------------------------------------//
//
namespace utility
{
struct stream;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct statistics;
//
//--------------------------------------------------------------------------------------//
//
//                              operations
//
//--------------------------------------------------------------------------------------//
//
//  components that provide the invocation (i.e. WHAT the components need to do)
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Up>
struct is_enabled
{
    // shorthand for available + non-void
    using Vp = typename Up::value_type;
    static constexpr bool value =
        (trait::is_available<Up>::value && !(std::is_same<Vp, void>::value));
};
//
//--------------------------------------------------------------------------------------//
//
template <typename U>
using is_enabled_t = typename is_enabled<U>::type;
//
//--------------------------------------------------------------------------------------//
//
template <typename Up>
struct has_data
{
    // shorthand for non-void
    using Vp                    = typename Up::value_type;
    static constexpr bool value = (!std::is_same<Vp, void>::value);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename V = typename T::value_type>
struct check_record_type
{
    static constexpr bool value =
        (!std::is_same<V, void>::value && is_enabled<T>::value &&
         std::is_same<
             V, typename function_traits<decltype(&T::record)>::result_type>::value);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Up, typename Vp>
struct stats_enabled
{
    using EmptyT = std::tuple<>;

    static constexpr bool value =
        (trait::record_statistics<Up>::value && !(std::is_same<Vp, void>::value) &&
         !(std::is_same<Vp, EmptyT>::value) &&
         !(std::is_same<Vp, statistics<void>>::value) &&
         !(std::is_same<Vp, statistics<EmptyT>>::value));
};
//
//--------------------------------------------------------------------------------------//
//
template <typename U, typename StatsT>
struct enabled_statistics
{
    using EmptyT = std::tuple<>;

    static constexpr bool value =
        (trait::record_statistics<U>::value && !std::is_same<StatsT, EmptyT>::value);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename U>
using has_data_t = typename has_data<U>::type;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct init_storage;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct construct;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct set_prefix;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct set_scope;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct insert_node;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct pop_node;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct record;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct reset;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct measure;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct sample;
//
//--------------------------------------------------------------------------------------//
//
template <typename Ret, typename Lhs, typename Rhs>
struct compose;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct priority_start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct standard_start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct delayed_start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct priority_stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct standard_stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct delayed_stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct mark_begin;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct mark_end;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct store;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct audit;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct plus;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct minus;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct multiply;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct divide;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get_data;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get_labeled_data;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct base_printer;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print_header;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print_statistics;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print_storage;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct add_secondary;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct add_statistics;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct serialization;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, bool Enabled = trait::echo_enabled<T>::value>
struct echo_measurement;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct copy;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct assemble;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct derive;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename Op>
struct pointer_operator;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct pointer_deleter;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct pointer_counter;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename Op>
struct generic_operator;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct generic_deleter;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct generic_counter;
//
//--------------------------------------------------------------------------------------//
//
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct mpi_get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct upc_get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct dmp_get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct print;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct merge;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct flamegraph;
//
//--------------------------------------------------------------------------------------//
//
namespace base
{
//
//--------------------------------------------------------------------------------------//
//
struct TIMEMORY_OPERATIONS_DLL print
{
    using this_type   = print;
    using stream_type = std::shared_ptr<utility::stream>;

    print(const std::string& _label, bool _forced_json)
    : json_forced(_forced_json)
    , label(_label)
    {}

    virtual void setup()        = 0;
    virtual void execute()      = 0;
    virtual void read_json()    = 0;
    virtual void print_dart()   = 0;
    virtual void update_data()  = 0;
    virtual void print_custom() = 0;

    virtual void write(std::ostream& os, stream_type stream);
    virtual void print_cout(stream_type stream);
    virtual void print_text(const std::string& fname, stream_type stream);
    virtual void print_plot(const std::string& fname, const std::string suffix);

    auto get_label() const { return label; }
    auto get_text_output_name() const { return text_outfname; }
    auto get_json_output_name() const { return json_outfname; }
    auto get_json_input_name() const { return json_inpfname; }
    auto get_text_diff_name() const { return text_diffname; }
    auto get_json_diff_name() const { return json_diffname; }

    void set_debug(bool v) { debug = v; }
    void set_update(bool v) { update = v; }
    void set_file_output(bool v) { file_output = v; }
    void set_text_output(bool v) { text_output = v; }
    void set_json_output(bool v) { json_output = v; }
    void set_dart_output(bool v) { dart_output = v; }
    void set_plot_output(bool v) { plot_output = v; }
    void set_verbose(int32_t v) { verbose = v; }
    void set_max_call_stack(int64_t v) { max_call_stack = v; }

    int64_t get_max_depth() const
    {
        return (max_depth > 0) ? max_depth
                               : std::min<int64_t>(max_call_stack, settings::max_depth());
    }

protected:
    // default initialized
    bool    debug          = settings::debug();
    bool    update         = true;
    bool    json_forced    = false;
    bool    file_output    = settings::file_output();
    bool    cout_output    = settings::cout_output();
    bool    json_output    = (settings::json_output() || json_forced) && file_output;
    bool    text_output    = settings::text_output() && file_output;
    bool    dart_output    = settings::dart_output();
    bool    plot_output    = settings::plot_output() && json_output;
    bool    flame_output   = settings::flamegraph_output() && file_output;
    bool    node_init      = dmp::is_initialized();
    int32_t node_rank      = dmp::rank();
    int32_t node_size      = dmp::size();
    int32_t verbose        = settings::verbose();
    int64_t max_depth      = 0;
    int64_t max_call_stack = settings::max_depth();

protected:
    int64_t     data_concurrency  = 1;
    int64_t     input_concurrency = 1;
    std::string label             = "";
    std::string description       = "";
    std::string text_outfname     = "";
    std::string json_outfname     = "";
    std::string json_inpfname     = "";
    std::string text_diffname     = "";
    std::string json_diffname     = "";
    stream_type data_stream       = stream_type{};
    stream_type diff_stream       = stream_type{};
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace base
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print<Tp, true> : public base::print
{
    static constexpr bool has_data = true;
    using base_type                = base::print;
    using this_type                = print<Tp, has_data>;
    using type                     = Tp;
    using storage_type             = impl::storage<Tp, has_data>;
    using result_type              = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = typename storage_type::uintvector_t;
    using callback_type            = std::function<void(this_type*)>;
    using stream_type              = std::shared_ptr<utility::stream>;

    static callback_type& get_default_callback()
    {
        static callback_type _instance = [](this_type*) {};
        return _instance;
    }

    print(const std::string& _label, storage_type* _data)
    : base_type(_label, trait::requires_json<Tp>::value)
    , data(_data)
    {}

    virtual ~print() {}

    virtual void execute()
    {
        if(!data)
            return;

        if(update)
            update_data();
        else
            setup();

        if(node_init && node_rank > 0)
            return;

        if(file_output)
        {
            if(json_output)
                print_json(json_outfname, node_results, data_concurrency);
            if(text_output)
                print_text(text_outfname, data_stream);
            if(plot_output)
                print_plot(json_outfname, "");
        }

        if(cout_output)
            print_cout(data_stream);
        else
            printf("\n");

        if(dart_output)
            print_dart();

        if(!node_input.empty() && !node_delta.empty() && settings::diff_output())
        {
            if(file_output)
            {
                if(json_output)
                    print_json(json_diffname, node_delta, data_concurrency);
                if(text_output)
                    print_text(text_diffname, diff_stream);
                if(plot_output)
                {
                    std::stringstream ss;
                    ss << "Difference vs. " << json_inpfname;
                    if(input_concurrency != data_concurrency)
                    {
                        auto delta_conc = (data_concurrency - input_concurrency);
                        ss << " with " << delta_conc << " "
                           << ((delta_conc > 0) ? "more" : "less") << "threads";
                    }
                    print_plot(json_diffname, ss.str());
                }
            }

            if(cout_output)
                print_cout(diff_stream);
            else
                printf("\n");
        }

        print_custom();
    }

    virtual void update_data();
    virtual void setup();
    virtual void read_json();

    virtual void print_dart();
    virtual void print_custom()
    {
        try
        {
            callback(this);
        } catch(std::exception& e)
        {
            fprintf(stderr, "Exception: %s\n", e.what());
        }
    }

    void write_stream(stream_type& stream, result_type& results);
    void print_json(const std::string& fname, result_type& results, int64_t concurrency);
    auto get_data() const { return data; }
    auto get_node_results() const { return node_results; }
    auto get_node_input() const { return node_input; }
    auto get_node_delta() const { return node_delta; }

    template <typename Archive>
    void print_metadata(true_type, Archive& ar, const Tp& obj);

    template <typename Archive>
    void print_metadata(false_type, Archive& ar, const Tp& obj);

    std::vector<result_node*> get_flattened(result_type& results)
    {
        std::vector<result_node*> flat;
        for(auto& ritr : results)
            for(auto& itr : ritr)
            {
                flat.push_back(&itr);
            }
        return flat;
    }

protected:
    storage_type* data         = nullptr;
    callback_type callback     = get_default_callback();
    result_type   node_results = {};
    result_type   node_input   = {};
    result_type   node_delta   = {};
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print<Tp, false> : base::print
{
    static constexpr bool has_data = false;
    using this_type                = print<Tp, has_data>;
    using storage_type             = impl::storage<Tp, has_data>;

    print(storage_type&);
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
