#!/bin/bash

: ${MPI_HEADER:=@MPI_HEADER@}
: ${OUT:=/dev/stdout}

tolower()
{
    if [ "$(uname)" == "Darwin" ]; then
        echo "$@" | awk -F '\\\|\\\|\\\|' '{print tolower($1)}';
    else
        echo "$@" | awk -F \|~\| '{print tolower($1)}';
    fi
}

get_mpi_functions()
{
    local N=0
    local funcs=$(grep '^int MPI_' ${MPI_HEADER} | grep '(' \
        | sed 's/(/ /g' | awk '{print $2}' \
        | egrep -v 'MPI_T_|MPI_Pcontrol')
    for i in ${funcs}
    do
    	echo "        TIMEMORY_C_GOTCHA(mpip_gotcha_t, ${N}, $i);"
	    N=$((${N}+1))
    done
    echo "SIZE ${N}"
}

GOTCHA_SPEC=$(get_mpi_functions | grep -v "^SIZE")
GOTCHA_SIZE=$(get_mpi_functions | grep "^SIZE" | awk '{print $NF}')

cat <<EOF>> ${OUT}

#include <timemory/library.h>
#include <timemory/timemory.hpp>

#include <memory>
#include <set>
#include <unordered_map>

using namespace tim::component;
using stringset_t   = std::set<std::string>;
using mpi_toolset_t = tim::auto_timer;
using mpip_gotcha_t = tim::component::gotcha<${GOTCHA_SIZE}, mpi_toolset_t>;
using mpip_tuple_t  = tim::component_tuple<tim::auto_timer_tuple_t, mpip_gotcha_t>;
using mpip_list_t   = tim::auto_timer_list_t;
using mpip_hybrid_t = tim::component_hybrid<mpip_tuple_t, mpip_list_t>;
using toolset_ptr_t = std::shared_ptr<mpip_hybrid_t>;
using record_map_t  = std::unordered_map<uint64_t, toolset_ptr_t>;

extern "C"
{
void
create_record(const char* name, uint64_t* id, int, int*)
{
    *id                                     = timemory_get_unique_id();
    auto obj                                = toolset_ptr_t(new mpip_hybrid_t(name));
    timemory_tl_static<record_map_t>()[*id] = std::move(obj);
}

void
delete_record(uint64_t nid)
{
    auto& record_ids = timemory_tl_static<record_map_t>();
    // erase key from map which stops recording when object is destroyed
    record_ids.erase(nid);
}

void
init_timemory_mpip_tools()
{
    static bool is_initialized = false;
    if(is_initialized)
        return;
    is_initialized = true;

    // initialize manager
    auto manager = tim::manager::instance();
    tim::consume_parameters(manager);

    printf("[timemory]> %s...\n", __FUNCTION__);

    // activate gotcha without start/stop
    mpip_gotcha_t::get_default_ready() = true;

    // generate the gotcha wrappers
    mpip_gotcha_t::get_initializer() = []()
    {
${GOTCHA_SPEC}
    };

    // provide environment variable for suppressing wrappers
    mpip_gotcha_t::get_blacklist() = []() {
        auto blacklist     = tim::get_env<std::string>("TIMEMORY_MPIP_BLACKLIST", "");
        auto blacklist_vec = tim::delimit(blacklist);
        stringset_t blacklist_set;
        for(const auto& itr : blacklist_vec)
            blacklist_set.insert(itr);
        return blacklist_set;
    };

    // provide environment variable for enabling/disabling
    if(tim::get_env<bool>("ENABLE_TIMEMORY_MPIP", true))
    {
        timemory_create_function = (timemory_create_func_t) &create_record;
        timemory_delete_function = (timemory_delete_func_t) &delete_record;
        mpip_gotcha_t::get_initializer()();
        // static auto hold = timemory_get_begin_record(__FUNCTION__);
        // tim::consume_parameters(hold);
    }
}

void
init_timemory_mpip_tools_()
{
    init_timemory_mpip_tools();
}

}  // extern "C"
EOF
