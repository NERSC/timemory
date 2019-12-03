#!/bin/bash

: ${MPI_HEADER:=@MPI_HEADER@}
: ${OUT:=/dev/stdout}
: ${SED:=$(which ssed)}

# if no ssed, use sed
if [ -z "${SED}" ]; then SED=$(which sed); fi

if test $(echo "hello" | ${SED} -r 's/h/t/g' 2> /dev/null); then
    SED="${SED} -r"
fi

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
    local funcs=$(grep -E 'int[ \t].*MPI_' ${MPI_HEADER} \
        | grep '(' \
        | sed 's/OMPI_DECLSPEC//g' \
        | ${SED} 's/[(].*//g' \
        | awk '{print $2}' \
        | grep -E -v 'PMPI_|typedef|OMPI_|MPI_T_|MPI_Pcontrol|MPI_Test|MPI_Type|MPI_Init|MPI_Finalize|MPI_Abort|MPI_DUP|MPI_Address|MPI_Attr|MPI_Errhandler|MPI_Keyval' | sort -u)
    for i in ${funcs}
    do
        if [ -z "$(echo $i | grep MPI_)" ]; then continue; fi
        if [ -n "$(echo $i | grep -E '_c2f|MPI_Fint')" ]; then continue; fi
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
#include <timemory/components/gotcha.hpp>

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
    // provide environment variable for enabling/disabling using custom record types
    if(tim::get_env<bool>("ENABLE_TIMEMORY_MPIP_RECORD_TYPES", false))
    {
        timemory_create_function = (timemory_create_func_t) &create_record;
        timemory_delete_function = (timemory_delete_func_t) &delete_record;
    }

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
    mpip_gotcha_t::get_reject_list() = []() {
        auto reject_list     = tim::get_env<std::string>("TIMEMORY_MPIP_REJECT_LIST", "");
        if(reject_list.length() == 0)
            return stringset_t{};
        auto reject_list_vec = tim::delimit(reject_list);
        stringset_t reject_list_set;
        for(const auto& itr : reject_list_vec)
            reject_list_set.insert(itr);
        return reject_list_set;
    };

    // provide environment variable for selecting wrappers
    mpip_gotcha_t::get_permit_list() = []() {
        auto permit_list     = tim::get_env<std::string>("TIMEMORY_MPIP_PERMIT_LIST", "");
        if(permit_list.length() == 0)
            return stringset_t{};
        auto permit_list_vec = tim::delimit(permit_list);
        stringset_t permit_list_set;
        for(const auto& itr : permit_list_vec)
            permit_list_set.insert(itr);
        return permit_list_set;
    };

    // provide environment variable for enabling/disabling
    if(tim::get_env<bool>("ENABLE_TIMEMORY_MPIP", true))
    {
        mpip_gotcha_t::get_initializer()();
    }
}

void
init_timemory_mpip_tools_()
{
    init_timemory_mpip_tools();
}

}  // extern "C"
EOF
