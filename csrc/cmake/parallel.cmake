# Build concurrency, derived from the machine instead of hardcoded.
#
# Two independent knobs multiply into the build's CPU load:
#   * how many translation units nvcc/HOST compilers may work on at once
#     (the build tool's -j, capped here with a Ninja job pool), and
#   * how many threads ONE nvcc spends inside a TU (--threads).
# A fixed --threads=4 on top of an unbounded `-j` launches 4x as many
# runnable compilers as the box has cores; the machine then throttles and
# the build ends up slower than the bounded one. Both knobs are derived from
# the topology here and both stay overridable:
#
#   -DASTRAI_BUILD_CPU=<logical cores to spend>    (default: 3/4 of the box)
#   -DASTRAI_BUILD_JOBS=<concurrent compile TUs>   (default: budget / 2, <= 8)
#   -DASTRAI_NVCC_THREADS=<threads per nvcc>       (default: budget / jobs, <= 8)
#
# Each knob also reads the same-named environment variable; an explicit -D
# wins over the environment, the environment wins over the derived default.

foreach(_knob ASTRAI_BUILD_CPU ASTRAI_BUILD_JOBS ASTRAI_NVCC_THREADS)
    if(NOT DEFINED ${_knob} AND DEFINED ENV{${_knob}})
        set(${_knob} "$ENV{${_knob}}" CACHE STRING "AstrAI build concurrency")
    endif()
endforeach()

cmake_host_system_information(RESULT _astrai_cores QUERY NUMBER_OF_LOGICAL_CORES)
cmake_host_system_information(RESULT _astrai_mem_mb QUERY TOTAL_PHYSICAL_MEMORY)
if(NOT _astrai_cores OR _astrai_cores LESS 1)
    set(_astrai_cores 4) # unknown topology: a conservative floor
endif()

# Leave a quarter of the machine for everything else in the session.
if(NOT ASTRAI_BUILD_CPU)
    math(EXPR ASTRAI_BUILD_CPU "${_astrai_cores} - ${_astrai_cores} / 4")
    if(ASTRAI_BUILD_CPU LESS 1)
        set(ASTRAI_BUILD_CPU 1)
    endif()
endif()

# Concurrent TUs: half the budget (a compile is not purely CPU-bound), <= 8.
math(EXPR _astrai_jobs "${ASTRAI_BUILD_CPU} / 2")
if(_astrai_jobs LESS 1)
    set(_astrai_jobs 1)
elseif(_astrai_jobs GREATER 8)
    set(_astrai_jobs 8)
endif()

# One nvcc front-end with its cicc child is ~1.5-2GB resident; do not swap.
if(_astrai_mem_mb)
    math(EXPR _astrai_jobs_by_mem "${_astrai_mem_mb} / 4096")
    if(_astrai_jobs_by_mem LESS 1)
        set(_astrai_jobs_by_mem 1)
    endif()
    if(_astrai_jobs GREATER _astrai_jobs_by_mem)
        set(_astrai_jobs ${_astrai_jobs_by_mem})
    endif()
endif()

if(NOT ASTRAI_BUILD_JOBS)
    set(ASTRAI_BUILD_JOBS ${_astrai_jobs} CACHE STRING "Concurrent compile jobs" FORCE)
endif()

# Threads inside one nvcc: fill the remaining budget, bounded by cicc's own
# useful ceiling (8).
math(EXPR _astrai_threads "${ASTRAI_BUILD_CPU} / ${ASTRAI_BUILD_JOBS}")
if(_astrai_threads LESS 1)
    set(_astrai_threads 1)
elseif(_astrai_threads GREATER 8)
    set(_astrai_threads 8)
endif()
if(NOT ASTRAI_NVCC_THREADS)
    set(ASTRAI_NVCC_THREADS ${_astrai_threads} CACHE STRING "Threads per nvcc" FORCE)
endif()

# Ninja honours the pool even when the caller passes a larger `-j`; other
# generators keep their own -j and only get the recommended value printed.
if(CMAKE_GENERATOR MATCHES "Ninja")
    set(CMAKE_JOB_POOLS compile=${ASTRAI_BUILD_JOBS})
    set(ASTRAI_JOB_POOL compile)
endif()

message(STATUS
    "AstrAI build concurrency: ${ASTRAI_BUILD_JOBS} compile jobs x "
    "${ASTRAI_NVCC_THREADS} nvcc threads (budget ${ASTRAI_BUILD_CPU} of "
    "${_astrai_cores} cores)")
if(NOT ASTRAI_JOB_POOL)
    message(STATUS
        "AstrAI: '${CMAKE_GENERATOR}' has no compile job pool; pass "
        "--parallel ${ASTRAI_BUILD_JOBS} to the build step to match")
endif()
