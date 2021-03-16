
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#######################################################
function(find_and_set_linker use_alternative_linker)
  if(${use_alternative_linker} MATCHES ${IS_FALSE_PATTERN})
    return()
  endif()
  if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    return()
  endif()

  macro(add_to_linker_flags flag)
    string(APPEND CMAKE_EXE_LINKER_FLAGS flag)
    string(APPEND CMAKE_STATIC_LINKER_FLAGS flag)
    string(APPEND CMAKE_SHARED_LINKER_FLAGS flag)
    string(APPEND CMAKE_MODULE_LINKER_FLAGS flag)
    message(STATUS "Added \"${flag}\" to linker flags")
  endmacro(add_to_linker_flags)

  find_program(MOLD_BIN "mold")
  find_program(LLD_BIN "lld")

  if(MOLD_BIN)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.1)
      get_filename_component(MOLD_INSTALLATION_PREFIX "${MOLD_BIN}" DIRECTORY)
      get_filename_component(MOLD_INSTALLATION_PREFIX "${MOLD_INSTALLATION_PREFIX}" DIRECTORY)
      find_path(
        MOLD_LIBEXEC_DIR "ld"
        NO_DEFAULT_PATH
        HINTS "${MOLD_INSTALLATION_PREFIX}"
        PATH_SUFFIXES "libexec/mold" "lib/mold" "lib64/mold"
        NO_CACHE
      )
      if(MOLD_LIBEXEC_DIR)
        add_to_linker_flags(" -B \"${MOLD_LIBEXEC_DIR}\"")
        return()
      endif()
    else()
      add_to_linker_flags(" -fuse-ld=mold")
      return()
    endif()
  endif()

  if(LLD_BIN) 
    add_to_linker_flags(" -fuse-ld=lld")
  endif()
endfunction(find_and_set_linker)
