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

if(USE_VLLM)
  message(STATUS "Build with vllm paged attention kernel.")
  include_directories(src/runtime/contrib/vllm)
  set(CMAKE_CUDA_ARCHITECTURES 80) # without this, cmake tries to compile with compute_52
  tvm_file_glob(GLOB VLLM_CONTRIB_SRC src/runtime/contrib/vllm/*.cu)
  list(APPEND RUNTIME_SRCS ${VLLM_CONTRIB_SRC})
endif(USE_VLLM)
