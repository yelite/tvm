/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <dlpack/dlpack.h>
#include <torch/custom_class.h>
#include <torch/script.h>


#include <map>
#include <string>
#include <vector>

#include "../utils.h"

class TVMScriptRuntimeClass : public torch::jit::CustomClassHolder {
 public:
  TVMScriptRuntimeClass(const int64_t num_inputs, const int64_t num_outputs,
                       const std::string& device) 
  : num_inputs_(num_inputs), num_outputs_(num_outputs), device_(device) {
    std::cout<<"TVM scirpte init"<<std::endl;
      }

    void testfunc() {
        std::cout<<"test func"<<std::endl;
    }

 private:
//   runtime::Module module;
  const int64_t num_inputs_;
  const int64_t num_outputs_;
  const std::string& device_;

};


TORCH_LIBRARY(tvm_torch, m) {
  m.class_<TVMScriptRuntimeClass>("TVMScriptRuntime")
  .def(torch::init<const int64_t, const int64_t, const std::string&>())
  .def("testfunc", &TVMScriptRuntimeClass::testfunc);
}
