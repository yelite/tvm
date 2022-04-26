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
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>

#include <map>
#include <string>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace contrib {

// NOTE : this struct should be defined before TVMScriptRuntimeClass
struct ThreadLocalStore {
  tvm::runtime::Module mod;
  static ThreadLocalStore* ThreadLocal() {
    thread_local ThreadLocalStore tls;
    return &tls;
  }
};

class TVMScriptRuntimeClass : public torch::jit::CustomClassHolder {
 public:
  TVMScriptRuntimeClass() {
    std::cout<<"TVMScriptRuntimeClass initializaion"<<std::endl;
    mod_ = ThreadLocalStore::ThreadLocal() -> mod;
  }

  DLTensor forward(const DLTensor inputs) {

    
    return inputs;

  }


 private:

  tvm::runtime::Module mod_;

};


TVM_REGISTER_GLOBAL("tvmtorch.save_runtime_mod").set_body_typed([](tvm::runtime::Module mod) {
  ThreadLocalStore::ThreadLocal()->mod = mod;
});

TORCH_LIBRARY(tvm_torch, m) {
  m.class_<TVMScriptRuntimeClass>("TVMScriptRuntime")
  .def(torch::init<>())
  .def("forward", &TVMScriptRuntimeClass::forward);
}

}
}