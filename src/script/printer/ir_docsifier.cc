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
#include "./ir_docsifier.h"

#include "tvm/runtime/container/base.h"

namespace tvm {
namespace script {
namespace printer {

IRDocsifier::IRDocsifier(Map<String, String> ir_prefix) {
  auto n = make_object<IRDocsifierNode>();
  n->ir_prefix = std::move(ir_prefix);
  data_ = std::move(n);
}

IRDocsifier::FType& IRDocsifier::vtable() {
  static IRDocsifier::FType inst;
  return inst;
}

IRDocsifier::FType& IRDocsifier::var_type_vtable() {
  static IRDocsifier::FType inst;
  return inst;
}

TVM_REGISTER_NODE_TYPE(IRDocsifierNode);

}  // namespace printer
}  // namespace script
}  // namespace tvm
