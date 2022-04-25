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
#include "./frame.h"

namespace tvm {
namespace script {
namespace printer {

MetadataFrame::MetadataFrame(SymbolTable sym) {
  ObjectPtr<MetadataFrameNode> n = make_object<MetadataFrameNode>();
  n->objs.clear();
  n->sym = std::move(sym);
  n->metadata.clear();
  data_ = std::move(n);
}

VarDefFrame::VarDefFrame(SymbolTable sym) {
  ObjectPtr<VarDefFrameNode> n = make_object<VarDefFrameNode>();
  n->objs.clear();
  n->sym = std::move(sym);
  n->stmts.clear();
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(FrameNode);
TVM_REGISTER_NODE_TYPE(MetadataFrameNode);
TVM_REGISTER_NODE_TYPE(VarDefFrameNode);

}  // namespace printer
}  // namespace script
}  // namespace tvm
