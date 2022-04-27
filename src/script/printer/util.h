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
#ifndef TVM_SCRIPT_PRINTER_UTIL_H_
#define TVM_SCRIPT_PRINTER_UTIL_H_

#include "./ir_docsifier.h"

namespace tvm {
namespace script {
namespace printer {
template <typename DocType, typename NodeType>

Array<DocType> AsDocArray(const Array<NodeType>& refs, const IRDocsifier& ir_docsifier) {
  Array<DocType> result;
  for (auto& ref : refs) {
    result.push_back(ir_docsifier->AsExprDoc(ref));
  }
  return result;
}

template <typename NodeType>
Array<ExprDoc> AsExprDocArray(const Array<NodeType>& refs, const IRDocsifier& ir_docsifier) {
  return AsDocArray<ExprDoc>(refs, ir_docsifier);
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif
