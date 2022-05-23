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
#ifndef TVM_SCRIPT_PRINTER_TIR_UTIL_H_
#define TVM_SCRIPT_PRINTER_TIR_UTIL_H_

#include "../ir_docsifier.h"
#include "./tir.h"

namespace tvm {
namespace script {
namespace printer {

inline bool AllowConciseScoping(const IRDocsifier& p) {
  const auto* f = p->frames.back().as<TIRFrameNode>();
  return f != nullptr && f->allow_concise_scoping_;
}

inline Array<StmtDoc> AsStmtDocArray(const ObjectRef& obj, IRDocsifier p) {
  Doc doc = p->AsDoc<Doc>(obj);
  if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
    return stmt_block->stmts;
  } else if (const auto* stmt_node = doc.as<StmtDocNode>()) {
    return {GetRef<StmtDoc>(stmt_node)};
  } else {
    LOG(FATAL) << "Expect to get StmtBlockDoc or StmtDoc, got "
               << Object::TypeIndex2Key(doc->type_index());
    throw;
  }
}

inline IdDoc TIR(const IRDocsifier& p) { return IdDoc(p->ir_prefix.Get("tir").value_or("T")); }

inline LiteralDoc DType2Literal(const DLDataType& dtype) {
  using runtime::DLDataType2String;
  return LiteralDoc::Str(DLDataType2String(dtype));
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_UTIL_H_
