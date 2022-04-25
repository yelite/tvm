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
#ifndef TVM_SCRIPT_PRINTER_TIR_TIR_H_
#define TVM_SCRIPT_PRINTER_TIR_TIR_H_

#include <tvm/tir/buffer.h>

#include "../ir_docsifier.h"

namespace tvm {
namespace script {
namespace printer {

class TIRFrameNode : public FrameNode {
 public:
  Array<StmtDoc> stmts;
  bool allow_concise_scoping_{false};

  void VisitAttrs(AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    v->Visit("stmts", &stmts);
    v->Visit("allow_concise_scoping", &allow_concise_scoping_);
  }

  static constexpr const char* _type_key = "script.TIRFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRFrameNode, FrameNode);
};

class TIRFrame : public Frame {
 protected:
  TIRFrame() = default;

 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRFrame, Frame, TIRFrameNode);
};

struct BufferPrintInfo {
  tir::Buffer buffer;
  Array<PrimExpr> shape;
  Optional<ExprDoc> dtype;
  Optional<tir::Var> data;
  Optional<Array<PrimExpr>> strides;
  Optional<PrimExpr> elem_offset;
  Optional<ExprDoc> scope;
  Optional<ExprDoc> align;
  Optional<ExprDoc> offset_factor;
  Optional<ExprDoc> buffer_type;

  ExprDoc AsCall(const ExprDoc& prefix, std::function<ExprDoc(const PrimExpr&)> converter) const;
};

std::vector<BufferPrintInfo> GetBufferPrintInfo(
    const std::vector<tir::Buffer>& buffers,  //
    std::function<bool(const tir::VarNode*)> f_var_defined,
    std::unordered_set<const tir::VarNode*>* var_explicit_def,
    std::unordered_map<const tir::VarNode*, const tir::BufferNode*>* var_associated_def);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_TIR_H_
