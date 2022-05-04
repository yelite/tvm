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
#include <tvm/tir/expr.h>

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
  explicit TIRFrame(SymbolTable sym) {
    ObjectPtr<TIRFrameNode> n = make_object<TIRFrameNode>();
    n->sym = sym.get();
    data_ = std::move(n);
  }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRFrame, Frame, TIRFrameNode);
};

class TIRGeneralFrameNode : public TIRFrameNode {
 public:
  static constexpr const char* _type_key = "script.TIRGeneralFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRGeneralFrameNode, FrameNode);
};

class TIRGeneralFrame : public TIRFrame {
 public:
  using TIRFrame::TIRFrame;
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRGeneralFrame, TIRFrame, TIRGeneralFrameNode);
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

ExprDoc PrintOpCall(tir::Call call, IRDocsifier p);

Doc VarDef(tir::Var v, IRDocsifier p);                               // a = T.var("int32")
Doc VarDecl(tir::Var, IRDocsifier p);                                // a: T.int32
Doc IterVarBlockVarDef(tir::IterVar, IRDocsifier p);                 // a = T.axis.S/R(...)
Doc IterVarLaunchThreadDef(tir::IterVar, IRDocsifier p);             // a = T.launch_thread(...)
Doc BufferMatchBuffer(tir::Buffer, BufferPrintInfo, IRDocsifier p);  // a = T.match_buffer(...)
Doc BufferDef(tir::Buffer, BufferPrintInfo, IRDocsifier p);          // a = T.buffer_decl(...)
Doc BufferDecl(tir::Buffer, BufferPrintInfo, IRDocsifier p);         // a: T.Buffer()

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_TIR_H_
