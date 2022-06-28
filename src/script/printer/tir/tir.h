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
#include <tvm/tir/stmt.h>

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

class TIRLoopFrameNode : public TIRFrameNode {
 public:
  Array<tir::For> loops{};  // the first element is the outer-most loop

  static constexpr const char* _type_key = "script.TIRLoopFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRLoopFrameNode, FrameNode);
};

class TIRLoopFrame : public TIRFrame {
 public:
  explicit TIRLoopFrame(SymbolTable sym) {
    ObjectPtr<TIRLoopFrameNode> n = make_object<TIRLoopFrameNode>();
    n->sym = sym.get();
    data_ = std::move(n);
  }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRLoopFrame, TIRFrame, TIRLoopFrameNode);
};

Map<tir::Var, tir::For> GetLoopVarMap(IRDocsifier p);

ExprDoc PrintOpCall(TracedObject<tir::Call> call, IRDocsifier p);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_TIR_H_
