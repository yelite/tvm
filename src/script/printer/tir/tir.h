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

#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace script {
namespace printer {

class TIRFrameNode : public FrameNode {
 public:
  mutable bool allow_concise_scoping{false};

  void VisitAttrs(AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    v->Visit("allow_concise_scoping", &allow_concise_scoping);
  }

  static constexpr const char* _type_key = "script.printer.TIRFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRFrameNode, FrameNode);
};

class TIRFrame : public Frame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRFrame, Frame, TIRFrameNode);
};

class TIRTopLevelFrameNode : public TIRFrameNode {
 public:
  Array<StmtDoc> free_var_definitions;
  Map<tir::Var, AssignDoc> env_thread_definitions;

  void VisitAttrs(AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("free_var_definitions", &free_var_definitions);
  }

  static constexpr const char* _type_key = "script.printer.TIRTopLevelFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRTopLevelFrameNode, TIRFrameNode);
};

class TIRTopLevelFrame : public TIRFrame {
 public:
  TIRTopLevelFrame();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRTopLevelFrame, TIRFrame,
                                                    TIRTopLevelFrameNode);
};

class TIRGeneralFrameNode : public TIRFrameNode {
 public:
  static constexpr const char* _type_key = "script.printer.TIRGeneralFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRGeneralFrameNode, TIRFrameNode);
};

class TIRGeneralFrame : public TIRFrame {
 public:
  TIRGeneralFrame();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRGeneralFrame, TIRFrame, TIRGeneralFrameNode);
};

class TIRLoopFrameNode : public TIRFrameNode {
 public:
  /*
   * This field is for loop merging. For example,
   * \code
   * for i, j, k in T.grid(128, 128, 128):
   *     ...
   * \endcode
   * corresponds to three nested Loop statements, but
   * it only creates one TIRLoopFrame during printing.
   *
   * The first element is the outer-most loop.
   */
  Array<tir::For> loops{};

  static constexpr const char* _type_key = "script.printer.TIRLoopFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRLoopFrameNode, TIRFrameNode);
};

class TIRLoopFrame : public TIRFrame {
 public:
  TIRLoopFrame();
  explicit TIRLoopFrame(tir::For for_stmt);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRLoopFrame, TIRFrame, TIRLoopFrameNode);
};

class TIRPrimFuncFrameNode : public TIRFrameNode {
 public:
  static constexpr const char* _type_key = "script.printer.TIRPrimFuncFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(TIRPrimFuncFrameNode, TIRFrameNode);
};

class TIRPrimFuncFrame : public TIRFrame {
 public:
  TIRPrimFuncFrame();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRPrimFuncFrame, TIRFrame,
                                                    TIRPrimFuncFrameNode);
};

inline IdDoc TIR(const IRDocsifier& p) { return IdDoc(p->ir_prefix.Get("tir").value_or("T")); }

ExprDoc PrintOpCall(TracedObject<tir::Call> call, IRDocsifier p);

void PostOrderVisitExprTraced(const TracedObject<PrimExpr>& expr,
                              const std::function<void(const TracedObject<PrimExpr>&)>& callback);

void PostOrderVisitStmtExprTraced(
    const TracedObject<tir::Stmt>& expr,
    const std::function<void(const TracedObject<ObjectRef>&)>& callback);

Array<StmtDoc> AsStmtDocArray(const TracedObject<tir::Stmt>& obj, IRDocsifier p);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TIR_TIR_H_
