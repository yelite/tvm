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

#include "utils.h"

#include <tvm/tir/op.h>

#include "../visit_traced.h"

namespace tvm {
namespace script {
namespace printer {

namespace {
std::vector<TracedObject<tir::Stmt>> FlattenSeqStmt(const TracedObject<tir::Stmt>& stmt) {
  std::vector<TracedObject<tir::Stmt>> result;

  if (stmt.IsInstance<tir::SeqStmt>()) {
    auto seq = stmt.Downcast<tir::SeqStmt>().GetAttr(&tir::SeqStmtNode::seq);
    for (const TracedObject<tir::Stmt>& child : seq) {
      std::vector<TracedObject<tir::Stmt>> flattened_child = FlattenSeqStmt(child);
      result.insert(result.end(), flattened_child.begin(), flattened_child.end());
    }
  } else {
    result.push_back(stmt);
  }

  return result;
}

Array<StmtDoc> FlattenStmtDoc(const Doc& doc) {
  if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
    return stmt_block->stmts;
  } else if (const auto* stmt_doc = doc.as<StmtDocNode>()) {
    return {GetRef<StmtDoc>(stmt_doc)};
  } else {
    LOG(FATAL) << "Expect to get StmtBlockDoc or StmtDoc, got " << doc->GetTypeKey();
    throw;
  }
}
}  // namespace

Array<StmtDoc> AsStmtDocArray(const TracedObject<tir::Stmt>& obj, IRDocsifier p) {
  Array<StmtDoc> result;
  std::vector<TracedObject<tir::Stmt>> flattened_stmts = FlattenSeqStmt(obj);

  const auto* frame_node = p->frames.back().as<TIRFrameNode>();
  ICHECK_NOTNULL(frame_node);

  size_t length = flattened_stmts.size();

  const bool old_concise_scoping_status = frame_node->allow_concise_scoping_;
  frame_node->allow_concise_scoping_ = false;
  for (size_t i = 0; i < length; i++) {
    if (i == length - 1) {
      frame_node->allow_concise_scoping_ = true;
    }
    result = runtime::Concat(result, FlattenStmtDoc(p->AsDoc<Doc>(flattened_stmts[i])));
  }
  frame_node->allow_concise_scoping_ = old_concise_scoping_status;

  return result;
}

ExprDoc GetTypeAnnotationDocForVar(const TracedObject<tir::Var>& var, const IRDocsifier& p) {
  auto type_annotation = var.GetAttr(&tir::VarNode::type_annotation);
  if (type_annotation.Get().defined()) {
    return p->AsExprDoc(type_annotation);
  } else {
    auto dtype = var.GetAttr(&tir::VarNode::dtype);
    Type raw_type = GetTypeFromRuntimeDataType(dtype.Get());
    return p->AsExprDoc(MakeTraced(raw_type, dtype.GetPath()));
  }
}

void PostOrderVisitExprTraced(const TracedObject<PrimExpr>& expr,
                              const std::function<void(const TracedObject<PrimExpr>&)>& callback) {
  PostOrderVisitTraced(
      expr, [](const ObjectRef& object) { return object->IsInstance<PrimExprNode>(); },
      [&](const TracedObject<ObjectRef>& object) { callback(object.Downcast<PrimExpr>()); });
}

void PostOrderVisitStmtExprTraced(
    const TracedObject<tir::Stmt>& stmt,
    const std::function<void(const TracedObject<ObjectRef>&)>& callback) {
  PostOrderVisitTraced(
      stmt,
      [](const ObjectRef& object) {
        return object->IsInstance<PrimExprNode>() || object->IsInstance<tir::StmtNode>();
      },
      [&](const TracedObject<ObjectRef>& object) { callback(object); });
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
