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

#include <tvm/node/visit_traced.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace script {
namespace printer {

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