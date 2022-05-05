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

#include "tvm/tir/stmt.h"

#include <tvm/ir/expr.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>

#include "../util.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

class FreeVariableVisitor : public tir::ExprVisitor {
 public:
  SymbolTableNode* sym;

  std::unordered_set<ObjectRef, ObjectPtrHash, ObjectPtrEqual> free_vars;

  explicit FreeVariableVisitor(SymbolTableNode* sym) : sym(sym){};

 protected:
  void VisitExpr_(const tir::VarNode* op) override {
    auto ref = GetRef<tir::Var>(op);
    if (sym->GetObjectDoc(ref) == nullptr) {
      free_vars.insert(ref);
    }
  }

  void VisitExpr_(const tir::BufferLoadNode* op) override {
    const tir::Buffer& buffer = op->buffer;
    if (sym->GetObjectDoc(buffer) == nullptr) {
      free_vars.insert(buffer);
    }
    tir::ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const tir::LetNode* op) override {
    free_vars.insert(op->var);
    tir::ExprVisitor::VisitExpr_(op);
    free_vars.erase(op->var);
  };
};

Array<ObjectRef> GetFreeVariablesFromExpr(const PrimExpr& expr, const SymbolTable& sym) {
  FreeVariableVisitor visitor{sym.get()};
  visitor(expr);
  return {visitor.free_vars.begin(), visitor.free_vars.end()};
}

Array<StmtDoc> GetFreeVariableDefinitions(const PrimExpr& expr, IRDocsifier p) {
  Array<StmtDoc> defs;
  Array<ObjectRef> free_vars = GetFreeVariablesFromExpr(expr, p->sym);
  for (const ObjectRef& free_var : free_vars) {
    if (const auto* var_node = free_var.as<tir::VarNode>()) {
      tir::Var var = GetRef<tir::Var>(var_node);
      IdDoc var_doc =
          p->GetFrame<TIRFrame>().value()->DefByName(var, p->sym->GetUniqueName(var->name_hint));
      AssignDoc var_def =
          AssignDoc(var_doc, TIR(p)->Attr("var")->Call({DType2Literal(var->dtype)}), NullOpt);
      defs.push_back(var_def);
    } else if (const auto* buf_node = free_var.as<tir::BufferNode>()) {
      tir::Buffer buf = GetRef<tir::Buffer>(buf_node);
      // TODO: Print T.decl_buffer
      throw;
    } else {
      LOG(FATAL) << "Unknown variable type " << Object::TypeIndex2Key(free_var->type_index());
      throw;
    }
  }
  return defs;
}

Array<StmtDoc> AsStmtDocArray(const ObjectRef& obj, IRDocsifier p) {
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

StmtBlockDoc AsConciseScopedStmts(Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body,
                                  Optional<StmtDoc> concise_stmt_override, IRDocsifier p) {
  if (p->GetFrame<TIRFrame>().value()->allow_concise_scoping_) {
    StmtDoc first_doc = ExprStmtDoc(rhs);
    if (concise_stmt_override) {
      first_doc = concise_stmt_override.value();
    } else if (lhs) {
      first_doc = AssignDoc(lhs.value(), rhs, NullOpt);
    }

    return StmtBlockDoc(runtime::Concat({first_doc}, body));
  } else {
    return StmtBlockDoc({ScopeDoc(lhs, rhs, body)});
  }
}

StmtBlockDoc AsConciseScopedStmts(Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body,
                                  IRDocsifier p) {
  return AsConciseScopedStmts(lhs, rhs, body, NullOpt, p);
}

StmtBlockDoc AsConciseScopedStmts(ExprDoc rhs, Array<StmtDoc> body, IRDocsifier p) {
  return AsConciseScopedStmts(NullOpt, rhs, body, NullOpt, p);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::SeqStmt>([](tir::SeqStmt stmt, IRDocsifier p) {
      Array<StmtDoc> result;
      for (const tir::Stmt& child_stmt : stmt->seq) {
        result = runtime::Concat(result, AsStmtDocArray(child_stmt, p));
      }
      return StmtBlockDoc(result);
    });

StmtBlockDoc PrintAssertStmt(tir::AssertStmt stmt, IRDocsifier p) {
  // TODO: extract free vars from expr
  ExprDoc condition_expr = p->AsExprDoc(stmt->condition);
  ExprDoc message_expr = p->AsExprDoc(stmt->message);
  Array<StmtDoc> body = AsStmtDocArray(stmt->body, p);

  ExprDoc assert_call_expr = TIR(p)->Attr("Assert")->Call({condition_expr, message_expr});
  StmtDoc assert_stmt =
      ExprStmtDoc(OperationDoc(OperationDocNode::Kind::kAssert, {condition_expr, message_expr}));
  return AsConciseScopedStmts(NullOpt, assert_call_expr, body, assert_stmt, p);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::AssertStmt>(PrintAssertStmt);

StmtDoc PrintStore(tir::Store stmt, IRDocsifier p) {
  // TODO: extract free vars from expr
  Array<ExprDoc> args = AsExprDocArray(
      Array<PrimExpr>{stmt->buffer_var, stmt->index, stmt->value, stmt->predicate}, p);
  return ExprStmtDoc(TIR(p)->Attr("store")->Call(args));
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Store>(PrintStore);

StmtDoc PrintBufferStore(tir::BufferStore stmt, IRDocsifier p) {
  Array<ExprDoc> indices = AsExprDocArray(stmt->indices, p);
  Array<Doc> index_docs(indices.begin(), indices.end());
  return AssignDoc(p->AsExprDoc(stmt->buffer)->Index(index_docs), p->AsExprDoc(stmt->value),
                   NullOpt);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferStore>(PrintBufferStore);

StmtDoc PrintIfThenElse(tir::IfThenElse stmt, IRDocsifier p) {
  ExprDoc predicate = p->AsExprDoc(stmt->condition);
  Array<StmtDoc> then_branch = AsStmtDocArray(stmt->then_case, p);
  Array<StmtDoc> else_branch = AsStmtDocArray(stmt->else_case, p);
  return IfDoc(predicate, then_branch, else_branch);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::IfThenElse>(PrintIfThenElse);

StmtDoc PrintWhile(tir::While stmt, IRDocsifier p) {
  return WhileDoc(p->AsExprDoc(stmt->condition), AsStmtDocArray(stmt->body, p));
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::While>(PrintWhile);

StmtDoc PrintPrefetch(tir::Prefetch stmt, IRDocsifier p) {
  return ExprStmtDoc(
      TIR(p)->Attr("prefetch")->Call({p->AsExprDoc(stmt->buffer), p->AsExprDoc(stmt->bounds)}));
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Prefetch>(PrintPrefetch);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Evaluate>([](tir::Evaluate stmt, IRDocsifier p) {
      return ExprStmtDoc(p->AsExprDoc(stmt->value));
    });

StmtBlockDoc PrintLetStmt(tir::LetStmt stmt, IRDocsifier p) {
  auto current_frame = p->GetFrame<TIRFrame>().value();
  TIRGeneralFrame new_frame(p->sym);
  WithCtx with_frame = p->WithFrame(new_frame);

  // TODO: PrintNonHeaderBufferDeclarations
  Array<StmtDoc> body = AsStmtDocArray(stmt->body, p);

  IdDoc var_doc = new_frame->DefByName(stmt->var, p->sym->GetUniqueName(stmt->var->name_hint));
  AssignDoc var_def =
      AssignDoc(var_doc, p->AsExprDoc(stmt->value), p->AsExprDoc(GetType(stmt->var)));
  ExprDoc let_call =
      TIR(p)->Attr("let")->Call({LiteralDoc::Str(var_doc->name), p->AsExprDoc(stmt->value)});
  return AsConciseScopedStmts(NullOpt, let_call, body, var_def, p);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::LetStmt>(PrintLetStmt);

StmtDoc PrintAllocate(tir::Allocate stmt, IRDocsifier p) {}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Allocate>(PrintAllocate);

StmtDoc PrintAllocateConst(tir::AllocateConst stmt, IRDocsifier p) {}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::AllocateConst>(PrintAllocateConst);

StmtBlockDoc PrintAttrStmt(tir::AttrStmt stmt, IRDocsifier p) {
  if (stmt->node->IsInstance<tir::BufferNode>() && stmt->attr_key == "realize_scope" &&
      stmt->body->IsInstance<tir::BufferRealizeNode>()) {
    // BufferRealize
    const auto* realize = Downcast<tir::BufferRealize>(stmt->body).get();
    ICHECK(realize->buffer.same_as(stmt->node));

    Array<ExprDoc> realize_args;
    realize_args.push_back(p->AsExprDoc(realize->buffer));
    realize_args.push_back(ListDoc(AsExprDocArray(realize->bounds, p)));
    realize_args.push_back(p->AsExprDoc(stmt->value));
    if (!tir::is_one(realize->condition)) {
      realize_args.push_back(p->AsExprDoc(realize->condition));
    }

    Array<StmtDoc> body = AsStmtDocArray(realize->body, p);

    return AsConciseScopedStmts(TIR(p)->Attr("realize")->Call(realize_args), body, p);
  } else if (stmt->node->IsInstance<tir::IterVarNode>() &&
             (stmt->attr_key == "thread_extent" || stmt->attr_key == "virtual_thread")) {
    // IterVar
  } else {
    // General Form
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::AttrStmt>(PrintAttrStmt);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRealize>([](tir::BufferRealize stmt, IRDocsifier p) -> Doc {
      LOG(FATAL) << "TVM Script Printer Internal Error: All the BufferRealize should be folded "
                    "with Attr";
      throw;
    });
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerStore>([](auto stmt, IRDocsifier p) -> Doc {
      LOG(FATAL) << "ProducerStore cannot be printed";
      throw;
    });
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerRealize>([](auto stmt, IRDocsifier p) -> Doc {
      LOG(FATAL) << "ProducerRealize cannot be printed";
      throw;
    });
// • ForNode: Class
// • BlockNode: Class
// • BlockRealizeNode: Class
//
}  // namespace printer
}  // namespace script
}  // namespace tvm
