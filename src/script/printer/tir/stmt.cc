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

#include <tvm/arith/analyzer.h>
#include <tvm/ir/expr.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../../tir/transforms/ir_utils.h"
#include "../util.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

class FreeVariableFinder : public tir::ExprVisitor {
 public:
  SymbolTableNode* sym;

  std::unordered_set<ObjectRef, ObjectPtrHash, ObjectPtrEqual> free_vars;

  explicit FreeVariableFinder(SymbolTableNode* sym) : sym(sym){};

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

Array<ObjectRef> FindFreeVariables(const PrimExpr& expr, const SymbolTable& sym) {
  FreeVariableFinder visitor{sym.get()};
  visitor(expr);
  return {visitor.free_vars.begin(), visitor.free_vars.end()};
}

class BufferVarUsageFinder : public tir::StmtExprVisitor {
 public:
  explicit BufferVarUsageFinder(tir::Var buffer_var) : target_buffer_var_(buffer_var) {}

  void VisitExpr_(const tir::BufferLoadNode* op) final {
    VisitBuffer(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    VisitBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  Array<tir::Buffer> usage;

 private:
  void VisitBuffer(const tir::Buffer& buffer) {
    if (static_cast<ObjectRef>(buffer->data) == buffer->data &&  // Avoid PrimExpr's == operator
        std::find(usage.begin(), usage.end(), buffer) == usage.end()) {
      usage.push_back(buffer);
    }
  }

  tir::Var target_buffer_var_;
};

Array<tir::Buffer> FindBufferVarUsage(tir::Var buffer_var, tir::Stmt body) {
  BufferVarUsageFinder visitor(buffer_var);
  visitor(body);
  return std::move(visitor.usage);
}

Array<StmtDoc> GetFreeVariableDefinitions(const PrimExpr& expr, IRDocsifier p) {
  Array<StmtDoc> defs;
  Array<ObjectRef> free_vars = FindFreeVariables(expr, p->sym);
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

StmtBlockDoc AsConciseScopedStmts(Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body,
                                  Optional<StmtDoc> concise_stmt_override, TIRFrame frame) {
  if (frame->allow_concise_scoping_) {
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
                                  TIRFrame frame) {
  return AsConciseScopedStmts(lhs, rhs, body, NullOpt, frame);
}

StmtBlockDoc AsConciseScopedStmts(ExprDoc rhs, Array<StmtDoc> body, TIRFrame frame) {
  return AsConciseScopedStmts(NullOpt, rhs, body, NullOpt, frame);
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
  return AsConciseScopedStmts(NullOpt, assert_call_expr, body, assert_stmt,
                              p->GetFrame<TIRFrame>().value());
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
  return AsConciseScopedStmts(NullOpt, let_call, body, var_def, current_frame);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::LetStmt>(PrintLetStmt);

namespace {
struct AllocUsage {
  tir::Buffer alloc_buffer;
  Array<tir::Buffer> aliasing_buffers;
};

template <typename AllocNode>
AllocUsage FindAllocateUsage(AllocNode* op) {
  Array<tir::Buffer> buffer_usage = FindBufferVarUsage(op->buffer_var, op->body);

  auto is_exact_match = [](tir::Buffer a, tir::Buffer b) {
    if (a->dtype != b->dtype) return false;
    if (a->shape.size() != b->shape.size()) return false;

    arith::Analyzer analyzer;
    for (size_t i = 0; i < a->shape.size(); i++) {
      if (!analyzer.CanProveEqual(a->shape[i], b->shape[i])) {
        return false;
      }
    }
    return true;
  };

  // If the buffer allocated via T.allocate is an exact match to the
  // usage of the buffer later on, then that buffer is the return
  // value of T.allocate, and no T.buffer_decl statement is needed.
  tir::Buffer alloc_buffer(op->buffer_var, op->dtype, op->extents, {}, 0, op->buffer_var->name_hint,
                           0, 0, tir::kDefault);
  bool found_alloc_buf = false;
  Array<tir::Buffer> aliasing_buffers;
  for (const auto& buf : buffer_usage) {
    if (!found_alloc_buf && is_exact_match(buf, alloc_buffer)) {
      alloc_buffer = buf;
      found_alloc_buf = true;
    } else {
      aliasing_buffers.push_back(buf);
    }
  }

  return AllocUsage{alloc_buffer, aliasing_buffers};
}
}  // namespace

StmtBlockDoc PrintAllocate(tir::Allocate stmt, IRDocsifier p) {
  // TODO: Print aliasing buffers
  AllocUsage usage = FindAllocateUsage(stmt.get());

  auto f_var_defined = [&p](const tir::VarNode* var) -> bool {
    return p->sym->GetObjectDoc(GetRef<tir::Var>(var)).defined();
  };
  std::unordered_set<const tir::VarNode*> var_explicit_def;
  std::unordered_map<const tir::VarNode*, const tir::BufferNode*> var_associated_def;
  BufferPrintInfo alloc_buffer_info = GetBufferPrintInfo({usage.alloc_buffer}, f_var_defined,
                                                         &var_explicit_def, &var_associated_def)
                                          .at(0);
  std::vector<BufferPrintInfo> aliasing_buffer_infos = GetBufferPrintInfo(
      std::vector<tir::Buffer>(usage.aliasing_buffers.begin(), usage.aliasing_buffers.end()),
      f_var_defined, &var_explicit_def, &var_associated_def);

  TIRGeneralFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);

  IdDoc alloc_buffer_id_doc =
      frame->DefByName(usage.alloc_buffer, p->sym->GetUniqueName(usage.alloc_buffer->name));
  frame->DefByDoc(usage.alloc_buffer->data, alloc_buffer_id_doc->Attr("data"));

  Array<ExprDoc> alloc_args;
  Array<String> alloc_kwarg_keys;
  Array<ExprDoc> alloc_kwarg_values;

  alloc_args.push_back(ListDoc(AsExprDocArray(stmt->extents, p)));
  alloc_args.push_back(DType2Literal(stmt->dtype));
  alloc_args.push_back(LiteralDoc::Str(tir::GetPtrStorageScope(stmt->buffer_var)));
  if (!tir::is_one(stmt->condition)) {
    alloc_args.push_back(p->AsExprDoc(stmt->condition));
  }
  if (!stmt->annotations.empty()) {
    alloc_kwarg_keys.push_back("annotations");
    alloc_kwarg_values.push_back(AsDictDoc(stmt->annotations, p));
  }
  ExprDoc alloc_buffer_expr_doc =
      TIR(p)->Attr("allocate")->Call(alloc_args, alloc_kwarg_keys, alloc_kwarg_values);

  Array<StmtDoc> body;
  for (const BufferPrintInfo& aliasing_buffer_info : aliasing_buffer_infos) {
    IdDoc lhs = frame->DefByName(aliasing_buffer_info.buffer,
                                 p->sym->GetUniqueName(aliasing_buffer_info.buffer->name));
    ExprDoc rhs =
        aliasing_buffer_info.AsCall(TIR(p)->Attr("buffer_decl"),
                                    [&p](const PrimExpr& e) -> ExprDoc { return p->AsExprDoc(e); });
    body.push_back(AssignDoc(lhs, NullOpt, rhs));
  }
  body = runtime::Concat(body, AsStmtDocArray(stmt->body, p));

  return AsConciseScopedStmts(alloc_buffer_id_doc, alloc_buffer_expr_doc, body,
                              p->GetFrame<TIRFrame>().value());
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Allocate>(PrintAllocate);

StmtDoc PrintAllocateConst(tir::AllocateConst stmt, IRDocsifier p) { throw; }

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

    return AsConciseScopedStmts(TIR(p)->Attr("realize")->Call(realize_args), body,
                                p->GetFrame<TIRFrame>().value());
  } else if (stmt->node->IsInstance<tir::IterVarNode>() &&
             (stmt->attr_key == "thread_extent" || stmt->attr_key == "virtual_thread")) {
    // IterVar
    const auto iter_var = Downcast<tir::IterVar>(stmt->node);
    TIRGeneralFrame new_frame(p->sym);
    WithCtx with_frame = p->WithFrame(new_frame);

    IdDoc var_doc =
        new_frame->DefByName(iter_var->var, p->sym->GetUniqueName(iter_var->var->name_hint));

    ExprDoc launch_thread_call =
        TIR(p)->Attr("launch_thread")->Call({var_doc, p->AsExprDoc(stmt->value)});
    Array<StmtDoc> body = AsStmtDocArray(stmt->body, p);

    return AsConciseScopedStmts(launch_thread_call, body, p->GetFrame<TIRFrame>().value());
  } else {
    // General Form
    ExprDoc attr_expr = TIR(p)->Attr("attr")->Call(
        {p->AsExprDoc(stmt->node), LiteralDoc::Str(stmt->attr_key), p->AsExprDoc(stmt->value)});
    Array<StmtDoc> body = AsStmtDocArray(stmt->body, p);
    return AsConciseScopedStmts(attr_expr, body, p->GetFrame<TIRFrame>().value());
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::AttrStmt>(PrintAttrStmt);

Map<tir::Var, tir::For> GetLoopVarMap(IRDocsifier p) {
  Map<tir::Var, tir::For> result;
  for (const TIRLoopFrame& frame : p->GetFrames<TIRLoopFrame>()) {
    for (const tir::For& loop : frame->loops) {
      result.Set(loop->loop_var, loop);
    }
  }
  return result;
}

bool IsSimpleLoop(const tir::ForNode* stmt,
                  const std::vector<const tir::ForNode*>& previous_loops) {
  auto is_used_by_previous_loops = [&previous_loops](const tir::VarNode* v) {
    return std::find_if(previous_loops.begin(), previous_loops.end(),
                        [v](const tir::ForNode* for_node) {
                          return for_node->loop_var.get() == v;
                        }) != previous_loops.end();
  };
  return stmt->kind == tir::ForKind::kSerial && stmt->annotations.empty() &&
         tir::is_zero(stmt->min) && !tir::UsesVar(stmt->min, is_used_by_previous_loops) &&
         !tir::UsesVar(stmt->extent, is_used_by_previous_loops);
}

StmtDoc PrintRegularLoop(const tir::For& stmt, IRDocsifier p) {
  TIRLoopFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);
  // TODO: consider moving to constructor
  frame->loops.push_back(stmt);

  IdDoc loop_var_doc =
      frame->DefByName(stmt->loop_var, p->sym->GetUniqueName(stmt->loop_var->name_hint));

  Array<ExprDoc> loop_var_vars;
  Array<String> loop_var_kwarg_keys;
  Array<ExprDoc> loop_var_kwarg_values;
  if (tir::is_zero(stmt->min)) {
    loop_var_vars.push_back(p->AsExprDoc(stmt->extent));
  } else {
    arith::Analyzer analyzer;
    loop_var_vars.push_back(p->AsExprDoc(stmt->min));
    loop_var_vars.push_back(p->AsExprDoc(analyzer.Simplify(stmt->min + stmt->extent)));
  }
  if (stmt->thread_binding.defined()) {
    loop_var_kwarg_keys.push_back("thread");
    loop_var_kwarg_values.push_back(LiteralDoc::Str(stmt->thread_binding.value()->thread_tag));
  }
  if (!stmt->annotations.empty()) {
    loop_var_kwarg_keys.push_back("annotations");
    loop_var_kwarg_values.push_back(AsDictDoc(stmt->annotations, p));
  }
  ExprDoc loop_var_rhs = TIR(p)
                             ->Attr(tir::ForKind2String(stmt->kind))
                             ->Call(loop_var_vars, loop_var_kwarg_keys, loop_var_kwarg_values);

  Array<StmtDoc> body = AsStmtDocArray(stmt->body, p);

  return ForDoc(loop_var_doc, loop_var_rhs, body);
}

StmtDoc PrintMergedSimpleLoops(const std::vector<const tir::ForNode*>& stmts, IRDocsifier p) {
  TIRLoopFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);

  Array<ExprDoc> loop_var_docs;
  Array<ExprDoc> loop_var_extent_docs;
  for (const tir::ForNode* loop : stmts) {
    frame->loops.push_back(GetRef<tir::For>(loop));
    loop_var_docs.push_back(
        frame->DefByName(loop->loop_var, p->sym->GetUniqueName(loop->loop_var->name_hint)));
    loop_var_extent_docs.push_back(p->AsExprDoc(loop->extent));
  }

  Array<ExprDoc> loop_var_vars;
  ExprDoc loop_var_rhs = TIR(p)->Attr("grid")->Call(loop_var_extent_docs);

  Array<StmtDoc> body = AsStmtDocArray(stmts.back()->body, p);

  return ForDoc(TupleDoc(loop_var_docs), loop_var_rhs, body);
}

StmtDoc PrintFor(tir::For stmt, IRDocsifier p) {
  std::vector<const tir::ForNode*> simple_loops;

  tir::Stmt target_stmt = stmt;
  while (const auto* for_stmt = target_stmt.as<tir::ForNode>()) {
    if (IsSimpleLoop(for_stmt, simple_loops)) {
      simple_loops.push_back(for_stmt);
    } else {
      break;
    }
  }

  if (simple_loops.size() > 1) {
    return PrintMergedSimpleLoops(simple_loops, p);
  } else if (simple_loops.size() == 1) {
    return PrintRegularLoop(GetRef<tir::For>(simple_loops[0]), p);
  } else {
    return PrintRegularLoop(stmt, p);
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::For>(PrintFor);

StmtBlockDoc PrintBlock(tir::Block stmt, IRDocsifier p) {
  TIRLoopFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);
  Array<StmtDoc> body;

  auto f_var_defined = [&p](const tir::VarNode* var) -> bool {
    return p->sym->GetObjectDoc(GetRef<tir::Var>(var)).defined();
  };
  std::unordered_set<const tir::VarNode*> var_explicit_def;
  std::unordered_map<const tir::VarNode*, const tir::BufferNode*> var_associated_def;
  std::vector<BufferPrintInfo> buffer_print_infos = GetBufferPrintInfo(
      std::vector<tir::Buffer>(stmt->alloc_buffers.begin(), stmt->alloc_buffers.end()),
      f_var_defined, &var_explicit_def, &var_associated_def);

  for (const auto& buffer_print_info : buffer_print_infos) {
    const tir::Buffer& buf = buffer_print_info.buffer;
    IdDoc lhs = frame->DefByName(buf, p->sym->GetUniqueName(buf->name));
    frame->DefByDoc(buf->data, lhs->Attr("data"));
    ExprDoc rhs =
        buffer_print_info.AsCall(TIR(p)->Attr("alloc_buffer"),
                                 [&p](const PrimExpr& e) -> ExprDoc { return p->AsExprDoc(e); });
    body.push_back(AssignDoc(lhs, NullOpt, rhs));
  }

  for (const auto& match_buffer : stmt->match_buffers) {
    IdDoc lhs =
        frame->DefByName(match_buffer->buffer, p->sym->GetUniqueName(match_buffer->buffer->name));
    BufferPrintInfo info = GetBufferPrintInfo({match_buffer->buffer}, f_var_defined,
                                              &var_explicit_def, &var_associated_def)
                               .at(0);
    ExprDoc rhs = info.AsCall(TIR(p)->Attr("match_buffer"), {p->AsExprDoc(match_buffer->source)},
                              [&p](const PrimExpr& e) -> ExprDoc { return p->AsExprDoc(e); });
    body.push_back(AssignDoc(lhs, NullOpt, rhs));
  }

  if (stmt->init.defined()) {
    body.push_back(ScopeDoc(TIR(p)->Attr("init")->Call({}), AsStmtDocArray(stmt->init.value(), p)));
  }

  body = runtime::Concat(body, AsStmtDocArray(stmt->body, p));

  return StmtBlockDoc(body);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Block>(PrintBlock);

namespace {
struct BlockVarBinding {
  tir::IterVar lhs;
  PrimExpr rhs;

  BlockVarBinding(tir::IterVar lhs, PrimExpr rhs) : lhs(lhs), rhs(rhs) {}
};
}  // namespace

std::vector<std::vector<BlockVarBinding>> GetBlockVarGroups(
    const Array<tir::IterVar>& iter_vars, const Array<PrimExpr>& values,
    const Map<tir::Var, tir::For>& loop_var_map) {
  ICHECK_EQ(iter_vars.size(), values.size());
  ICHECK(iter_vars.size() > 0);

  std::vector<std::vector<BlockVarBinding>> result;
  result.emplace_back();

  tir::ExprDeepEqual expr_equal;
  auto is_simple_remap = [&expr_equal, &loop_var_map](const tir::IterVar& iter_var,
                                                      const PrimExpr& value) -> bool {
    if (iter_var->iter_type != tir::kDataPar && iter_var->iter_type != tir::kCommReduce)
      return false;
    if (!value->IsInstance<tir::VarNode>()) {
      return false;
    }
    const auto& var = Downcast<tir::Var>(value);
    auto it = loop_var_map.find(var);
    return it != loop_var_map.end() && expr_equal((*it).second->min, iter_var->dom->min) &&
           expr_equal((*it).second->extent, iter_var->dom->extent);
  };

  bool last_is_simple_remap = true;
  for (size_t i = 0; i < iter_vars.size(); i++) {
    const tir::IterVar& iter_var = iter_vars[i];
    const PrimExpr& value = values[i];
    bool current_is_simple_remap = is_simple_remap(iter_var, value);
    if (!(current_is_simple_remap && last_is_simple_remap)) {
      // Group continues iff. both current var and previous var are simple remapping
      result.emplace_back();
    }
    result.back().emplace_back(iter_var, value);
    last_is_simple_remap = current_is_simple_remap;
  }

  return result;
}

AssignDoc PrintBlockVar(const BlockVarBinding& block_var_binding, IRDocsifier p) {
  ExprDoc lhs = p->AsExprDoc(block_var_binding.lhs);
  String iter_type;
  switch (block_var_binding.lhs->iter_type) {
    case tir::kDataPar:
      iter_type = "spatial";
      break;
    case tir::kCommReduce:
      iter_type = "reduce";
      break;
    case tir::kOrdered:
      iter_type = "scan";
      break;
    case tir::kOpaque:
      iter_type = "opaque";
      break;
    default:
      LOG(FATAL) << "Unknown block var iter type: " << block_var_binding.lhs->iter_type;
      break;
  }
  Array<ExprDoc> args;
  const Range& dom = block_var_binding.lhs->dom;
  if (tir::is_zero(dom->min)) {
    args.push_back(p->AsExprDoc(dom->extent));
  } else {
    args.push_back(TupleDoc({p->AsExprDoc(dom->min), p->AsExprDoc(dom->min + dom->extent)}));
  }
  args.push_back(p->AsExprDoc(block_var_binding.rhs));
  ExprDoc rhs = TIR(p)->Attr("axis")->Attr(iter_type)->Call(args);

  return AssignDoc(lhs, NullOpt, rhs);
}

AssignDoc PrintGroupedSimpleRemappingBlockVars(
    const std::vector<BlockVarBinding>& block_var_bindings, IRDocsifier p) {
  std::vector<ExprDoc> iter_var_ids;
  std::vector<ExprDoc> iter_value_docs;
  std::string iter_type;

  for (const BlockVarBinding& binding : block_var_bindings) {
    const tir::IterVar& iter_var = binding.lhs;
    const PrimExpr& iter_value = binding.rhs;

    iter_var_ids.emplace_back(p->AsExprDoc(iter_var));
    if (iter_var->iter_type == tir::kDataPar) {
      iter_type += "S";
    } else if (iter_var->iter_type == tir::kCommReduce) {
      iter_type += "R";
    } else {
      ICHECK(false);
    }
    iter_value_docs.emplace_back(p->AsExprDoc(iter_value));
  }

  ExprDoc lhs = TupleDoc(iter_var_ids);
  ExprDoc rhs = TIR(p)->Attr("axis")->Attr("remap")->Call(
      {LiteralDoc::Str(iter_type), ListDoc(iter_value_docs)});

  return AssignDoc(lhs, NullOpt, rhs);
}

Array<StmtDoc> PrintBlockVars(const tir::BlockRealize& stmt, IRDocsifier p) {
  Map<tir::Var, tir::For> loop_var_map = GetLoopVarMap(p);
  Array<StmtDoc> result;
  for (const std::vector<BlockVarBinding>& binding_group :
       GetBlockVarGroups(stmt->block->iter_vars, stmt->iter_values, loop_var_map)) {
    ICHECK_NE(binding_group.size(), 0);
    if (binding_group.size() == 1) {
      result.push_back(PrintBlockVar(binding_group[0], p));
    } else {
      result.push_back(PrintGroupedSimpleRemappingBlockVars(binding_group, p));
    }
  }
  return result;
}

Array<StmtDoc> PrintBlockAttrs(const tir::BlockRealize& stmt, IRDocsifier p) {
  std::vector<ExprDoc> attr_exprs;
  const tir::Block& block = stmt->block;

  if (!tir::is_one(stmt->predicate)) {
    attr_exprs.emplace_back(TIR(p)->Attr("where")->Call({p->AsExprDoc(stmt->predicate)}));
  }
  attr_exprs.push_back(TIR(p)->Attr("reads")->Call(AsExprDocArray(block->reads, p)));
  attr_exprs.push_back(TIR(p)->Attr("writes")->Call(AsExprDocArray(block->writes, p)));
  if (!block->annotations.empty()) {
    attr_exprs.push_back(TIR(p)->Attr("block_attr")->Call({AsDictDoc(block->annotations, p)}));
  }

  Array<StmtDoc> result;
  for (const ExprDoc& attr : attr_exprs) {
    result.push_back(ExprStmtDoc(attr));
  }

  return result;
}

StmtDoc PrintBlockRealize(tir::BlockRealize stmt, IRDocsifier p) {
  const tir::Block& block = stmt->block;

  Array<ExprDoc> block_args;
  if (!block->name_hint.empty()) {
    block_args.push_back(LiteralDoc::Str(block->name_hint));
  }
  ExprDoc block_begin_expr = TIR(p)->Attr("block")->Call(block_args);

  TIRGeneralFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);
  for (const tir::IterVar& iter_var : block->iter_vars) {
    frame->DefByName(iter_var->var, p->sym->GetUniqueName(iter_var->var->name_hint));
  }

  Array<StmtDoc> body = PrintBlockVars(stmt, p);
  body = runtime::Concat(body, PrintBlockAttrs(stmt, p));
  body = runtime::Concat(body, AsStmtDocArray(block, p));

  return ScopeDoc(block_begin_expr, body);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BlockRealize>(PrintBlockRealize);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRealize>([](tir::BufferRealize stmt, IRDocsifier p) -> Doc {
      LOG(FATAL) << "TVM Script Printer Internal Error: All the BufferRealize should be folded "
                    "with Attr";
      throw;
    });
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerStore>([](tir::ProducerStore stmt, IRDocsifier p) -> Doc {
      LOG(FATAL) << "ProducerStore cannot be printed";
      throw;
    });
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerRealize>([](tir::ProducerRealize stmt, IRDocsifier p) -> Doc {
      LOG(FATAL) << "ProducerRealize cannot be printed";
      throw;
    });
// • BlockNode: Class
// • BlockRealizeNode: Class
//
}  // namespace printer
}  // namespace script
}  // namespace tvm
