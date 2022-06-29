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
#include <tvm/node/object_path.h>
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
#include "./buffer.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

static TracedOptional<tir::Buffer> GetUsedBuffer(const TracedObject<ObjectRef>& stmt_or_expr) {
  if (auto load = stmt_or_expr.TryDowncast<tir::BufferLoad>()) {
    return load.value().GetAttr(&tir::BufferLoadNode::buffer);
  } else if (auto store = stmt_or_expr.TryDowncast<tir::BufferStore>()) {
    return store.value().GetAttr(&tir::BufferStoreNode::buffer);
  } else {
    return TracedOptional<tir::Buffer>({}, {});
  }
}

std::vector<TracedObject<tir::Buffer>> FindBufferVarUsage(tir::Var buffer_var,
                                                          TracedObject<tir::Stmt> body) {
  std::vector<TracedObject<tir::Buffer>> ret;
  PostOrderVisitStmtExprTraced(body, [&](const TracedObject<ObjectRef>& stmt_or_expr) {
    if (auto buffer_opt = GetUsedBuffer(stmt_or_expr)) {
      auto buffer = buffer_opt.value();
      if (buffer.Get()->data.same_as(buffer_var) &&
          std::find_if(ret.begin(), ret.end(),
                       [&](const auto& b) { return b.Get() == buffer.Get(); }) == ret.end()) {
        ret.push_back(buffer);
      }
    }
  });
  return ret;
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
    .set_dispatch<tir::SeqStmt>([](TracedObject<tir::SeqStmt> stmt, IRDocsifier p) {
      Array<StmtDoc> result;
      for (const auto& child_stmt : stmt.GetAttr(&tir::SeqStmtNode::seq)) {
        result = runtime::Concat(result, AsStmtDocArray(child_stmt, p));
      }
      return StmtBlockDoc(result);
    });

StmtBlockDoc PrintAssertStmt(TracedObject<tir::AssertStmt> stmt, IRDocsifier p) {
  // TODO: extract free vars from expr
  ExprDoc condition_expr = p->AsExprDoc(stmt.GetAttr(&tir::AssertStmtNode::condition));
  ExprDoc message_expr = p->AsExprDoc(stmt.GetAttr(&tir::AssertStmtNode::message));
  Array<StmtDoc> body = AsStmtDocArray(stmt.GetAttr(&tir::AssertStmtNode::body), p);

  ExprDoc assert_call_expr = TIR(p)->Attr("Assert")->Call({condition_expr, message_expr});
  StmtDoc assert_stmt =
      ExprStmtDoc(OperationDoc(OperationDocNode::Kind::kAssert, {condition_expr, message_expr}));
  return AsConciseScopedStmts(NullOpt, assert_call_expr, body, assert_stmt,
                              p->GetFrame<TIRFrame>().value());
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::AssertStmt>(PrintAssertStmt);

StmtDoc PrintStore(TracedObject<tir::Store> stmt, IRDocsifier p) {
  // TODO: extract free vars from expr
  Array<ExprDoc> args =
      AsExprDocArray({TracedObject<ObjectRef>(stmt.GetAttr(&tir::StoreNode::buffer_var)),
                      TracedObject<ObjectRef>(stmt.GetAttr(&tir::StoreNode::index)),
                      TracedObject<ObjectRef>(stmt.GetAttr(&tir::StoreNode::value)),
                      TracedObject<ObjectRef>(stmt.GetAttr(&tir::StoreNode::predicate))},
                     p);
  return ExprStmtDoc(TIR(p)->Attr("store")->Call(args));
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Store>(PrintStore);

StmtDoc PrintBufferStore(TracedObject<tir::BufferStore> stmt, IRDocsifier p) {
  Array<ExprDoc> indices = AsExprDocArray(stmt.GetAttr(&tir::BufferStoreNode::indices), p);
  Array<Doc> index_docs(indices.begin(), indices.end());
  return AssignDoc(p->AsExprDoc(stmt.GetAttr(&tir::BufferStoreNode::buffer))->Index(index_docs),
                   p->AsExprDoc(stmt.GetAttr(&tir::BufferStoreNode::value)), NullOpt);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferStore>(PrintBufferStore);

StmtDoc PrintIfThenElse(TracedObject<tir::IfThenElse> stmt, IRDocsifier p) {
  ExprDoc predicate = p->AsExprDoc(stmt.GetAttr(&tir::IfThenElseNode::condition));
  Array<StmtDoc> then_branch = AsStmtDocArray(stmt.GetAttr(&tir::IfThenElseNode::then_case), p);
  Array<StmtDoc> else_branch = AsStmtDocArray(stmt.GetAttr(&tir::IfThenElseNode::else_case), p);
  return IfDoc(predicate, then_branch, else_branch);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::IfThenElse>(PrintIfThenElse);

StmtDoc PrintWhile(TracedObject<tir::While> stmt, IRDocsifier p) {
  return WhileDoc(p->AsExprDoc(stmt.GetAttr(&tir::WhileNode::condition)),
                  AsStmtDocArray(stmt.GetAttr(&tir::WhileNode::body), p));
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::While>(PrintWhile);

StmtDoc PrintPrefetch(TracedObject<tir::Prefetch> stmt, IRDocsifier p) {
  auto buffer = stmt.GetAttr(&tir::PrefetchNode::buffer);
  auto bounds = stmt.GetAttr(&tir::PrefetchNode::bounds);
  return ExprStmtDoc(TIR(p)->Attr("prefetch")->Call({p->AsExprDoc(buffer), AsListDoc(bounds, p)}));
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Prefetch>(PrintPrefetch);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Evaluate>([](TracedObject<tir::Evaluate> stmt, IRDocsifier p) {
      return ExprStmtDoc(p->AsExprDoc(stmt.GetAttr(&tir::EvaluateNode::value)));
    });

StmtBlockDoc PrintLetStmt(TracedObject<tir::LetStmt> stmt, IRDocsifier p) {
  auto current_frame = p->GetFrame<TIRFrame>().value();
  TIRGeneralFrame new_frame(p->sym);
  WithCtx with_frame = p->WithFrame(new_frame);

  auto var = stmt.GetAttr(&tir::LetStmtNode::var);
  IdDoc var_doc = DefineVariable(var, new_frame);

  // TODO: PrintNonHeaderBufferDeclarations
  Array<StmtDoc> body = AsStmtDocArray(stmt.GetAttr(&tir::LetStmtNode::body), p);

  auto value_doc = p->AsExprDoc(stmt.GetAttr(&tir::LetStmtNode::value));
  auto dtype = var.GetAttr(&tir::VarNode::dtype);
  auto type_annotation_doc = GetTypeAnnotationDocForVar(var, p);

  if (current_frame->allow_concise_scoping_) {
    AssignDoc var_def = AssignDoc(var_doc, value_doc, type_annotation_doc);
    return StmtBlockDoc(runtime::Concat({var_def}, body));
  } else {
    StmtDoc var_def_doc =
        AssignDoc(var_doc, TIR(p)->Attr("var")->Call({DType2Literal(dtype)}), type_annotation_doc);
    ExprDoc let_call = TIR(p)->Attr("let")->Call({var_doc, value_doc});
    return StmtBlockDoc({var_def_doc, ScopeDoc(NullOpt, let_call, body)});
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::LetStmt>(PrintLetStmt);

namespace {
struct AllocUsage {
  TracedOptional<tir::Buffer> alloc_buffer;
  std::vector<TracedObject<tir::Buffer>> aliasing_buffers;
};

template <typename AllocRef>
AllocUsage FindAllocateUsage(const TracedObject<AllocRef>& op) {
  auto body = op.template GetAttr(&AllocRef::ContainerType::body);
  std::vector<TracedObject<tir::Buffer>> buffer_usage =
      FindBufferVarUsage(op.Get()->buffer_var, body);

  auto is_exact_match = [](tir::Buffer a, const AllocRef& b) {
    if (a->dtype != b->dtype) return false;
    if (a->shape.size() != b->extents.size()) return false;

    arith::Analyzer analyzer;
    for (size_t i = 0; i < a->shape.size(); i++) {
      if (!analyzer.CanProveEqual(a->shape[i], b->extents[i])) {
        return false;
      }
    }
    return true;
  };

  // If the buffer allocated via T.allocate is an exact match to the
  // usage of the buffer later on, then that buffer is the return
  // value of T.allocate, and no T.buffer_decl statement is needed.
  AllocUsage ret = {TracedOptional<tir::Buffer>(NullOpt, ObjectPath{}), {}};
  for (auto buf : buffer_usage) {
    if (!ret.alloc_buffer.defined() && is_exact_match(buf.Get(), op.Get())) {
      ret.alloc_buffer = buf;
    } else {
      ret.aliasing_buffers.push_back(buf);
    }
  }
  return ret;
}

IdDoc DefineAllocBuffer(const TracedOptional<tir::Buffer>& alloc_buffer,
                        const TracedObject<tir::Allocate>& stmt, const Frame& frame) {
  if (alloc_buffer.defined()) {
    IdDoc id = DefineBuffer(alloc_buffer.value(), frame);
    DefineBufferDataVariable(alloc_buffer.value().Get(), frame);
    return id;
  } else {
    auto buffer_var = stmt.GetAttr(&tir::AllocateNode::buffer_var);
    auto buffer_var_name_hint = buffer_var.GetAttr(&tir::VarNode::name_hint);
    String buffer_name = frame->sym->GetUniqueName(buffer_var_name_hint.Get());
    frame->DefByDoc(stmt.Get()->buffer_var, [buffer_name](ObjectPath path) {
      auto ret = IdDoc(buffer_name)->Attr("data");
      ret->paths.push_back(path);
      return ret;
    });

    IdDoc id(buffer_name);
    id->paths.push_back(buffer_var_name_hint.GetPath());
    return id;
  }
}

tir::Buffer CreateFakeBuffer(const tir::Allocate& allocate_stmt) {
  return tir::Buffer(allocate_stmt->buffer_var, allocate_stmt->dtype, allocate_stmt->extents, {}, 0,
                     allocate_stmt->buffer_var->name_hint, 0, 0, tir::kDefault);
}

LiteralDoc GetStorageScope(const TracedObject<tir::Allocate>& stmt) {
  auto buffer_var = stmt.GetAttr(&tir::AllocateNode::buffer_var);
  auto type = buffer_var.GetAttr(&tir::VarNode::type_annotation).Downcast<PointerType>();
  auto scope = type.GetAttr(&PointerTypeNode::storage_scope);
  return LiteralDoc::Str(scope);
}

}  // namespace

StmtBlockDoc PrintAllocate(TracedObject<tir::Allocate> stmt, IRDocsifier p) {
  // TODO: Print aliasing buffers
  AllocUsage usage = FindAllocateUsage(stmt);

  auto f_var_defined = [&p](const tir::VarNode* var) -> bool {
    return p->sym->IsObjectDefined(GetRef<tir::Var>(var));
  };
  std::unordered_map<const tir::VarNode*, ObjectPath> var_explicit_def;
  AssociatedVariables associated_vars;
  if (usage.alloc_buffer.defined()) {
    GetBufferPrintInfo({usage.alloc_buffer.value()}, f_var_defined, &var_explicit_def,
                       associated_vars);
  } else {
    associated_vars.AssociateIfNotAlready(stmt.Get()->buffer_var.get(),
                                          CreateFakeBuffer(stmt.Get()));
  }
  std::vector<BufferPrintInfo> aliasing_buffer_infos =
      GetBufferPrintInfo(usage.aliasing_buffers, f_var_defined, &var_explicit_def, associated_vars);

  TIRFrame previous_frame = p->GetFrame<TIRFrame>().value();
  TIRGeneralFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);

  IdDoc alloc_buffer_id_doc = DefineAllocBuffer(usage.alloc_buffer, stmt, frame);

  Array<ExprDoc> alloc_args;
  Array<String> alloc_kwarg_keys;
  Array<ExprDoc> alloc_kwarg_values;

  auto extents = stmt.GetAttr(&tir::AllocateNode::extents);
  alloc_args.push_back(AsListDoc(extents, p));

  auto dtype = stmt.GetAttr(&tir::AllocateNode::dtype);
  alloc_args.push_back(DType2Literal(dtype));

  alloc_args.push_back(GetStorageScope(stmt));

  auto condition = stmt.GetAttr(&tir::AllocateNode::condition);
  if (!tir::is_one(condition.Get())) {
    alloc_args.push_back(p->AsExprDoc(condition));
  }

  auto annotations = stmt.GetAttr(&tir::AllocateNode::annotations);
  if (!annotations.empty()) {
    alloc_kwarg_keys.push_back("annotations");
    alloc_kwarg_values.push_back(AsDictDoc(annotations, p));
  }
  ExprDoc alloc_buffer_expr_doc =
      TIR(p)->Attr("allocate")->Call(alloc_args, alloc_kwarg_keys, alloc_kwarg_values);

  Array<StmtDoc> body;
  for (const BufferPrintInfo& aliasing_buffer_info : aliasing_buffer_infos) {
    IdDoc lhs = DefineBuffer(aliasing_buffer_info.buffer, frame);
    ExprDoc rhs = aliasing_buffer_info.AsCall(
        TIR(p)->Attr("buffer_decl"),
        [&p](const TracedObject<PrimExpr>& e) -> ExprDoc { return p->AsExprDoc(e); });
    body.push_back(AssignDoc(lhs, rhs, NullOpt));
  }

  body = runtime::Concat(body, AsStmtDocArray(stmt.GetAttr(&tir::AllocateNode::body), p));

  return AsConciseScopedStmts(alloc_buffer_id_doc, alloc_buffer_expr_doc, body,
                              previous_frame);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Allocate>(PrintAllocate);

StmtDoc PrintAllocateConst(TracedObject<tir::AllocateConst> stmt, IRDocsifier p) { throw; }

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::AllocateConst>(PrintAllocateConst);

StmtBlockDoc PrintAttrStmt(TracedObject<tir::AttrStmt> stmt, IRDocsifier p) {
  auto value_doc = p->AsExprDoc(stmt.GetAttr(&tir::AttrStmtNode::value));
  auto node = stmt.GetAttr(&tir::AttrStmtNode::node);
  auto body = stmt.GetAttr(&tir::AttrStmtNode::body);
  auto attr_key = stmt.GetAttr(&tir::AttrStmtNode::attr_key);

  if (node.IsInstance<tir::Buffer>() && attr_key.Get() == "realize_scope" &&
      body.IsInstance<tir::BufferRealize>()) {
    // BufferRealize
    auto realize = body.Downcast<tir::BufferRealize>();
    auto buffer = node.Downcast<tir::Buffer>();
    ICHECK(realize.Get()->buffer.same_as(buffer.Get()));

    Array<ExprDoc> realize_args;
    realize_args.push_back(p->AsExprDoc(buffer));
    realize_args.push_back(
        ListDoc(AsExprDocArray(realize.GetAttr(&tir::BufferRealizeNode::bounds), p)));
    realize_args.push_back(value_doc);

    auto condition = realize.GetAttr(&tir::BufferRealizeNode::condition);
    if (!tir::is_one(condition.Get())) {
      realize_args.push_back(p->AsExprDoc(condition));
    }

    Array<StmtDoc> body = AsStmtDocArray(realize.GetAttr(&tir::BufferRealizeNode::body), p);

    return AsConciseScopedStmts(TIR(p)->Attr("realize")->Call(realize_args), body,
                                p->GetFrame<TIRFrame>().value());
  } else if (node.IsInstance<tir::IterVar>() &&
             (attr_key.Get() == "thread_extent" || attr_key.Get() == "virtual_thread")) {
    // IterVar
    auto iter_var = node.Downcast<tir::IterVar>();
    auto var = iter_var.GetAttr(&tir::IterVarNode::var);
    TIRFrame previous_frame = p->GetFrame<TIRFrame>().value();
    TIRGeneralFrame new_frame(p->sym);
    WithCtx with_frame = p->WithFrame(new_frame);

    IdDoc var_doc = DefineVariable(var, new_frame);

    ExprDoc launch_thread_call = TIR(p)->Attr("launch_thread")->Call({var_doc, value_doc});
    Array<StmtDoc> body_docs = AsStmtDocArray(body, p);

    return AsConciseScopedStmts(launch_thread_call, body_docs, previous_frame);
  } else {
    // General Form
    ExprDoc attr_expr =
        TIR(p)->Attr("attr")->Call({p->AsExprDoc(node), LiteralDoc::Str(attr_key), value_doc});
    Array<StmtDoc> body_docs = AsStmtDocArray(body, p);
    return AsConciseScopedStmts(attr_expr, body_docs, p->GetFrame<TIRFrame>().value());
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

bool IsSimpleLoop(const TracedObject<tir::For>& stmt,
                  const std::vector<TracedObject<tir::For>>& previous_loops) {
  auto is_used_by_previous_loops = [&previous_loops](const tir::VarNode* v) {
    return std::find_if(previous_loops.begin(), previous_loops.end(),
                        [v](const TracedObject<tir::For>& for_stmt) {
                          return for_stmt.Get()->loop_var.get() == v;
                        }) != previous_loops.end();
  };
  return stmt.Get()->kind == tir::ForKind::kSerial && stmt.Get()->annotations.empty() &&
         tir::is_zero(stmt.Get()->min) &&
         !tir::UsesVar(stmt.Get()->min, is_used_by_previous_loops) &&
         !tir::UsesVar(stmt.Get()->extent, is_used_by_previous_loops);
}

StmtDoc PrintRegularLoop(const TracedObject<tir::For>& stmt, IRDocsifier p) {
  TIRLoopFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);
  // TODO: consider moving to constructor
  frame->loops.push_back(stmt.Get());

  auto loop_var = stmt.GetAttr(&tir::ForNode::loop_var);
  IdDoc loop_var_doc = DefineVariable(loop_var, frame);

  Array<ExprDoc> loop_var_vars;
  Array<String> loop_var_kwarg_keys;
  Array<ExprDoc> loop_var_kwarg_values;
  auto min = stmt.GetAttr(&tir::ForNode::min);
  auto extent = stmt.GetAttr(&tir::ForNode::extent);
  if (tir::is_zero(min.Get())) {
    auto extent_doc = p->AsExprDoc(extent);
    // Also source the doc to `min`, so that we have something to highlight
    extent_doc->paths.push_back(min.GetPath());
    loop_var_vars.push_back(extent_doc);
  } else {
    arith::Analyzer analyzer;
    loop_var_vars.push_back(p->AsExprDoc(min));
    auto raw_max = analyzer.Simplify(min.Get() + extent.Get());
    auto max = MakeTraced(raw_max, extent.GetPath());
    loop_var_vars.push_back(p->AsExprDoc(max));
  }
  auto thread_binding = stmt.GetAttr(&tir::ForNode::thread_binding);
  if (thread_binding.defined()) {
    loop_var_kwarg_keys.push_back("thread");
    loop_var_kwarg_values.push_back(
        LiteralDoc::Str(thread_binding.value().GetAttr(&tir::IterVarNode::thread_tag)));
  }
  auto annotations = stmt.GetAttr(&tir::ForNode::annotations);
  if (!annotations.empty()) {
    loop_var_kwarg_keys.push_back("annotations");
    loop_var_kwarg_values.push_back(AsDictDoc(annotations, p));
  }
  auto kind = stmt.GetAttr(&tir::ForNode::kind);
  auto kind_str = kind.ApplyFunc(tir::ForKind2String);
  ExprDoc loop_var_rhs =
      TIR(p)->Attr(kind_str)->Call(loop_var_vars, loop_var_kwarg_keys, loop_var_kwarg_values);

  Array<StmtDoc> body = AsStmtDocArray(stmt.GetAttr(&tir::ForNode::body), p);

  return ForDoc(loop_var_doc, loop_var_rhs, body);
}

StmtDoc PrintMergedSimpleLoops(const std::vector<TracedObject<tir::For>>& stmts, IRDocsifier p) {
  TIRLoopFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);

  Array<ExprDoc> loop_var_docs;
  Array<ExprDoc> loop_var_extent_docs;
  for (const TracedObject<tir::For>& loop : stmts) {
    frame->loops.push_back(loop.Get());
    auto loop_var = loop.GetAttr(&tir::ForNode::loop_var);
    loop_var_docs.push_back(DefineVariable(loop_var, frame));
    auto extent = loop.GetAttr(&tir::ForNode::extent);
    loop_var_extent_docs.push_back(p->AsExprDoc(extent));
  }

  ExprDoc loop_var_rhs = TIR(p)->Attr("grid")->Call(loop_var_extent_docs);

  Array<StmtDoc> body = AsStmtDocArray(stmts.back().GetAttr(&tir::ForNode::body), p);

  return ForDoc(TupleDoc(loop_var_docs), loop_var_rhs, body);
}

StmtDoc PrintFor(TracedObject<tir::For> stmt, IRDocsifier p) {
  std::vector<TracedObject<tir::For>> simple_loops;

  auto next_for = stmt;
  while (true) {
    if (!IsSimpleLoop(next_for, simple_loops)) {
      break;
    }
    simple_loops.push_back(next_for);

    auto body = next_for.GetAttr(&tir::ForNode::body);
    if (!body.Get()->IsInstance<tir::ForNode>()) {
      break;
    }

    next_for = body.Downcast<tir::For>();
  }

  if (simple_loops.size() > 1) {
    return PrintMergedSimpleLoops(simple_loops, p);
  } else {
    return PrintRegularLoop(stmt, p);
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::For>(PrintFor);

StmtBlockDoc PrintBlock(TracedObject<tir::Block> stmt, IRDocsifier p) {
  TIRLoopFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);
  Array<StmtDoc> body;

  auto alloc_buffers = stmt.GetAttr(&tir::BlockNode::alloc_buffers);

  auto f_var_defined = [&p](const tir::VarNode* var) -> bool {
    return p->sym->IsObjectDefined(GetRef<tir::Var>(var));
  };
  std::unordered_map<const tir::VarNode*, ObjectPath> var_explicit_def;
  AssociatedVariables associated_vars;
  std::vector<BufferPrintInfo> buffer_print_infos = GetBufferPrintInfo(
      std::vector<TracedObject<tir::Buffer>>(alloc_buffers.begin(), alloc_buffers.end()),
      f_var_defined, &var_explicit_def, associated_vars);

  for (const auto& buffer_print_info : buffer_print_infos) {
    const TracedObject<tir::Buffer>& buf = buffer_print_info.buffer;
    IdDoc lhs = DefineBuffer(buf, frame);
    DefineBufferDataVariable(buf.Get(), frame);
    ExprDoc rhs = buffer_print_info.AsCall(
        TIR(p)->Attr("alloc_buffer"),
        [&p](const TracedObject<PrimExpr>& e) -> ExprDoc { return p->AsExprDoc(e); });
    body.push_back(AssignDoc(lhs, rhs, NullOpt));
  }

  auto match_buffers = stmt.GetAttr(&tir::BlockNode::match_buffers);
  for (auto match_buffer : match_buffers) {
    auto buffer = match_buffer.GetAttr(&tir::MatchBufferRegionNode::buffer);
    auto source = match_buffer.GetAttr(&tir::MatchBufferRegionNode::source);

    IdDoc lhs = DefineBuffer(buffer, frame);
    BufferPrintInfo info =
        GetBufferPrintInfo({buffer}, f_var_defined, &var_explicit_def, associated_vars).at(0);
    ExprDoc rhs =
        info.AsCall(TIR(p)->Attr("match_buffer"), {p->AsExprDoc(source)},
                    [&p](const TracedObject<PrimExpr>& e) -> ExprDoc { return p->AsExprDoc(e); });
    body.push_back(AssignDoc(lhs, rhs, NullOpt));
  }

  auto init = stmt.GetAttr(&tir::BlockNode::init);
  if (init.defined()) {
    body.push_back(ScopeDoc(TIR(p)->Attr("init")->Call({}), AsStmtDocArray(init.value(), p)));
  }

  body = runtime::Concat(body, AsStmtDocArray(stmt.GetAttr(&tir::BlockNode::body), p));

  return StmtBlockDoc(body);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Block>(PrintBlock);

namespace {
struct BlockVarBinding {
  TracedObject<tir::IterVar> lhs;
  TracedObject<PrimExpr> rhs;

  BlockVarBinding(const TracedObject<tir::IterVar>& lhs, const TracedObject<PrimExpr>& rhs)
      : lhs(lhs), rhs(rhs) {}
};
}  // namespace

std::vector<std::vector<BlockVarBinding>> GetBlockVarGroups(
    const TracedArray<tir::IterVar>& iter_vars, const TracedArray<PrimExpr>& values,
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
    TracedObject<tir::IterVar> iter_var = iter_vars[i];
    TracedObject<PrimExpr> value = values[i];
    bool current_is_simple_remap = is_simple_remap(iter_var.Get(), value.Get());
    if (!(current_is_simple_remap && last_is_simple_remap)) {
      // Group continues iff. both current var and previous var are simple remapping
      result.emplace_back();
    }
    result.back().emplace_back(iter_var, value);
    last_is_simple_remap = current_is_simple_remap;
  }

  return result;
}

static String GetIterTypeStr(tir::IterVarType iter_type) {
  switch (iter_type) {
    case tir::kDataPar:
      return "spatial";
    case tir::kCommReduce:
      return "reduce";
    case tir::kOrdered:
      return "scan";
    case tir::kOpaque:
      return "opaque";
    default:
      LOG(FATAL) << "Unknown block var iter type: " << iter_type;
      break;
  }
}

AssignDoc PrintBlockVar(const BlockVarBinding& block_var_binding, IRDocsifier p) {
  ExprDoc lhs = p->AsExprDoc(block_var_binding.lhs.GetAttr(&tir::IterVarNode::var));
  auto iter_type = block_var_binding.lhs.GetAttr(&tir::IterVarNode::iter_type);
  auto iter_type_str = MakeTraced(GetIterTypeStr(iter_type.Get()), iter_type.GetPath());

  Array<ExprDoc> args;
  auto dom = block_var_binding.lhs.GetAttr(&tir::IterVarNode::dom);
  auto min = dom.GetAttr(&RangeNode::min);
  auto extent = dom.GetAttr(&RangeNode::extent);

  if (tir::is_zero(min.Get())) {
    auto extent_doc = p->AsExprDoc(extent);
    extent_doc->paths.push_back(min.GetPath());
    args.push_back(extent_doc);
  } else {
    auto max = MakeTraced(min.Get() + extent.Get(), extent.GetPath());
    args.push_back(TupleDoc({p->AsExprDoc(min), p->AsExprDoc(max)}));
  }
  args.push_back(p->AsExprDoc(block_var_binding.rhs));
  ExprDoc rhs = TIR(p)->Attr("axis")->Attr(iter_type_str)->Call(args);

  return AssignDoc(lhs, rhs, NullOpt);
}

AssignDoc PrintGroupedSimpleRemappingBlockVars(
    const std::vector<BlockVarBinding>& block_var_bindings, IRDocsifier p) {
  std::vector<ExprDoc> iter_var_ids;
  std::vector<ExprDoc> iter_value_docs;
  std::string iter_type_str;
  Array<ObjectPath> iter_type_paths;

  for (const BlockVarBinding& binding : block_var_bindings) {
    const TracedObject<tir::IterVar>& iter_var = binding.lhs;
    const TracedObject<PrimExpr>& iter_value = binding.rhs;

    iter_var_ids.emplace_back(p->AsExprDoc(iter_var));

    auto iter_type = iter_var.GetAttr(&tir::IterVarNode::iter_type);
    if (iter_type.Get() == tir::kDataPar) {
      iter_type_str += "S";
    } else if (iter_type.Get() == tir::kCommReduce) {
      iter_type_str += "R";
    } else {
      ICHECK(false);
    }
    iter_type_paths.push_back(iter_type.GetPath());
    iter_value_docs.emplace_back(p->AsExprDoc(iter_value));
  }

  auto iter_type_doc = LiteralDoc::Str(iter_type_str);
  iter_type_doc->paths = iter_type_paths;

  ExprDoc lhs = TupleDoc(iter_var_ids);
  ExprDoc rhs =
      TIR(p)->Attr("axis")->Attr("remap")->Call({iter_type_doc, ListDoc(iter_value_docs)});

  return AssignDoc(lhs, rhs, NullOpt);
}

Array<StmtDoc> PrintBlockVars(const TracedObject<tir::BlockRealize>& stmt, IRDocsifier p) {
  auto block = stmt.GetAttr(&tir::BlockRealizeNode::block);
  auto iter_vars = block.GetAttr(&tir::BlockNode::iter_vars);
  auto iter_values = stmt.GetAttr(&tir::BlockRealizeNode::iter_values);

  Map<tir::Var, tir::For> loop_var_map = GetLoopVarMap(p);
  Array<StmtDoc> result;
  for (const std::vector<BlockVarBinding>& binding_group :
       GetBlockVarGroups(iter_vars, iter_values, loop_var_map)) {
    ICHECK_NE(binding_group.size(), 0);
    if (binding_group.size() == 1) {
      result.push_back(PrintBlockVar(binding_group[0], p));
    } else {
      result.push_back(PrintGroupedSimpleRemappingBlockVars(binding_group, p));
    }
  }
  return result;
}

Array<StmtDoc> PrintBlockAttrs(const TracedObject<tir::BlockRealize>& stmt, IRDocsifier p) {
  std::vector<ExprDoc> attr_exprs;
  auto block = stmt.GetAttr(&tir::BlockRealizeNode::block);
  auto predicate = stmt.GetAttr(&tir::BlockRealizeNode::predicate);

  if (!tir::is_one(predicate.Get())) {
    attr_exprs.emplace_back(TIR(p)->Attr("where")->Call({p->AsExprDoc(predicate)}));
  }

  auto reads = block.GetAttr(&tir::BlockNode::reads);
  attr_exprs.push_back(TIR(p)->Attr("reads")->Call(AsExprDocArray(reads, p)));

  auto writes = block.GetAttr(&tir::BlockNode::writes);
  attr_exprs.push_back(TIR(p)->Attr("writes")->Call(AsExprDocArray(writes, p)));

  auto annotations = block.GetAttr(&tir::BlockNode::annotations);
  if (!annotations.empty()) {
    attr_exprs.push_back(TIR(p)->Attr("block_attr")->Call({AsDictDoc(annotations, p)}));
  }

  Array<StmtDoc> result;
  for (const ExprDoc& attr : attr_exprs) {
    result.push_back(ExprStmtDoc(attr));
  }

  return result;
}

StmtDoc PrintBlockRealize(TracedObject<tir::BlockRealize> stmt, IRDocsifier p) {
  auto block = stmt.GetAttr(&tir::BlockRealizeNode::block);
  auto block_name_hint = block.GetAttr(&tir::BlockNode::name_hint);

  Array<ExprDoc> block_args;
  if (!block_name_hint.Get().empty()) {
    block_args.push_back(LiteralDoc::Str(block_name_hint));
  }
  ExprDoc block_begin_expr = TIR(p)->Attr("block")->Call(block_args);

  TIRGeneralFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);
  for (auto iter_var : block.GetAttr(&tir::BlockNode::iter_vars)) {
    auto var = iter_var.GetAttr(&tir::IterVarNode::var);
    DefineVariable(var, frame);
  }

  Array<StmtDoc> body = PrintBlockVars(stmt, p);
  body = runtime::Concat(body, PrintBlockAttrs(stmt, p));
  body = runtime::Concat(body, AsStmtDocArray(block, p));

  return ScopeDoc(block_begin_expr, body);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BlockRealize>(PrintBlockRealize);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRealize>([](TracedObject<tir::BufferRealize> stmt,
                                         IRDocsifier p) -> Doc {
      LOG(FATAL) << "TVM Script Printer Internal Error: All the BufferRealize should be folded "
                    "with Attr";
      throw;
    });
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerStore>([](TracedObject<tir::ProducerStore> stmt,
                                         IRDocsifier p) -> Doc {
      LOG(FATAL) << "ProducerStore cannot be printed";
      throw;
    });
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerRealize>([](TracedObject<tir::ProducerRealize> stmt,
                                           IRDocsifier p) -> Doc {
      LOG(FATAL) << "ProducerRealize cannot be printed";
      throw;
    });
// • BlockNode: Class
// • BlockRealizeNode: Class
//
}  // namespace printer
}  // namespace script
}  // namespace tvm
