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

#include <tvm/arith/analyzer.h>
#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/traced_object_functor.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../utils.h"
#include "./allocate.h"
#include "./buffer.h"
#include "./tir.h"
#include "./var.h"

namespace tvm {
namespace script {
namespace printer {

/*
 * \brief Helper to print stmt in the concise scoping form.
 *
 * For example, the allocate statment in TIR can be written as
 * \code
 * ...
 * with T.allocate([16], "float32", "global") as buf:
 *     buf[0] = 0.0  # inside the allocate
 * T.evaluate(T.call_extern(...))  # outside the allocate
 * \endcode
 * This representation is ambiguilty-free, but it adds one extra indent to
 * the code, which reduces readability if multiple statements are nested together.
 *
 * If the allocate statement is the last statement in its parent, it can be
 * written in the concise scoping form, avoiding adding extra level of indent.
 * \code
 * ...
 * buf = T.allocate([16], "float32", "global")
 * buf[0] = 0.0
 * ...
 * \endcode
 *
 * This builder class helps print stmt in the concise scoping form. The attributes
 * of this builder map to the output as,
 * \code
 * # Normal form
 * with <parent_expr> as <target>:
 *     <body>
 *
 * # Concise form
 * <target> = <parent_expr>
 * <body>
 *
 * # Concise form if the `concise_stmt_override` is defined
 * <concise_stmt_override>
 * <body>
 *
 * \endcode
 *
 */
class ConciseScopedStmtBuilder {
 public:
  Optional<ExprDoc> target{NullOpt};
  ExprDoc parent_expr{nullptr};
  Array<StmtDoc> body;
  Optional<StmtDoc> concise_stmt_override{NullOpt};

  ConciseScopedStmtBuilder() {}

  using TSelf = ConciseScopedStmtBuilder;

  TSelf& WithBody(Array<StmtDoc> body) {
    this->body = body;
    return *this;
  }

  TSelf& WithConciseFormStmt(StmtDoc stmt) {
    this->concise_stmt_override = stmt;
    return *this;
  }

  TSelf& WithTarget(ExprDoc target) {
    this->target = target;
    return *this;
  }

  TSelf& WithParentExpr(ExprDoc expr) {
    this->parent_expr = expr;
    return *this;
  }

  StmtBlockDoc ToDoc(const IRDocsifier& p) { return ToDoc(p->GetFrame<TIRFrame>().value()); }

  StmtBlockDoc ToDoc(const TIRFrame& frame) {
    ICHECK(parent_expr.defined());
    if (frame->allow_concise_scoping) {
      StmtDoc first_doc = ExprStmtDoc(parent_expr);
      if (concise_stmt_override) {
        first_doc = concise_stmt_override.value();
      } else if (target.defined()) {
        first_doc = AssignDoc(target.value(), parent_expr, NullOpt);
      }

      return StmtBlockDoc(runtime::Concat({first_doc}, body));
    } else {
      return StmtBlockDoc({ScopeDoc(target, parent_expr, body)});
    }
  }
};

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::SeqStmt>([](TracedObject<tir::SeqStmt> stmt, IRDocsifier p) -> Doc {
      if (!p->frames.back()->IsInstance<TIRTopLevelFrameNode>()) {
        // Throw error
        LOG(FATAL) << "tir::SeqStmt can only be printed when it's the top level statement. "
                      "Use AsStmtDocArray to print the body of statement";
        throw;
      }
      return StmtBlockDoc(AsStmtDocArray(stmt, p));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::AssertStmt>([](TracedObject<tir::AssertStmt> stmt, IRDocsifier p) {
      ExprDoc condition_expr = p->AsExprDoc(stmt.GetAttr(&tir::AssertStmtNode::condition));
      ExprDoc message_expr = p->AsExprDoc(stmt.GetAttr(&tir::AssertStmtNode::message));
      Array<StmtDoc> body = AsStmtDocArray(stmt.GetAttr(&tir::AssertStmtNode::body), p);

      return ConciseScopedStmtBuilder()
          .WithParentExpr(TIR(p)->Attr("Assert")->Call({condition_expr, message_expr}))
          .WithConciseFormStmt(AssertDoc(condition_expr, message_expr))
          .WithBody(body)
          .ToDoc(p);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferStore>([](TracedObject<tir::BufferStore> stmt, IRDocsifier p) {
      Array<ExprDoc> indices = AsExprDocArray(stmt.GetAttr(&tir::BufferStoreNode::indices), p);
      Array<Doc> index_docs(indices.begin(), indices.end());
      return AssignDoc(p->AsExprDoc(stmt.GetAttr(&tir::BufferStoreNode::buffer))[index_docs],
                       p->AsExprDoc(stmt.GetAttr(&tir::BufferStoreNode::value)), NullOpt);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::IfThenElse>([](TracedObject<tir::IfThenElse> stmt, IRDocsifier p) {
      ExprDoc predicate = p->AsExprDoc(stmt.GetAttr(&tir::IfThenElseNode::condition));
      Array<StmtDoc> then_branch = AsStmtDocArray(stmt.GetAttr(&tir::IfThenElseNode::then_case), p);
      Array<StmtDoc> else_branch = AsStmtDocArray(stmt.GetAttr(&tir::IfThenElseNode::else_case), p);
      return IfDoc(predicate, then_branch, else_branch);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::While>([](TracedObject<tir::While> stmt, IRDocsifier p) {
      return WhileDoc(p->AsExprDoc(stmt.GetAttr(&tir::WhileNode::condition)),
                      AsStmtDocArray(stmt.GetAttr(&tir::WhileNode::body), p));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Prefetch>([](TracedObject<tir::Prefetch> stmt, IRDocsifier p) {
      auto buffer = stmt.GetAttr(&tir::PrefetchNode::buffer);
      auto bounds = stmt.GetAttr(&tir::PrefetchNode::bounds);
      return ExprStmtDoc(
          TIR(p)->Attr("prefetch")->Call({p->AsExprDoc(buffer)[AsDocArray<Doc>(bounds, p)]}));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::LetStmt>([](TracedObject<tir::LetStmt> stmt, IRDocsifier p) {
      TIRFrame previous_frame = p->GetFrame<TIRFrame>().value();
      TIRGeneralFrame let_frame;
      WithCtx ctx = p->WithFrame(let_frame);

      auto var = stmt.GetAttr(&tir::LetStmtNode::var);
      bool is_var_defined_previously = p->vars->IsVarDefined(var.Get());
      ExprDoc var_doc{nullptr};
      if (is_var_defined_previously) {
        var_doc = p->vars->GetVarDoc(var).value();
      } else {
        var_doc = p->vars->Define(var.Get(), GetVarNameHint(var), let_frame);
      }

      auto value_doc = p->AsExprDoc(stmt.GetAttr(&tir::LetStmtNode::value));
      auto dtype = var.GetAttr(&tir::VarNode::dtype);
      auto type_annotation_doc = GetTypeAnnotationForVar(var, p);

      TracedObject<tir::Stmt> body_stmt = stmt.GetAttr(&tir::LetStmtNode::body);
      Array<StmtDoc> body_doc;

      // Print definition of buffers that aliases the variable of this Let stmt.
      std::vector<TracedObject<tir::Buffer>> aliasing_buffers =
          FindAliasingBuffers(var.Get(), body_stmt);
      std::vector<TracedObject<tir::Buffer>> buffers_to_define;
      for (const TracedObject<tir::Buffer>& buffer : aliasing_buffers) {
        if (!p->vars->IsVarDefined(buffer.Get())) {
          buffers_to_define.push_back(buffer);
        }
      }
      DefineBuffers(buffers_to_define, {}, let_frame, p, TIR(p)->Attr("decl_buffer"),
                    [&body_doc](IdDoc buf_identifier, ExprDoc buf_definition) {
                      body_doc.push_back(AssignDoc(buf_identifier, buf_definition, NullOpt));
                    });

      body_doc = runtime::Concat(body_doc, AsStmtDocArray(body_stmt, p));

      if (previous_frame->allow_concise_scoping) {
        // dtype won't be linked to a doc object if it does concise scoping
        // here we manually link it to type annotation
        type_annotation_doc->source_paths.push_back(dtype.GetPath());
        AssignDoc var_assignment = AssignDoc(var_doc, value_doc, type_annotation_doc);
        return StmtBlockDoc(runtime::Concat({var_assignment}, body_doc));
      } else {
        Array<StmtDoc> result;
        if (!is_var_defined_previously) {
          result.push_back(AssignDoc(var_doc, TIR(p)->Attr("var")->Call({DType2Literal(dtype)}),
                                     type_annotation_doc));
        }
        ExprDoc let_call = TIR(p)->Attr("let")->Call({var_doc, value_doc});
        result.push_back(ScopeDoc(NullOpt, let_call, body_doc));
        return StmtBlockDoc(result);
      }
    });

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

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Allocate>([](TracedObject<tir::Allocate> stmt, IRDocsifier p) {
      AllocateUsage allocate_usage = FindAllocateUsage(stmt);

      // If there is no exact matched buffer in the Allocate body
      // Create a dummy one so the definition of other aliasing
      // buffers can refer to its data pointer.
      TracedObject<tir::Buffer> allocated_buffer = MakeTraced(CreateFakeBuffer(stmt.Get()));
      if (allocate_usage.alloc_buffer.defined()) {
        allocated_buffer = allocate_usage.alloc_buffer.value();
      }

      TIRFrame previous_frame = p->GetFrame<TIRFrame>().value();
      TIRGeneralFrame current_frame;
      WithCtx with_frame = p->WithFrame(current_frame);

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

      IdDoc alloc_buffer_id_doc = DefineBuffers({allocated_buffer}, {}, current_frame, p,
                                                TIR(p)->Attr("Buffer"), [](IdDoc, ExprDoc) {})[0];
      alloc_buffer_id_doc->source_paths = {stmt.GetAttr(&tir::AllocateNode::buffer_var).GetPath()};

      Array<StmtDoc> body_doc;
      DefineBuffers(allocate_usage.aliasing_buffers, {}, current_frame, p,
                    TIR(p)->Attr("decl_buffer"),
                    [&body_doc](IdDoc buf_identifier, ExprDoc buf_definition) {
                      body_doc.push_back(AssignDoc(buf_identifier, buf_definition, NullOpt));
                    });
      body_doc =
          runtime::Concat(body_doc, AsStmtDocArray(stmt.GetAttr(&tir::AllocateNode::body), p));

      return ConciseScopedStmtBuilder()
          .WithParentExpr(alloc_buffer_expr_doc)
          .WithTarget(alloc_buffer_id_doc)
          .WithBody(body_doc)
          .ToDoc(previous_frame);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::AttrStmt>([](TracedObject<tir::AttrStmt> stmt, IRDocsifier p) {
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

        Array<Doc> indices = AsDocArray<Doc>(realize.GetAttr(&tir::BufferRealizeNode::bounds), p);
        Array<ExprDoc> realize_args;
        realize_args.push_back(p->AsExprDoc(buffer)[indices]);
        realize_args.push_back(value_doc);

        auto condition = realize.GetAttr(&tir::BufferRealizeNode::condition);
        if (!tir::is_one(condition.Get())) {
          realize_args.push_back(p->AsExprDoc(condition));
        }

        return ConciseScopedStmtBuilder()
            .WithParentExpr(TIR(p)->Attr("realize")->Call(realize_args))
            .WithBody(AsStmtDocArray(realize.GetAttr(&tir::BufferRealizeNode::body), p))
            .ToDoc(p);
      } else if (node.IsInstance<tir::IterVar>() &&
                 (attr_key.Get() == "thread_extent" || attr_key.Get() == "virtual_thread")) {
        // IterVar
        auto iter_var = node.Downcast<tir::IterVar>();
        TIRFrame previous_frame = p->GetFrame<TIRFrame>().value();
        TIRGeneralFrame new_frame;
        WithCtx with_frame = p->WithFrame(new_frame);

        // TODO(yelite): When implementing the PrimFunc printing, the logic here
        // needs to change, putting variable def into PrimFuncFrame if it exists.
        TIRTopLevelFrame top_level_frame = p->GetFrame<TIRTopLevelFrame>().value();
        auto add_env_thread = [&defs = top_level_frame->env_thread_definitions](
                                  tir::Var var, AssignDoc env_thread_doc) {
          if (defs.count(var) == 0) {
            defs.Set(var, env_thread_doc);
          }
        };

        ExprDoc launch_thread_call = IterVarLaunchThread(
            iter_var, stmt.GetAttr(&tir::AttrStmtNode::value), new_frame, p, add_env_thread);

        return ConciseScopedStmtBuilder()
            .WithParentExpr(launch_thread_call)
            .WithBody(AsStmtDocArray(body, p))
            .ToDoc(previous_frame);
      } else {
        // General Form
        ExprDoc attr_expr =
            TIR(p)->Attr("attr")->Call({p->AsExprDoc(node), LiteralDoc::Str(attr_key), value_doc});
        return ConciseScopedStmtBuilder()
            .WithParentExpr(attr_expr)
            .WithBody(AsStmtDocArray(body, p))
            .ToDoc(p);
      }
    });

Map<tir::Var, tir::For> GetLoopVarMap(IRDocsifier p) {
  Map<tir::Var, tir::For> result;
  for (const TIRLoopFrame& frame : GetFrames<TIRLoopFrame>(p)) {
    for (const tir::For& loop : frame->loops) {
      result.Set(loop->loop_var, loop);
    }
  }
  return result;
}

bool IsSimpleLoop(const TracedObject<tir::For>& stmt,
                  const std::vector<TracedObject<tir::For>>& previous_loops) {
  auto is_var_of_previous_loops = [&previous_loops](const tir::VarNode* v) {
    return std::find_if(previous_loops.begin(), previous_loops.end(),
                        [v](const TracedObject<tir::For>& for_stmt) {
                          return for_stmt.Get()->loop_var.get() == v;
                        }) != previous_loops.end();
  };
  return stmt.Get()->kind == tir::ForKind::kSerial && stmt.Get()->annotations.empty() &&
         tir::is_zero(stmt.Get()->min) &&
         !tir::UsesVar(stmt.Get()->min, is_var_of_previous_loops) &&
         !tir::UsesVar(stmt.Get()->extent, is_var_of_previous_loops);
}

StmtDoc PrintRegularLoop(const TracedObject<tir::For>& stmt, IRDocsifier p) {
  TIRLoopFrame frame(stmt.Get());
  WithCtx with_frame = p->WithFrame(frame);

  auto loop_var = stmt.GetAttr(&tir::ForNode::loop_var);
  IdDoc loop_var_doc = DefineVar(loop_var, frame, p);

  Array<ExprDoc> loop_var_vars;
  Array<String> loop_var_kwarg_keys;
  Array<ExprDoc> loop_var_kwarg_values;
  auto min = stmt.GetAttr(&tir::ForNode::min);
  auto extent = stmt.GetAttr(&tir::ForNode::extent);
  if (tir::is_zero(min.Get())) {
    auto extent_doc = p->AsExprDoc(extent);
    // Also source the doc to `min`, so that we have something to highlight
    extent_doc->source_paths.push_back(min.GetPath());
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
  auto kind_str =
      kind.ApplyFunc([](tir::ForKind kind) { return String(tir::ForKind2String(kind)); });
  ExprDoc loop_var_rhs =
      TIR(p)->Attr(kind_str)->Call(loop_var_vars, loop_var_kwarg_keys, loop_var_kwarg_values);

  Array<StmtDoc> body = AsStmtDocArray(stmt.GetAttr(&tir::ForNode::body), p);

  return ForDoc(loop_var_doc, loop_var_rhs, body);
}

StmtDoc PrintMergedSimpleLoops(const std::vector<TracedObject<tir::For>>& stmts, IRDocsifier p) {
  TIRLoopFrame frame;
  WithCtx with_frame = p->WithFrame(frame);

  Array<ExprDoc> loop_var_docs;
  Array<ExprDoc> loop_var_extent_docs;
  for (const TracedObject<tir::For>& loop : stmts) {
    frame->loops.push_back(loop.Get());
    auto loop_var = loop.GetAttr(&tir::ForNode::loop_var);
    loop_var_docs.push_back(DefineVar(loop_var, frame, p));
    auto extent = loop.GetAttr(&tir::ForNode::extent);
    loop_var_extent_docs.push_back(p->AsExprDoc(extent));
  }

  ExprDoc loop_var_rhs = TIR(p)->Attr("grid")->Call(loop_var_extent_docs);

  Array<StmtDoc> body = AsStmtDocArray(stmts.back().GetAttr(&tir::ForNode::body), p);

  return ForDoc(TupleDoc(loop_var_docs), loop_var_rhs, body);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::For>([](TracedObject<tir::For> stmt, IRDocsifier p) {
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
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Block>([](TracedObject<tir::Block> stmt, IRDocsifier p) {
      TIRGeneralFrame frame;
      WithCtx with_frame = p->WithFrame(frame);

      Array<StmtDoc> body;

      // Print definition of the iter vars in order to print the block
      // fragment in a round-trippable way.
      for (auto iter_var : stmt.GetAttr(&tir::BlockNode::iter_vars)) {
        auto var = iter_var.GetAttr(&tir::IterVarNode::var);
        if (!p->vars->IsVarDefined(var.Get())) {
          IdDoc var_doc = p->AsDoc<IdDoc>(var);
          body.push_back(ExprStmtDoc(IterVarDef(iter_var, p)));
        }
      }

      auto reads = stmt.GetAttr(&tir::BlockNode::reads);
      if (!reads.empty()) {
        body.push_back(ExprStmtDoc(TIR(p)->Attr("reads")->Call(AsExprDocArray(reads, p))));
      }

      auto writes = stmt.GetAttr(&tir::BlockNode::writes);
      if (!writes.empty()) {
        body.push_back(ExprStmtDoc(TIR(p)->Attr("writes")->Call(AsExprDocArray(writes, p))));
      }

      auto annotations = stmt.GetAttr(&tir::BlockNode::annotations);
      if (!annotations.empty()) {
        body.push_back(ExprStmtDoc(TIR(p)->Attr("block_attr")->Call({AsDictDoc(annotations, p)})));
      }

      auto alloc_buffers = stmt.GetAttr(&tir::BlockNode::alloc_buffers);
      std::vector<TracedObject<tir::Buffer>> buffers_to_allocate(alloc_buffers.begin(),
                                                                 alloc_buffers.end());
      DefineBuffers(
          buffers_to_allocate, {}, frame, p, TIR(p)->Attr("alloc_buffer"),
          [&body](IdDoc buf, ExprDoc def) { body.push_back(AssignDoc(buf, def, NullOpt)); });

      auto match_buffers = stmt.GetAttr(&tir::BlockNode::match_buffers);
      for (auto match_buffer : match_buffers) {
        auto buffer = match_buffer.GetAttr(&tir::MatchBufferRegionNode::buffer);
        auto source = match_buffer.GetAttr(&tir::MatchBufferRegionNode::source);

        DefineBuffers(
            {buffer}, {p->AsExprDoc(source)}, frame, p, TIR(p)->Attr("match_buffer"),
            [&body](IdDoc buf, ExprDoc def) { body.push_back(AssignDoc(buf, def, NullOpt)); });
      }

      auto init = stmt.GetAttr(&tir::BlockNode::init);
      if (init.defined()) {
        body.push_back(ScopeDoc(TIR(p)->Attr("init")->Call({}), AsStmtDocArray(init.value(), p)));
      }

      body = runtime::Concat(body, AsStmtDocArray(stmt.GetAttr(&tir::BlockNode::body), p));

      return StmtBlockDoc(body);
    });

struct BlockVarBinding {
  TracedObject<tir::IterVar> lhs;
  TracedObject<PrimExpr> rhs;

  BlockVarBinding(const TracedObject<tir::IterVar>& lhs, const TracedObject<PrimExpr>& rhs)
      : lhs(lhs), rhs(rhs) {}
};

std::vector<std::vector<BlockVarBinding>> GetBlockVarGroups(
    const TracedArray<tir::IterVar>& iter_vars, const TracedArray<PrimExpr>& values,
    const Map<tir::Var, tir::For>& loop_var_map) {
  ICHECK_EQ(iter_vars.size(), values.size());
  ICHECK(iter_vars.size() > 0);

  std::vector<std::vector<BlockVarBinding>> result;

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
    if (!(current_is_simple_remap && last_is_simple_remap) || result.empty()) {
      // Add new group if either:
      // 1. The result vector is empty.
      // 2. At least one of the current var and previous var is not simple remap
      result.emplace_back();
    }
    result.back().emplace_back(iter_var, value);
    last_is_simple_remap = current_is_simple_remap;
  }

  return result;
}

AssignDoc PrintBlockVar(const BlockVarBinding& block_var_binding, IRDocsifier p) {
  const auto& iter_var = block_var_binding.lhs;
  ExprDoc lhs = p->AsExprDoc(iter_var.GetAttr(&tir::IterVarNode::var));
  ExprDoc rhs = IterVarBlockVar(iter_var, block_var_binding.rhs, p);
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

    iter_var_ids.emplace_back(p->AsExprDoc(iter_var.GetAttr(&tir::IterVarNode::var)));

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
  iter_type_doc->source_paths = iter_type_paths;

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

  Array<StmtDoc> result;
  for (const ExprDoc& attr : attr_exprs) {
    result.push_back(ExprStmtDoc(attr));
  }

  return result;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BlockRealize>([](TracedObject<tir::BlockRealize> stmt, IRDocsifier p) {
      auto block = stmt.GetAttr(&tir::BlockRealizeNode::block);
      auto block_name_hint = block.GetAttr(&tir::BlockNode::name_hint);

      Array<ExprDoc> block_args;
      if (!block_name_hint.Get().empty()) {
        block_args.push_back(LiteralDoc::Str(block_name_hint));
      }
      ExprDoc block_begin_expr = TIR(p)->Attr("block")->Call(block_args);

      TIRGeneralFrame frame;
      WithCtx with_frame = p->WithFrame(frame);

      for (auto iter_var : block.GetAttr(&tir::BlockNode::iter_vars)) {
        auto var = iter_var.GetAttr(&tir::IterVarNode::var);
        DefineVar(var, frame, p);
      }

      Array<StmtDoc> body = PrintBlockVars(stmt, p);
      body = runtime::Concat(body, PrintBlockAttrs(stmt, p));
      body = runtime::Concat(body, AsStmtDocArray(block, p));

      return ScopeDoc(block_begin_expr, body);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Evaluate>([](TracedObject<tir::Evaluate> stmt, IRDocsifier p) {
      return ExprStmtDoc(p->AsExprDoc(stmt.GetAttr(&tir::EvaluateNode::value)));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::Store>([](TracedObject<tir::Store> stmt, IRDocsifier p) -> Doc {
      LOG(FATAL) << "tir::Store cannot be printed. Store is replaced by BufferStore.";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::BufferRealize>([](TracedObject<tir::BufferRealize> stmt,
                                         IRDocsifier p) -> Doc {
      LOG(FATAL)
          << "tir::BufferRealize cannot be printed. All the BufferRealize should be nested inside "
             "with AttrStmt.";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerStore>([](TracedObject<tir::ProducerStore> stmt,
                                         IRDocsifier p) -> Doc {
      LOG(FATAL) << "tir::ProducerStore cannot be printed";
      throw;
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::ProducerRealize>([](TracedObject<tir::ProducerRealize> stmt,
                                           IRDocsifier p) -> Doc {
      LOG(FATAL) << "tir::ProducerRealize cannot be printed";
      throw;
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
