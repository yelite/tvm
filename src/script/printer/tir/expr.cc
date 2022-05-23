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

#include <tvm/node/functor.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include "../util.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

ExprDoc PrintStringImm(tir::StringImm s, IRDocsifier p) { return LiteralDoc::Str(s->value); }
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::StringImm>(PrintStringImm);

ExprDoc PrintIntImm(IntImm i, IRDocsifier p) { return LiteralDoc::Int(i); }
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<IntImm>(PrintIntImm);

ExprDoc PrintFloatImm(FloatImm f, IRDocsifier p) { return LiteralDoc::Float(f); }
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<FloatImm>(PrintFloatImm);

ExprDoc PrintCast(tir::Cast e, IRDocsifier p) {
  return TIR(p)->Attr("cast")->Call({p->AsExprDoc(e->value), DType2Literal(e->dtype)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Cast>(PrintCast);

template <typename BinOpType>
OperationDocNode::Kind GetBinaryOpKind() { throw; }

template <typename BinOpType>
ExprDoc PrintBinOp(BinOpType e, IRDocsifier p) {
  return OperationDoc(GetBinaryOpKind<BinOpType>(), {p->AsExprDoc(e->a), p->AsExprDoc(e->b)});
}

template<>
OperationDocNode::Kind GetBinaryOpKind<tir::AddNode>() { return OperationDocNode::Kind::kAdd; }
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Add>(PrintBinOp<tir::Add>);

ExprDoc PrintSelect(tir::Select e, IRDocsifier p) {
  return TIR(p)->Attr("Select")->Call(
      {p->AsExprDoc(e->condition), p->AsExprDoc(e->true_value), p->AsExprDoc(e->false_value)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Select>(PrintSelect);

ExprDoc PrintBufferLoad(tir::BufferLoad e, IRDocsifier p) {
  ExprDoc base = p->AsExprDoc(e->buffer);
  return base->Index(AsDocArray<Doc>(e->indices, p));
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferLoad>(PrintBufferLoad);

ExprDoc PrintProducerLoad(tir::ProducerLoad e, IRDocsifier p) {
  LOG(FATAL) << "Cannot print a tir.ProducerLoad as it is not valid in TIR Primfuncs. You need to "
                "lower this function first.";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::ProducerLoad>(PrintProducerLoad);

ExprDoc PrintLoad(tir::Load e, IRDocsifier p) {
  LOG(FATAL) << "Cannot print a tir.Load";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Load>(PrintLoad);

ExprDoc PrintRamp(tir::Ramp e, IRDocsifier p) {
  return TIR(p)->Attr("ramp")->Call(
      {p->AsExprDoc(e->base), p->AsExprDoc(e->stride), LiteralDoc::Int(e->lanes)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Ramp>(PrintRamp);

ExprDoc PrintBroadcast(tir::Broadcast e, IRDocsifier p) {
  return TIR(p)->Attr("broadcast")->Call({p->AsExprDoc(e->value), LiteralDoc::Int(e->lanes)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Broadcast>(PrintBroadcast);

ExprDoc PrintLet(tir::Let e, IRDocsifier p) {
  return TIR(p)->Attr("let")->Call(
      {p->AsExprDoc(e->var), p->AsExprDoc(e->value), p->AsExprDoc(e->body)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Let>(PrintLet);

ExprDoc PrintCall(tir::Call e, IRDocsifier p) {
  if (e->op->IsInstance<OpNode>()) {
    return PrintOpCall(e, p);
  } else {
    const auto* op_gvar = e->op.as<GlobalVarNode>();
    ICHECK(op_gvar != nullptr);
    return IdDoc(op_gvar->name_hint)->Call(AsExprDocArray(e->args, p));
  }
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Call>(PrintCall);

ExprDoc PrintShuffle(tir::Shuffle e, IRDocsifier p) {
  return TIR(p)->Attr("shuffle")->Call({p->AsExprDoc(e->vectors), p->AsExprDoc(e->indices)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Shuffle>(PrintShuffle);

ExprDoc PrintCommReducer(tir::CommReducer e, IRDocsifier p) {
  TIRGeneralFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);

  Array<IdDoc> reducer_args;
  for (const tir::Var& v_lhs : e->lhs) {
    IdDoc var_doc = frame->DefByName(v_lhs, p->sym->GetUniqueName(v_lhs->name_hint));
    reducer_args.push_back(var_doc);
  }
  for (const tir::Var& v_rhs : e->rhs) {
    IdDoc var_doc = frame->DefByName(v_rhs, p->sym->GetUniqueName(v_rhs->name_hint));
    reducer_args.push_back(var_doc);
  }

  ExprDoc reducer_body = TupleDoc();
  if (e->rhs.size() == 1) {
    reducer_body = p->AsExprDoc(e->result[0]);
  } else {
    reducer_body = TupleDoc(AsExprDocArray(e->result, p));
  }

  LambdaDoc reducer{reducer_args, reducer_body};
  ListDoc identity_elements{AsExprDocArray(e->identity_element, p)};

  return TIR(p)->Attr("comm_reducer")->Call({reducer, identity_elements});
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::CommReducer>(PrintCommReducer);

ExprDoc PrintReduce(tir::Reduce e, IRDocsifier p) {
  return TIR(p)->Attr("reduce")->Call({p->AsExprDoc(e->combiner), p->AsExprDoc(e->source),
                                       p->AsExprDoc(e->axis), LiteralDoc::Int(e->value_index)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Reduce>(PrintReduce);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<Range>([](Range e, IRDocsifier p) {
  return SliceDoc(p->AsExprDoc(e->min), p->AsExprDoc(e->min + e->extent));
});

ExprDoc PrintAny(tir::Any e, IRDocsifier p) {
  LOG(FATAL) << "Cannot print any shape";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Any>(PrintAny);

ExprDoc PrintBufferRegion(tir::BufferRegion buffer_region, IRDocsifier p) {
  Array<Doc> indices;

  for (const Range& range : buffer_region->region) {
    if (tir::is_one(range->extent)) {
      indices.push_back(p->AsExprDoc(range->min));
    } else {
      indices.push_back(p->AsExprDoc(range));
    }
  }

  return p->AsExprDoc(buffer_region->buffer)->Index(indices);
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferRegion>(PrintBufferRegion);

}  // namespace printer
}  // namespace script
}  // namespace tvm
