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

#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include "../util.h"
#include "./utils.h"
#include "tvm/node/functor.h"
#include "tvm/runtime/data_type.h"

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
  return TIR(p)->Attr("cast")->Call(
      {p->AsExprDoc(e->value), LiteralDoc::Str(runtime::DLDataType2String(e->dtype))});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Cast>(PrintCast);

template <typename BinOpType>
OperationDocNode::Kind GetBinaryOpKind() {}

template <typename BinOpType>
ExprDoc PrintBinOp(BinOpType e, IRDocsifier p) {
  return OperationDoc(GetBinaryOpKind<BinOpType>(), {p->AsExprDoc(e->a), p->AsExprDoc(e->b)});
}

ExprDoc PrintSelect(tir::Select e, IRDocsifier p) {
  return TIR(p)->Attr("Select")->Call(
      {p->AsExprDoc(e->condition), p->AsExprDoc(e->true_value), p->AsExprDoc(e->false_value)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Select>(PrintSelect);

ExprDoc PrintBufferLoad(tir::BufferLoad e, IRDocsifier p) {
  ExprDoc base = p->AsExprDoc(e->buffer);
  if (e->indices.size() == 0) {
    return base->Index({TupleDoc()});
  } else {
    return base->Index(AsDocArray<Doc>(e->indices, p));
  }
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferLoad>(PrintBufferLoad);

ExprDoc PrintProducerLoad(tir::ProducerLoad e, IRDocsifier p) {
  LOG(FATAL) << "Cannot print a tir.ProducerLoad as it is not valid in TIR Primfuncs. You need to "
                "lower this function first.";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::ProducerLoad>(PrintProducerLoad);

ExprDoc PrintLoad(tir::Load e, IRDocsifier p) {
  if (e->dtype == DataType::Float(32) && tir::is_one(e->predicate) &&
      e->buffer_var->dtype == DataType::Float(32)) {
    return p->AsExprDoc(e->buffer_var)->Index({p->AsExprDoc(e->index)});
  } else {
    Array<ExprDoc> args{LiteralDoc::Str(DLDataType2String(e->dtype)), p->AsExprDoc(e->buffer_var),
                        p->AsExprDoc(e->index)};
    if (!tir::is_one(e->predicate) || e->dtype.lanes() != 1) {
      args.push_back(p->AsExprDoc(e->predicate));
    }
    return TIR(p)->Attr("load")->Call(args);
  }
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
  ExprDoc callee = IdDoc("");
  if (const auto* op = e->op.as<OpNode>()) {
    std::string name = op->name;
    if (name.find("tir.") == 0) {
      callee = TIR(p)->Attr(name.substr(4));
    } else {
      callee = IdDoc(name);
    }
  } else {
    const auto* op_gvar = e->op.as<GlobalVarNode>();
    ICHECK(op_gvar != nullptr);
    callee = IdDoc(op_gvar->name_hint);
  }
  return callee->Call(AsExprDocArray(e->args, p), {"dtype"},
                      {LiteralDoc::Str(DLDataType2String(e->dtype))});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Call>(PrintCall);

ExprDoc PrintShuffle(tir::Shuffle e, IRDocsifier p) {
  return TIR(p)->Attr("shuffle")->Call({p->AsExprDoc(e->vectors), p->AsExprDoc(e->indices)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Shuffle>(PrintShuffle);

ExprDoc PrintCommReducer(tir::CommReducer e, IRDocsifier p) {
  TIRFrame frame(p->sym);
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

ExprDoc PrintAny(tir::Any e, IRDocsifier p) {
  LOG(FATAL) << "Cannot print any shape";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Any>(PrintAny);

}  // namespace printer
}  // namespace script
}  // namespace tvm
