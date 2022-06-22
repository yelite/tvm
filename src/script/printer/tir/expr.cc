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

ExprDoc PrintStringImm(tir::StringImm s, ObjectPath path, IRDocsifier p) {
  return LiteralDoc::Str(MakeTraced(s->value, path));
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::StringImm>(PrintStringImm);

ExprDoc PrintIntImm(IntImm i, ObjectPath path, IRDocsifier p) {
  return LiteralDoc::Int(MakeTraced(i, path));
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<IntImm>(PrintIntImm);

ExprDoc PrintFloatImm(FloatImm f, ObjectPath path, IRDocsifier p) {
  return LiteralDoc::Float(MakeTraced(f, path));
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<FloatImm>(PrintFloatImm);

ExprDoc PrintCast(tir::Cast raw_cast, ObjectPath path, IRDocsifier p) {
  auto cast = MakeTraced(raw_cast, path);
  auto value = cast.GetAttr<PrimExpr>("value");
  auto dtype = cast.GetAttr<DataType>("dtype");
  return TIR(p)->Attr("cast")->Call({p->AsExprDoc(value), DType2Literal(dtype)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Cast>(PrintCast);

template <typename BinOpType>
OperationDocNode::Kind GetBinaryOpKind() { throw; }

template <typename BinOpType>
ExprDoc PrintBinOp(BinOpType raw_expr, ObjectPath path, IRDocsifier p) {
  auto expr = MakeTraced(raw_expr, path);
  auto a = expr.template GetAttr<PrimExpr>("a");
  auto b = expr.template GetAttr<PrimExpr>("b");
  return OperationDoc(GetBinaryOpKind<BinOpType>(), {p->AsExprDoc(a), p->AsExprDoc(b)});
}

template<>
OperationDocNode::Kind GetBinaryOpKind<tir::Add>() { return OperationDocNode::Kind::kAdd; }
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Add>(PrintBinOp<tir::Add>);

ExprDoc PrintSelect(tir::Select raw_expr, ObjectPath path, IRDocsifier p) {
  auto expr = MakeTraced(raw_expr, path);
  auto condition = expr.GetAttr<PrimExpr>("condition");
  auto true_value = expr.GetAttr<PrimExpr>("true_value");
  auto false_value = expr.GetAttr<PrimExpr>("false_value");
  return TIR(p)->Attr("Select")->Call(
      {p->AsExprDoc(condition), p->AsExprDoc(true_value), p->AsExprDoc(false_value)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Select>(PrintSelect);

ExprDoc PrintBufferLoad(tir::BufferLoad raw_expr, ObjectPath path, IRDocsifier p) {
  auto expr = MakeTraced(raw_expr, path);
  auto buffer = expr.GetAttr<tir::Buffer>("buffer");
  auto indices = expr.GetAttr<Array<PrimExpr>>("indices");

  ExprDoc base = p->AsExprDoc(buffer);
  return base->Index(AsDocArray<Doc>(indices, p));
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferLoad>(PrintBufferLoad);

ExprDoc PrintProducerLoad(tir::ProducerLoad e, ObjectPath path, IRDocsifier p) {
  LOG(FATAL) << "Cannot print a tir.ProducerLoad as it is not valid in TIR Primfuncs. You need to "
                "lower this function first.";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::ProducerLoad>(PrintProducerLoad);

ExprDoc PrintLoad(tir::Load e, ObjectPath path, IRDocsifier p) {
  LOG(FATAL) << "Cannot print a tir.Load";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Load>(PrintLoad);

ExprDoc PrintRamp(tir::Ramp raw_expr, ObjectPath path, IRDocsifier p) {
  auto expr = MakeTraced(raw_expr, path);
  auto base = expr.GetAttr<PrimExpr>("base");
  auto stride = expr.GetAttr<PrimExpr>("stride");
  auto lanes = expr.GetAttr<int>("lanes");
  return TIR(p)->Attr("ramp")->Call(
      {p->AsExprDoc(base), p->AsExprDoc(stride), LiteralDoc::Int(lanes)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Ramp>(PrintRamp);

ExprDoc PrintBroadcast(tir::Broadcast raw_expr, ObjectPath path, IRDocsifier p) {
  auto expr = MakeTraced(raw_expr, path);
  auto value = expr.GetAttr<PrimExpr>("value");
  auto lanes = expr.GetAttr<int>("lanes");
  return TIR(p)->Attr("broadcast")->Call({p->AsExprDoc(value), LiteralDoc::Int(lanes)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Broadcast>(PrintBroadcast);

ExprDoc PrintLet(tir::Let raw_expr, ObjectPath path, IRDocsifier p) {
  auto expr = MakeTraced(raw_expr, path);
  auto var = expr.GetAttr<tir::Var>("var");
  auto value = expr.GetAttr<PrimExpr>("value");
  auto body = expr.GetAttr<PrimExpr>("body");
  return TIR(p)->Attr("let")->Call({p->AsExprDoc(var), p->AsExprDoc(value), p->AsExprDoc(body)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Let>(PrintLet);

ExprDoc PrintCall(tir::Call raw_call, ObjectPath path, IRDocsifier p) {
  auto call = MakeTraced(raw_call, path);
  auto op = call.GetAttr<RelayExpr>("op");

  if (op.Get()->IsInstance<OpNode>()) {
    return PrintOpCall(call, p);
  } else {
    auto op_gvar = op.Downcast<GlobalVar>();
    auto name_hint = op_gvar.GetAttr<String>("name_hint");
    auto args = call.GetAttr<Array<PrimExpr>>("args");

    IdDoc name_doc(name_hint.Get());
    name_doc->paths.push_back(name_hint.GetPath());

    return name_doc->Call(AsExprDocArray(args, p));
  }
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Call>(PrintCall);

ExprDoc PrintShuffle(tir::Shuffle raw_expr, ObjectPath path, IRDocsifier p) {
  auto expr = MakeTraced(raw_expr, path);
  auto vectors = expr.GetAttr<Array<PrimExpr>>("vectors");
  auto indices = expr.GetAttr<Array<PrimExpr>>("indices");
  return TIR(p)->Attr("shuffle")->Call({AsListDoc(vectors, p), AsListDoc(indices, p)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Shuffle>(PrintShuffle);

ExprDoc PrintCommReducer(tir::CommReducer raw_expr, ObjectPath path, IRDocsifier p) {
  auto expr = MakeTraced(raw_expr, path);
  TIRGeneralFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);

  auto lhs = expr.GetAttr<Array<tir::Var>>("lhs");
  auto rhs = expr.GetAttr<Array<tir::Var>>("rhs");

  Array<IdDoc> reducer_args;
  for (TracedObject<tir::Var> v_lhs : lhs) {
    IdDoc var_doc = DefineVariable(v_lhs, frame);
    reducer_args.push_back(var_doc);
  }
  for (TracedObject<tir::Var> v_rhs : rhs) {
    IdDoc var_doc = DefineVariable(v_rhs, frame);
    reducer_args.push_back(var_doc);
  }

  auto result = expr.GetAttr<Array<PrimExpr>>("result");

  ExprDoc reducer_body = rhs.size() == 1 ? p->AsExprDoc(result[0]) : AsTupleDoc(result, p);

  LambdaDoc reducer{reducer_args, reducer_body};

  auto identity_element = expr.GetAttr<Array<PrimExpr>>("identity_element");
  ListDoc identity_elements_doc = AsListDoc(identity_element, p);

  return TIR(p)->Attr("comm_reducer")->Call({reducer, identity_elements_doc});
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::CommReducer>(PrintCommReducer);

ExprDoc PrintReduce(tir::Reduce raw_expr, ObjectPath path, IRDocsifier p) {
  auto expr = MakeTraced(raw_expr, path);
  auto combiner = expr.GetAttr<tir::CommReducer>("combiner");
  auto source = expr.GetAttr<Array<PrimExpr>>("source");
  auto axis = expr.GetAttr<Array<tir::IterVar>>("axis");
  auto value_index = expr.GetAttr<int>("value_index");
  return TIR(p)->Attr("reduce")->Call({p->AsExprDoc(combiner), AsListDoc(source, p),
                                       AsListDoc(axis, p), LiteralDoc::Int(value_index)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Reduce>(PrintReduce);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Range>([](Range raw_expr, ObjectPath path, IRDocsifier p) {
      auto expr = MakeTraced(raw_expr, path);
      auto min = expr.GetAttr<PrimExpr>("min");
      auto extent = expr.GetAttr<PrimExpr>("extent");
      auto max = MakeTraced(min.Get() + extent.Get(), extent.GetPath());
      return SliceDoc(p->AsExprDoc(min), p->AsExprDoc(max));
    });

ExprDoc PrintAny(tir::Any e, ObjectPath path, IRDocsifier p) {
  LOG(FATAL) << "Cannot print any shape";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Any>(PrintAny);

ExprDoc PrintBufferRegion(tir::BufferRegion raw_buffer_region, ObjectPath path, IRDocsifier p) {
  auto buffer_region = MakeTraced(raw_buffer_region, path);
  auto region = buffer_region.GetAttr<Array<Range>>("region");

  Array<Doc> indices;

  for (TracedObject<Range> range : region) {
    auto extent = range.GetAttr<PrimExpr>("extent");
    if (tir::is_one(extent.Get())) {
      auto index = p->AsExprDoc(range.GetAttr<PrimExpr>("min"));
      index->paths.push_back(extent.GetPath());
      indices.push_back(std::move(index));
    } else {
      indices.push_back(p->AsExprDoc(range));
    }
  }

  auto buffer = buffer_region.GetAttr<tir::Buffer>("buffer");
  return p->AsExprDoc(buffer)->Index(indices);
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferRegion>(PrintBufferRegion);

}  // namespace printer
}  // namespace script
}  // namespace tvm
