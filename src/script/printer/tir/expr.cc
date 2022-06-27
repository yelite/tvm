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

ExprDoc PrintStringImm(TracedObject<tir::StringImm> s, IRDocsifier p) {
  auto value = s.GetAttr(&tir::StringImmNode::value);
  return LiteralDoc::Str(value);
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::StringImm>(PrintStringImm);

ExprDoc PrintIntImm(TracedObject<IntImm> i, IRDocsifier p) { return LiteralDoc::Int(i); }
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<IntImm>(PrintIntImm);

ExprDoc PrintFloatImm(TracedObject<FloatImm> f, IRDocsifier p) { return LiteralDoc::Float(f); }
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<FloatImm>(PrintFloatImm);

ExprDoc PrintCast(TracedObject<tir::Cast> cast, IRDocsifier p) {
  auto value = cast.GetAttr(&tir::CastNode::value);
  auto dtype = cast.GetAttr(&tir::CastNode::dtype);
  return TIR(p)->Attr("cast")->Call({p->AsExprDoc(value), DType2Literal(dtype)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Cast>(PrintCast);

template <typename BinOpType>
OperationDocNode::Kind GetBinaryOpKind() { throw; }

template <typename BinOpType>
ExprDoc PrintBinOp(TracedObject<BinOpType> expr, IRDocsifier p) {
  using NodeType = typename BinOpType::ContainerType;
  auto a = expr.GetAttr(&NodeType::a);
  auto b = expr.GetAttr(&NodeType::b);
  return OperationDoc(GetBinaryOpKind<BinOpType>(), {p->AsExprDoc(a), p->AsExprDoc(b)});
}

template<>
OperationDocNode::Kind GetBinaryOpKind<tir::Add>() { return OperationDocNode::Kind::kAdd; }
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Add>(PrintBinOp<tir::Add>);

ExprDoc PrintSelect(TracedObject<tir::Select> expr, IRDocsifier p) {
  auto condition = expr.GetAttr(&tir::SelectNode::condition);
  auto true_value = expr.GetAttr(&tir::SelectNode::true_value);
  auto false_value = expr.GetAttr(&tir::SelectNode::false_value);
  return TIR(p)->Attr("Select")->Call(
      {p->AsExprDoc(condition), p->AsExprDoc(true_value), p->AsExprDoc(false_value)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Select>(PrintSelect);

ExprDoc PrintBufferLoad(TracedObject<tir::BufferLoad> expr, IRDocsifier p) {
  auto buffer = expr.GetAttr(&tir::BufferLoadNode::buffer);
  auto indices = expr.GetAttr(&tir::BufferLoadNode::indices);

  ExprDoc base = p->AsExprDoc(buffer);
  return base->Index(AsDocArray<Doc>(indices, p));
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferLoad>(PrintBufferLoad);

ExprDoc PrintProducerLoad(TracedObject<tir::ProducerLoad> e, IRDocsifier p) {
  LOG(FATAL) << "Cannot print a tir.ProducerLoad as it is not valid in TIR Primfuncs. You need to "
                "lower this function first.";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::ProducerLoad>(PrintProducerLoad);

ExprDoc PrintLoad(TracedObject<tir::Load> e, IRDocsifier p) {
  LOG(FATAL) << "Cannot print a tir.Load";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Load>(PrintLoad);

ExprDoc PrintRamp(TracedObject<tir::Ramp> expr, IRDocsifier p) {
  auto base = expr.GetAttr(&tir::RampNode::base);
  auto stride = expr.GetAttr(&tir::RampNode::stride);
  auto lanes = expr.GetAttr(&tir::RampNode::lanes);
  return TIR(p)->Attr("ramp")->Call(
      {p->AsExprDoc(base), p->AsExprDoc(stride), LiteralDoc::Int(lanes)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Ramp>(PrintRamp);

ExprDoc PrintBroadcast(TracedObject<tir::Broadcast> expr, IRDocsifier p) {
  auto value = expr.GetAttr(&tir::BroadcastNode::value);
  auto lanes = expr.GetAttr(&tir::BroadcastNode::lanes);
  return TIR(p)->Attr("broadcast")->Call({p->AsExprDoc(value), LiteralDoc::Int(lanes)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Broadcast>(PrintBroadcast);

ExprDoc PrintLet(TracedObject<tir::Let> expr, IRDocsifier p) {
  auto var = expr.GetAttr(&tir::LetNode::var);
  auto value = expr.GetAttr(&tir::LetNode::value);
  auto body = expr.GetAttr(&tir::LetNode::body);
  return TIR(p)->Attr("let")->Call({p->AsExprDoc(var), p->AsExprDoc(value), p->AsExprDoc(body)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Let>(PrintLet);

ExprDoc PrintCall(TracedObject<tir::Call> call, IRDocsifier p) {
  auto op = call.GetAttr(&tir::CallNode::op);

  if (op.Get()->IsInstance<OpNode>()) {
    return PrintOpCall(call, p);
  } else {
    auto op_gvar = op.Downcast<GlobalVar>();
    auto name_hint = op_gvar.GetAttr(&GlobalVarNode::name_hint);
    auto args = call.GetAttr(&tir::CallNode::args);

    IdDoc name_doc(name_hint.Get());
    name_doc->paths.push_back(name_hint.GetPath());

    return name_doc->Call(AsExprDocArray(args, p));
  }
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Call>(PrintCall);

ExprDoc PrintShuffle(TracedObject<tir::Shuffle> expr, IRDocsifier p) {
  auto vectors = expr.GetAttr(&tir::ShuffleNode::vectors);
  auto indices = expr.GetAttr(&tir::ShuffleNode::indices);
  return TIR(p)->Attr("shuffle")->Call({AsListDoc(vectors, p), AsListDoc(indices, p)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Shuffle>(PrintShuffle);

ExprDoc PrintCommReducer(TracedObject<tir::CommReducer> expr, IRDocsifier p) {
  TIRGeneralFrame frame(p->sym);
  WithCtx with_frame = p->WithFrame(frame);

  auto lhs = expr.GetAttr(&tir::CommReducerNode::lhs);
  auto rhs = expr.GetAttr(&tir::CommReducerNode::rhs);

  Array<IdDoc> reducer_args;
  for (TracedObject<tir::Var> v_lhs : lhs) {
    IdDoc var_doc = DefineVariable(v_lhs, frame);
    reducer_args.push_back(var_doc);
  }
  for (TracedObject<tir::Var> v_rhs : rhs) {
    IdDoc var_doc = DefineVariable(v_rhs, frame);
    reducer_args.push_back(var_doc);
  }

  auto result = expr.GetAttr(&tir::CommReducerNode::result);

  ExprDoc reducer_body = rhs.size() == 1 ? p->AsExprDoc(result[0]) : AsTupleDoc(result, p);

  LambdaDoc reducer{reducer_args, reducer_body};

  auto identity_element = expr.GetAttr(&tir::CommReducerNode::identity_element);
  ListDoc identity_elements_doc = AsListDoc(identity_element, p);

  return TIR(p)->Attr("comm_reducer")->Call({reducer, identity_elements_doc});
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::CommReducer>(PrintCommReducer);

ExprDoc PrintReduce(TracedObject<tir::Reduce> expr, IRDocsifier p) {
  auto combiner = expr.GetAttr(&tir::ReduceNode::combiner);
  auto source = expr.GetAttr(&tir::ReduceNode::source);
  auto axis = expr.GetAttr(&tir::ReduceNode::axis);
  auto value_index = expr.GetAttr(&tir::ReduceNode::value_index);
  return TIR(p)->Attr("reduce")->Call({p->AsExprDoc(combiner), AsListDoc(source, p),
                                       AsListDoc(axis, p), LiteralDoc::Int(value_index)});
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Reduce>(PrintReduce);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Range>([](TracedObject<Range> expr, IRDocsifier p) {
      auto min = expr.GetAttr(&RangeNode::min);
      auto extent = expr.GetAttr(&RangeNode::extent);
      auto max = MakeTraced(min.Get() + extent.Get(), extent.GetPath());
      return SliceDoc(p->AsExprDoc(min), p->AsExprDoc(max));
    });

ExprDoc PrintAny(TracedObject<tir::Any> e, IRDocsifier p) {
  LOG(FATAL) << "Cannot print any shape";
  throw;
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::Any>(PrintAny);

ExprDoc PrintBufferRegion(TracedObject<tir::BufferRegion> buffer_region, IRDocsifier p) {
  auto region = buffer_region.GetAttr(&tir::BufferRegionNode::region);

  Array<Doc> indices;

  for (TracedObject<Range> range : region) {
    auto extent = range.GetAttr(&RangeNode::extent);
    if (tir::is_one(extent.Get())) {
      auto index = p->AsExprDoc(range.GetAttr(&RangeNode::min));
      index->paths.push_back(extent.GetPath());
      indices.push_back(std::move(index));
    } else {
      indices.push_back(p->AsExprDoc(range));
    }
  }

  auto buffer = buffer_region.GetAttr(&tir::BufferRegionNode::buffer);
  return p->AsExprDoc(buffer)->Index(indices);
}
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::BufferRegion>(PrintBufferRegion);

}  // namespace printer
}  // namespace script
}  // namespace tvm
