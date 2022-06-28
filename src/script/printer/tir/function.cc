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

#include "./buffer.h"

namespace tvm {
namespace script {
namespace printer {

class PrimFuncFrameNode : public TIRFrameNode {
 public:
  static constexpr const char* _type_key = "script.PrimFuncFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimFuncFrameNode, TIRFrameNode);
};

class PrimFuncFrame : public TIRFrame {
 public:
  explicit PrimFuncFrame(SymbolTable sym) {
    ObjectPtr<PrimFuncFrameNode> n = make_object<PrimFuncFrameNode>();
    n->sym = sym.get();
    data_ = std::move(n);
  }
  static constexpr const char* _type_key = "script.PrimFuncFrame";
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrimFuncFrame, TIRFrame, PrimFuncFrameNode);
};

static TracedObject<tir::Stmt> GetFunctionBody(const TracedObject<tir::PrimFunc>& func) {
  auto func_body = func.GetAttr(&tir::PrimFuncNode::body);
  if (auto block_realize = func_body.TryDowncast<tir::BlockRealize>()) {
    if (block_realize.value().Get()->iter_values.empty() &&
        block_realize.value().Get()->block->annotations.empty()) {
      return block_realize.value().GetAttr(&tir::BlockRealizeNode::block);
    }
  }
  return func_body;
}

Doc PrintPrimFunc(TracedObject<tir::PrimFunc> func, IRDocsifier p) {
  using namespace tvm::tir;

  PrimFuncFrame frame(p->sym);
  WithCtx with_dispatch = p->WithDispatchToken("tir");
  WithCtx with_frame = p->WithFrame(frame);

  std::unordered_map<const VarNode*, ObjectPath> var_explicit_def;
  AssociatedVariables associated_vars;
  std::unordered_map<const VarNode*, BufferPrintInfo> var2info;
  std::vector<BufferPrintInfo> match_buffer_info;
  {
    std::vector<Var> buffer_vars;
    std::vector<TracedObject<Buffer>> buffers;
    for (auto kv : func.GetAttr(&tir::PrimFuncNode::buffer_map)) {
      Var buffer_var = kv.first;
      TracedObject<Buffer> buffer = kv.second;
      buffers.push_back(buffer);
      buffer_vars.push_back(buffer_var);
    }
    auto f_var_defined = [&p](const VarNode* var) -> bool {
      return p->sym->IsObjectDefined(GetRef<Var>(var));
    };
    std::vector<BufferPrintInfo> buffer_infos =
        GetBufferPrintInfo(buffers, f_var_defined, &var_explicit_def, associated_vars);
    int n = buffers.size();
    for (int i = 0; i < n; ++i) {
      const BufferPrintInfo& info = buffer_infos[i];
      if (info.data.defined() || info.strides.defined() || info.elem_offset.defined() ||
          info.scope.defined() || info.align.defined() || info.offset_factor.defined() ||
          info.buffer_type.defined()) {
        match_buffer_info.push_back(info);
      } else {
        var2info.insert({buffer_vars[i].get(), info});
      }
    }
  }
  auto params = func.GetAttr(&tir::PrimFuncNode::params);
  Array<AssignDoc> args;
  args.reserve(params.size());
  for (TracedObject<Var> v : params) {
    auto it = var2info.find(v.Get().get());
    if (it == var2info.end()) {
      IdDoc lhs = DefineVariable(v, frame);
      ExprDoc type_annotation = GetTypeAnnotationDocForVar(v, p);
      args.push_back(AssignDoc(lhs, NullOpt, type_annotation));
      associated_vars.Disassociate(v.Get().get());
      var_explicit_def.erase(v.Get().get());
    } else {
      const BufferPrintInfo& info = it->second;
      IdDoc lhs = DefineBuffer(info.buffer, frame);
      ExprDoc type =
          info.AsCall(TIR(p)->Attr("Buffer"), [&p](const TracedObject<PrimExpr>& e) -> ExprDoc {
            // TODO: handle undefined vars, e.g. T.Buffer(('a', 'b'), "float32")
            return p->AsDoc<ExprDoc>(e);
          });
      args.push_back(AssignDoc(lhs, NullOpt, type));
      DefineBufferDataVariable(info.buffer.Get(), frame);
      associated_vars.Disassociate(info.buffer.Get()->data.get());
      ICHECK(!var_explicit_def.count(v.Get().get()));
    }
  }
  Array<StmtDoc> body;
  for (const auto& var_and_path : var_explicit_def) {
    auto var_ref = GetRef<Var>(var_and_path.first);
    auto var = MakeTraced(var_ref, var_and_path.second);
    IdDoc id = DefineVariable(var, frame);
    auto dtype = var.GetAttr(&VarNode::dtype);
    ExprDoc rhs = TIR(p)->Attr("var")->Call({DType2Literal(dtype)});
    body.push_back(AssignDoc(id, rhs, NullOpt));
  }

  std::vector<IdDoc> match_buffer_ids;
  match_buffer_ids.reserve(match_buffer_info.size());
  for (const BufferPrintInfo& info : match_buffer_info) {
    match_buffer_ids.push_back(DefineBuffer(info.buffer, frame));
  }
  associated_vars.DefineVariables(frame);

  for (size_t i = 0; i < match_buffer_info.size(); ++i) {
    ExprDoc rhs = match_buffer_info[i].AsCall(
        TIR(p)->Attr("match_buffer"),
        [&p](const TracedObject<PrimExpr>& e) -> ExprDoc { return p->AsDoc<ExprDoc>(e); });
    body.push_back(AssignDoc(match_buffer_ids.at(i), rhs, NullOpt));
  }
  // TODO: support T.func_attrs
  // TODO: support name
  // TODO: support preflatten buffers

  auto func_body = GetFunctionBody(func);
  body = runtime::Concat(body, AsStmtDocArray(func_body, p));

  auto ret_type = func.GetAttr(&PrimFuncNode::ret_type);

  return FunctionDoc(/*name=*/IdDoc("main"),  //
                     /*args=*/args,
                     /*decorators=*/{TIR(p)->Attr("prim_func")},
                     /*return_type=*/p->AsDoc<ExprDoc>(ret_type),
                     /*body=*/body);
}

TVM_REGISTER_NODE_TYPE(PrimFuncFrameNode);
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::PrimFunc>(PrintPrimFunc);

}  // namespace printer
}  // namespace script
}  // namespace tvm
