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

#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/traced_object_functor.h>
#include <tvm/tir/function.h>

#include "../utils.h"
#include "./buffer.h"
#include "./tir.h"
#include "./var.h"

namespace tvm {
namespace script {
namespace printer {

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

static IdDoc GetFunctionName(TracedObject<tir::PrimFunc> func, const IRDocsifier& p) {
  IdDoc func_name("main");

  // Get name from function attributes
  auto attrs = func.GetAttr(&tir::PrimFuncNode::attrs);
  if (attrs.defined()) {
    auto attrs_dict = attrs.GetAttr(&DictAttrsNode::dict);
    const auto& it = attrs_dict.find("global_symbol");
    if (it != attrs_dict.end()) {
      func_name = IdDoc(Downcast<String>((*it).second.Get()));
    }
  }

  // Get name from IRModule
  if (auto ir_module_frame = p->GetFrame<IRModuleFrame>()) {
    auto name_it = ir_module_frame.get()->function_names.find(func.Get());
    if (name_it != ir_module_frame.get()->function_names.end()) {
      func_name = IdDoc((*name_it).second->name_hint);
    }
  }

  return func_name;
}

Doc PrintPrimFunc(TracedObject<tir::PrimFunc> func, IRDocsifier p) {
  TIRPrimFuncFrame frame;
  WithCtx with_dispatch = p->WithDispatchToken("tir");
  WithCtx with_frame = p->WithFrame(frame);

  auto buffer_map = func.GetAttr(&tir::PrimFuncNode::buffer_map);

  std::unordered_map<const tir::VarNode*, ObjectPath> var_explicit_def;
  BufferAssociatedVariables associated_vars;
  std::unordered_map<const tir::VarNode*, BufferPrintInfo> var2info;
  std::vector<tir::Var> match_buffer_vars;
  std::vector<BufferPrintInfo> match_buffer_info;
  {
    std::vector<tir::Var> buffer_vars;
    std::vector<TracedObject<tir::Buffer>> buffers;
    for (auto kv : buffer_map) {
      tir::Var buffer_var = kv.first;
      TracedObject<tir::Buffer> buffer = kv.second;
      buffers.push_back(buffer);
      buffer_vars.push_back(buffer_var);
    }
    auto f_var_defined = [&p](const tir::VarNode* var) -> bool {
      return p->vars->IsVarDefined(GetRef<tir::Var>(var));
    };
    std::vector<BufferPrintInfo> buffer_infos =
        GetBufferPrintInfo(buffers, f_var_defined, &var_explicit_def, &associated_vars);
    int n = buffers.size();
    for (int i = 0; i < n; ++i) {
      const BufferPrintInfo& info = buffer_infos[i];
      if (info.data.defined() || info.strides.defined() || info.elem_offset.defined() ||
          info.scope.defined() || info.align.defined() || info.offset_factor.defined() ||
          info.buffer_type.defined()) {
        match_buffer_vars.push_back(buffer_vars[i]);
        match_buffer_info.push_back(info);
      } else {
        var2info.insert({buffer_vars[i].get(), info});
      }
    }
  }
  auto params = func.GetAttr(&tir::PrimFuncNode::params);
  Array<AssignDoc> args;
  args.reserve(params.size());
  for (TracedObject<tir::Var> v : params) {
    auto it = var2info.find(v.Get().get());
    if (it == var2info.end()) {
      DeclareVar(v, frame, p, [&args](AssignDoc arg) { args.push_back(arg); });
      var_explicit_def.erase(v.Get().get());
      associated_vars.Disassociate(v.Get().get());
    } else {
      const BufferPrintInfo& info = it->second;
      IdDoc lhs =
          p->vars->Define(info.buffer.Get(), info.buffer.GetAttr(&tir::BufferNode::name), frame);
      ExprDoc type = info.AsCall(
          TIR(p)->Attr("Buffer"),
          [&p](const TracedObject<PrimExpr>& e) -> ExprDoc { return p->AsDoc<ExprDoc>(e); });
      args.push_back(AssignDoc(lhs, NullOpt, type));
      p->vars->DefineByDoc(
          info.buffer.Get()->data, [lhs]() { return lhs->Attr("data"); }, frame);
      associated_vars.Disassociate(info.buffer.Get()->data.get());
      ICHECK(!var_explicit_def.count(v.Get().get()));
    }
  }
  Array<StmtDoc> body;

  auto attrs = func.GetAttr(&tir::PrimFuncNode::attrs);
  if (attrs.defined()) {
    auto attrs_dict = attrs.GetAttr(&DictAttrsNode::dict);
    if (!attrs_dict.empty()) {
      Array<ExprDoc> keys;
      Array<ExprDoc> values;
      for (const auto& entry : attrs_dict) {
        keys.push_back(LiteralDoc::Str(entry.first));
        values.push_back(p->AsExprDoc(entry.second));
      }
      body.push_back(ExprStmtDoc(TIR(p)->Attr("func_attr")->Call({DictDoc(keys, values)})));
    }
  }

  for (const auto& var_and_path : var_explicit_def) {
    auto var_ref = GetRef<tir::Var>(var_and_path.first);
    auto var = MakeTraced(var_ref, var_and_path.second);
    IdDoc id = DefineVar(var, frame, p, [&body](AssignDoc def) { body.push_back(def); });
  }

  std::vector<IdDoc> match_buffer_ids;
  match_buffer_ids.reserve(match_buffer_info.size());
  for (const BufferPrintInfo& info : match_buffer_info) {
    match_buffer_ids.push_back(
        p->vars->Define(info.buffer.Get(), info.buffer.GetAttr(&tir::BufferNode::name), frame));
  }
  associated_vars.Define(p->vars.get(), frame);

  for (size_t i = 0; i < match_buffer_info.size(); ++i) {
    ExprDoc rhs = match_buffer_info[i].AsCall(
        TIR(p)->Attr("match_buffer"),
        {p->AsExprDoc(MakeTraced(match_buffer_vars[i]))},
        [&p](const TracedObject<PrimExpr>& e) -> ExprDoc {
      return p->AsDoc<ExprDoc>(e); });
    body.push_back(AssignDoc(match_buffer_ids.at(i), rhs, NullOpt));
  }

  // preflatten buffers
  auto preflattened_buffers = func.GetAttr(&tir::PrimFuncNode::preflattened_buffer_map);
  for (const auto& param : params) {
    auto pf_buf_it = preflattened_buffers.find(param.Get());
    if (pf_buf_it != preflattened_buffers.end()) {
      const auto& preflattened = (*pf_buf_it).second;
      const auto& postflattened = buffer_map.at(param.Get());
      DefineBuffers({preflattened}, {p->AsExprDoc(postflattened)}, frame, p,
                    TIR(p)->Attr("preflattened_buffer"),
                    [&body](IdDoc id, ExprDoc def) { body.push_back(ExprStmtDoc(def)); });
    }
  }

  auto func_body = GetFunctionBody(func);
  body = runtime::Concat(body, AsStmtDocArray(func_body, p));

  auto ret_type = func.GetAttr(&tir::PrimFuncNode::ret_type);

  return FunctionDoc(/*name=*/GetFunctionName(func, p),  //
                     /*args=*/args,
                     /*decorators=*/{TIR(p)->Attr("prim_func")},
                     /*return_type=*/p->AsDoc<ExprDoc>(ret_type),
                     /*body=*/body);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::PrimFunc>(PrintPrimFunc);

}  // namespace printer
}  // namespace script
}  // namespace tvm
