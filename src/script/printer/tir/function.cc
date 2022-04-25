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

#include "./utils.h"

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
    n->sym = std::move(sym);
    data_ = std::move(n);
  }
  static constexpr const char* _type_key = "script.PrimFuncFrame";
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrimFuncFrame, TIRFrame, PrimFuncFrameNode);
};

Doc PrintPrimFunc(tir::PrimFunc func, IRDocsifier p) {
  using namespace tvm::tir;
  PrimFuncFrame frame(p->sym);
  WithCtx with_dispatch = p->WithDispatchToken("tir");
  WithCtx with_frame = p->WithFrame(frame);

  std::unordered_set<const VarNode*> var_explicit_def;
  std::unordered_map<const VarNode*, const tir::BufferNode*> var_associated_def;
  std::unordered_map<const VarNode*, BufferPrintInfo> var2info;
  std::vector<BufferPrintInfo> match_buffer_info;
  {
    std::vector<Buffer> buffers;
    for (const auto& kv : func->buffer_map) {
      Var data_var = kv.first;
      Buffer buffer = kv.second;
      ICHECK(buffer->data.same_as(data_var));
      buffers.push_back(buffer);
    }
    auto f_var_defined = [&p](const VarNode* var) -> bool {
      return p->sym->GetObjectDoc(GetRef<Var>(var)).defined();
    };
    std::vector<BufferPrintInfo> buffer_infos =
        GetBufferPrintInfo(buffers, f_var_defined, &var_explicit_def, &var_associated_def);
    int n = buffers.size();
    for (int i = 0; i < n; ++i) {
      const Buffer& buffer = buffers[i];
      const BufferPrintInfo& info = buffer_infos[i];
      if (info.data.defined() || info.strides.defined() || info.elem_offset.defined() ||
          info.scope.defined() || info.align.defined() || info.offset_factor.defined() ||
          info.buffer_type.defined()) {
        match_buffer_info.push_back(info);
      } else {
        var2info[buffer->data.get()] = info;
      }
    }
  }
  Array<AssignDoc> args;
  args.reserve(func->params.size());
  for (const Var& v : func->params) {
    auto it = var2info.find(v.get());
    if (it == var2info.end()) {
      IdDoc lhs = frame->DefByName(v, p->sym->GetUniqueName(v->name_hint));
      ExprDoc type = p->AsDoc<ExprDoc>(GetType(v));
      args.push_back(AssignDoc(lhs, NullOpt, type));
      if (var_explicit_def.count(v.get())) {
        var_explicit_def.erase(v.get());
      }
      if (var_associated_def.count(v.get())) {
        var_associated_def.erase(v.get());
      }
    } else {
      const BufferPrintInfo& info = it->second;
      Buffer buffer = func->buffer_map.at(v);
      IdDoc lhs = frame->DefByName(buffer, p->sym->GetUniqueName(buffer->name));
      ExprDoc type = info.AsCall(TIR(p)->Attr("Buffer"), [&p](const PrimExpr& e) -> ExprDoc {
        // TODO: handle undefined vars, e.g. T.Buffer(('a', 'b'), "float32")
        return p->AsDoc<ExprDoc>(e);
      });
      args.push_back(AssignDoc(lhs, NullOpt, type));
      frame->DefByDoc(v, lhs->Attr("data"));
      ICHECK(!var_explicit_def.count(v.get()));
      if (var_associated_def.count(v.get())) {
        var_associated_def.erase(v.get());
      }
    }
  }
  Array<StmtDoc> body;
  for (const VarNode* var : var_explicit_def) {
    frame->DefByName(GetRef<Var>(var), p->sym->GetUniqueName(var->name_hint));
  }
  for (const BufferPrintInfo& info : match_buffer_info) {
    const Buffer& buffer = info.buffer;
    frame->DefByName(buffer, p->sym->GetUniqueName(buffer->name));
  }
  for (const auto& kv : var_associated_def) {
    const VarNode* var = kv.first;
    const BufferNode* buffer = kv.second;
    if (Optional<ExprDoc> lhs = p->sym->GetObjectDoc(GetRef<Buffer>(buffer))) {
      if (buffer->data.get() == var) {
        frame->DefByDoc(GetRef<Var>(var), lhs.value()->Attr("data"));
      } else if (buffer->elem_offset.get() == var) {
        frame->DefByDoc(GetRef<Var>(var), lhs.value()->Attr("elem_offset"));
      } else {
        ICHECK(false) << "Unexpected association. Buffer: " << GetRef<Buffer>(buffer)
                      << "; Var: " << GetRef<Var>(var);
      }
    } else {
      LOG(FATAL) << "Undefined buffer: " << buffer->name;
    }
  }
  for (const VarNode* var : var_explicit_def) {
    if (Optional<ExprDoc> lhs = p->sym->GetObjectDoc(GetRef<Var>(var))) {
      ExprDoc rhs = TIR(p)->Attr("var")->Call({DType2Literal(var->dtype)});
      body.push_back(AssignDoc(lhs.value(), rhs, NullOpt));
    } else {
      LOG(FATAL) << "Undefined variable: " << var->name_hint;
    }
  }
  for (const BufferPrintInfo& info : match_buffer_info) {
    const Buffer& buffer = info.buffer;
    if (Optional<ExprDoc> lhs = p->sym->GetObjectDoc(buffer)) {
      ExprDoc rhs = info.AsCall(TIR(p)->Attr("match_buffer"), [&p](const PrimExpr& e) -> ExprDoc {
        return p->AsDoc<ExprDoc>(e);
      });
      body.push_back(AssignDoc(lhs.value(), rhs, NullOpt));
    } else {
      LOG(FATAL) << "Undefined buffer: " << buffer->name;
    }
  }
  // TODO: support T.func_attrs
  // TODO: support name
  // TODO: support preflatten buffers
  return FunctionDoc(/*name=*/IdDoc("main"),  //
                     /*args=*/args,
                     /*decorators=*/{TIR(p)->Attr("prim_func")},
                     /*return_type=*/p->AsDoc<ExprDoc>(func->ret_type),
                     /*body=*/body);
}

TVM_REGISTER_NODE_TYPE(PrimFuncFrameNode);
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<tir::PrimFunc>(PrintPrimFunc);

}  // namespace printer
}  // namespace script
}  // namespace tvm
