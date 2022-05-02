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

#include <tvm/ir/op.h>
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

using FTVMScriptOpSugar = runtime::TypedPackedFunc<ExprDoc(tir::Call, IRDocsifier)>;
constexpr const char kFTVMScriptOpSugarKey[] = "FTVMScriptOpSugar";

ExprDoc PrintOpCall(tir::Call call, IRDocsifier p) {
  static auto op_sugar_map = Op::GetAttrMap<FTVMScriptOpSugar>(kFTVMScriptOpSugarKey);

  const auto op = Downcast<Op>(call->op);

  if (op_sugar_map.count(op)) {
    return op_sugar_map[op](call, p);
  } else {
    Array<ExprDoc> args{LiteralDoc::Str(op->name)};
    args = Concat(args, AsExprDocArray(call->args, p));
    return TIR(p)->Attr("call")->Call(args);
  }
}

#define TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT(name)                                                 \
  TVM_REGISTER_OP("tir." #name)                                                               \
      .set_attr<FTVMScriptOpSugar>(kFTVMScriptOpSugarKey, [](tir::Call call, IRDocsifier p) { \
        return TIR(p)->Attr(#name)->Call(AsExprDocArray(call->args, p));                      \
      })

TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("trunc");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("exp");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("exp2");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("exp10");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("erf");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("tanh");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("sigmoid");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("sqrt");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("rsqrt");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("log");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("log2");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("log1p");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("log10");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("tan");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("cos");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("cosh");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("sin");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("sinh");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("asin");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("acos");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("atan");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("acosh");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("asinh");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("atanh");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("clz");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("atan2");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("nextafter");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("hypot");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("copysign");
TVM_SCRIPT_TIR_OP_SUGAR_DEFAULT("ldexp");

}  // namespace printer
}  // namespace script
}  // namespace tvm
