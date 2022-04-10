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

#include "doc.h"
#include "tvmscript_unified_printer.h"

namespace tvm {
namespace script {
namespace printer {

namespace {
using namespace tir;
}

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const PrimFunc& func, TVMScriptUnifiedPrinter& p) {
  FunctionDoc func_doc;
  PrinterFunctionContext context = p.context_manager->EnterContext<PrinterFunctionContext>();

  String func_name = "func";
  const auto& it = func->attrs->dict.find("global_symbol");
  if (it != func->attrs->dict.end()) {
    func_name = Downcast<String>((*it).second);
  }
  func_doc->name = func_name;

  Array<FunctionArgDoc>& params = func_doc->args;
  for (const Var& param : func->params) {
    auto it = func->buffer_map.find(param);
    if (it != func->buffer_map.end()) {
      // Buffer
      const Buffer& buf = (*it).second;
      p.context_manager->AddVar(buf);
      params.push_back({p.ToDoc<IdentifierDoc>(buf), p.GetBufferTypeDoc(buf)});
    } else {
      // Var
      p.context_manager->AddVar(param);
      params.push_back({p.ToDoc<IdentifierDoc>(param), p.GetVarTypeDoc(param)});
    }
  }
  func_doc->return_type = p.ToDoc<TypeDoc>(func->ret_type);

  SeqStmtDoc body;

  ObjectRef body_to_print = func->body;
  if (func->body->IsInstance<BlockRealizeNode>() &&
      func->body.as<BlockRealizeNode>()->iter_values.empty()) {
    const BlockNode* block = func->body.as<BlockRealizeNode>()->block.get();
    if (block->annotations.empty()) {
      // Skip print root block
      body_to_print = GetRef<ObjectRef>(block);
    }
  }
  body.Add(p.ToDoc<StmtDoc>(body_to_print));

  func_doc->body = std::move(body);

  p.context_manager->ExitContext(std::move(context));

  return func_doc;
});

SeqStmtDoc GetBlockVarsDeclarations(const BlockRealize block_realize, TVMScriptUnifiedPrinter& p) {
  SeqStmtDoc doc;
  const Block& block = block_realize->block;
  ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());

  // TODO: handle remap

  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& value = block_realize->iter_values[i];
    p.context_manager->AddVar(iter_var->var);
    AssignDoc assign_stmt;
    assign_stmt->target = p.ToDoc<IdentifierDoc>(iter_var->var);
    std::string axis_type;
    switch (iter_var->iter_type) {
      case kDataPar:
        axis_type = "spatial";
        break;
      case kCommReduce:
        axis_type = "reduce";
        break;
      case kOrdered:
        axis_type = "scan";
        break;
      case kOpaque:
        axis_type = "opaque";
        break;
      default:
        LOG(FATAL) << "Unknown block var iter type: " << iter_var->iter_type;
        break;
    }
    const Range& dom = iter_var->dom;
    ExprDoc dom_arg;
    if (is_zero(dom->min)) {
      dom_arg = p.ToExprDoc(dom->extent);
    } else {
      dom_arg = TupleDoc{p.ToExprDoc(dom->min), p.ToExprDoc(dom->min + dom->extent)};
    }
    assign_stmt->value = ConstDoc::TIRBuilder()
                             .AccessAttr("axis")
                             .AccessAttr(std::move(axis_type))
                             .CallWith(std::move(dom_arg), p.ToExprDoc(value));

    doc.Add(assign_stmt);
  }
  return doc;
}

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const BlockRealize& block_realize, TVMScriptUnifiedPrinter& p) {
  const Block& block = block_realize->block;
  ScopeDoc scope_doc;
  auto context = p.context_manager->EnterContext<PrinterBlockContext>();

  // TODO: optional info
  // print block name and block vars
  scope_doc->scope =
      ConstDoc::TIRBuilder().AccessAttr("block").CallWith(LiteralValueDoc(block->name_hint));

  SeqStmtDoc body;

  body.Extend(GetBlockVarsDeclarations(block_realize, p));
  body.Add(p.ToDoc<StmtDoc>(block));

  scope_doc->body = std::move(body);

  p.context_manager->ExitContext(std::move(context));
  return scope_doc;
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const Block& block, TVMScriptUnifiedPrinter& p) {
  // TODO: T.alloc_buffer and match_buffer and init
  return p.ToDoc<StmtDoc>(block->body);
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const For& for_ref, TVMScriptUnifiedPrinter& p) {
  ForDoc doc;
  auto context = p.context_manager->EnterContext<PrinterLoopContext>();
  p.context_manager->AddVar(for_ref->loop_var);

  doc->target = p.ToDoc<IdentifierDoc>(for_ref->loop_var);
  auto for_kind = ConstDoc::TIRBuilder().AccessAttr(ForKind2String(for_ref->kind));
  if (is_zero(for_ref->min)) {
    doc->iter = for_kind.CallWith(p.ToExprDoc(for_ref->extent));
  } else {
    doc->iter =
        for_kind.CallWith(p.ToExprDoc(for_ref->min), p.ToExprDoc(for_ref->min + for_ref->extent));
  }
  // TODO: annotation, thread binding
  doc->body = p.ToDoc<StmtDoc>(for_ref->body);

  p.context_manager->ExitContext(std::move(context));
  return doc;
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const PrimType& type, TVMScriptUnifiedPrinter& p) {
  return ExprTypeDoc::TIRPrimitive(runtime::DLDataType2String(type->dtype));
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const TupleType& type, TVMScriptUnifiedPrinter& p) {
  if (type->fields.empty()) {
    return TypeDoc::NoneType();
  } else {
    std::vector<TypeDoc> fields;
    for (const Type& field : type->fields) {
      fields.push_back(p.ToDoc<TypeDoc>(field));
    }
    return ExprTypeDoc::TIRPrimitive("Tuple").CallWith(fields);
  }
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const Buffer& buffer, TVMScriptUnifiedPrinter& p) {
  p.OnBufferUsed(buffer);
  return IdentifierDoc(buffer->name);
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const BufferStore& op, TVMScriptUnifiedPrinter& p) {
  AssignDoc doc;
  ExprDoc buf_var = p.ToExprDoc(op->buffer);
  doc->target = buf_var.IndexWith(p.ToExprDocArray(op->indices));
  doc->value = p.ToExprDoc(op->value);
  return doc;
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const Var& var, TVMScriptUnifiedPrinter& p) {
  p.OnVarUsed(var);
  return IdentifierDoc(var->name_hint);
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const BufferLoad& n, TVMScriptUnifiedPrinter& p) {
  return p.ToExprDoc(n->buffer).IndexWith(p.ToExprDocArray(n->indices));
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const FloatImm& n, TVMScriptUnifiedPrinter& p) {
  return LiteralValueDoc(n);
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const IntImm& n, TVMScriptUnifiedPrinter& p) {
  return LiteralValueDoc(n);
});

TVMSCRIPT_PRINTER_DOC_PRODUCER([](const String& n, TVMScriptUnifiedPrinter& p) {
  return LiteralValueDoc(n);
});

#define TVMSCRIPT_PRINTER_REGISTER_BINOP_DOC_PRODUCER(OpRef)                         \
  TVMSCRIPT_PRINTER_DOC_PRODUCER(([](const OpRef& ref, TVMScriptUnifiedPrinter& p) { \
    OperationDoc doc;                                                                \
    doc->kind = OperationDocNode::OperationKind::OpRef;                              \
    doc->operands = {p.ToExprDoc(ref->a), p.ToExprDoc(ref->b)};                      \
    return doc;                                                                      \
  }));

TVMSCRIPT_PRINTER_REGISTER_BINOP_DOC_PRODUCER(Mul)
TVMSCRIPT_PRINTER_REGISTER_BINOP_DOC_PRODUCER(Div)
TVMSCRIPT_PRINTER_REGISTER_BINOP_DOC_PRODUCER(FloorDiv)
TVMSCRIPT_PRINTER_REGISTER_BINOP_DOC_PRODUCER(Add)
TVMSCRIPT_PRINTER_REGISTER_BINOP_DOC_PRODUCER(Sub)

}  // namespace printer
}  // namespace script
}  // namespace tvm
