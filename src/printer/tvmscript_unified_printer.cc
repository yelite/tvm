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

#include "tvmscript_unified_printer.h"

#include <tvm/node/functor.h>

#include <cstdint>
#include <string>

#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/string.h"
#include "tvm/runtime/object.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/function.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt.h"

namespace tvm {

using namespace tir;

class DocPrinter {
 public:
  virtual ~DocPrinter() = default;

  virtual String Print(CodeDoc doc) {};

 protected:
  virtual void Print(const CodeDoc& doc){};

  virtual void Print(const LiteralStringDoc& doc) = 0;
  virtual void Print(const LiteralNumberDoc& doc) = 0;
  virtual void Print(const ConstDoc& doc) = 0;
  virtual void Print(const IdentifierDoc& doc) = 0;
  virtual void Print(const AttrAccessDoc& doc) = 0;
  virtual void Print(const IndexDoc& doc) = 0;
  virtual void Print(const BinOpDoc& doc) = 0;
  virtual void Print(const CallDoc& doc) = 0;
  virtual void Print(const TupleDoc& doc) = 0;
  virtual void Print(const SeqStmtDoc& doc) = 0;
  virtual void Print(const ScopeDoc& doc) = 0;
  virtual void Print(const ForDoc& doc) = 0;
  virtual void Print(const AssignDoc& doc) = 0;
  virtual void Print(const ExprTypeDoc& doc) = 0;
  virtual void Print(const TypeCallDoc& doc) = 0;
  virtual void Print(const FunctionDoc& doc) = 0;
  virtual void Print(const FunctionArgDoc& doc) = 0;
};

class PythonDocPrinter : public DocPrinter {
 protected:
  virtual void Print(const LiteralStringDoc& doc){};
  virtual void Print(const LiteralNumberDoc& doc){};
  virtual void Print(const ConstDoc& doc){};
  virtual void Print(const IdentifierDoc& doc){};
  virtual void Print(const AttrAccessDoc& doc){};
  virtual void Print(const IndexDoc& doc){};
  virtual void Print(const BinOpDoc& doc){};
  virtual void Print(const CallDoc& doc){};
  virtual void Print(const TupleDoc& doc){};
  virtual void Print(const SeqStmtDoc& doc){};
  virtual void Print(const ScopeDoc& doc){};
  virtual void Print(const ForDoc& doc){};
  virtual void Print(const AssignDoc& doc){};
  virtual void Print(const ExprTypeDoc& doc){};
  virtual void Print(const TypeCallDoc& doc){};
  virtual void Print(const FunctionDoc& doc){};
  virtual void Print(const FunctionArgDoc& doc){};
};

class TVMScriptUnifiedPrinter {
 public:
  explicit TVMScriptUnifiedPrinter(std::unique_ptr<DocPrinter> element_printer)
      : doc_printer_(std::move(element_printer)){};

  using FType = NodeFunctor<CodeDoc(const ObjectRef&, TVMScriptUnifiedPrinter&)>;
  static FType& vtable();

  String PrintNode(const ObjectRef& ref);

  template <typename T, typename = std::enable_if_t<std::is_base_of<CodeDoc, T>::value>>
  T ToDoc(const ObjectRef& ref);

  template <typename DocType, typename NodeType,
            typename = std::enable_if_t<std::is_base_of<CodeDoc, DocType>::value>,
            typename = std::enable_if_t<std::is_base_of<ObjectRef, NodeType>::value>>
  Array<DocType> ToDocArray(const Array<NodeType>& refs);

  ExprDoc ToExprDoc(const ObjectRef& ref) { return ToDoc<ExprDoc>(ref); }

  template <typename NodeType,
            typename = std::enable_if_t<std::is_base_of<ObjectRef, NodeType>::value>>
  Array<ExprDoc> ToExprDocArray(const Array<NodeType>& refs) {
    return ToDocArray<ExprDoc>(refs);
  }

  CodeDoc PrintExtraVarDeclaration();

  TypeDoc GetBufferTypeDoc(const Buffer& buf);
  TypeDoc GetVarTypeDoc(const Var& var);

 protected:
  std::unique_ptr<DocPrinter> doc_printer_;
};

TVMScriptUnifiedPrinter::FType& TVMScriptUnifiedPrinter::vtable() {
  static FType inst;
  return inst;
}

String TVMScriptUnifiedPrinter::PrintNode(const ObjectRef& ref) {
  auto element = ToDoc<CodeDoc>(ref);
  return doc_printer_->Print(element);
}

template <typename T, typename>
T TVMScriptUnifiedPrinter::ToDoc(const ObjectRef& ref) {
  CodeDoc element = vtable()(ref, *this);
  element->origin_ir_node = ref;
  return Downcast<T>(element);
}

template <typename DocType, typename NodeType, typename, typename>
Array<DocType> TVMScriptUnifiedPrinter::ToDocArray(const Array<NodeType>& refs) {
  Array<DocType> result;
  for (auto& n : refs) {
    result.push_back(ToDoc<DocType>(n));
  }
  return result;
}

TypeDoc TVMScriptUnifiedPrinter::GetBufferTypeDoc(const Buffer& buf) {
  TypeCallDoc type_doc;
  type_doc->base = TypeDoc::TIRPrimitive("buffer");
  for (auto d : ToExprDocArray(buf->shape)) {
    type_doc->params.push_back(ExprTypeDoc(d));
  }
  return type_doc;
}

TypeDoc TVMScriptUnifiedPrinter::GetVarTypeDoc(const Var& var) {
  return ToDoc<TypeDoc>(GetType(var));
}

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<PrimFuncNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto func = Downcast<PrimFunc>(n);
      FunctionDoc func_doc;
      String func_name = "func";
      const auto& it = func->attrs->dict.find("global_symbol");
      if (it != func->attrs->dict.end()) {
        func_name = Downcast<String>((*it).second);
      }
      func_doc->name = func_name;

      std::vector<FunctionArgDoc> params;
      for (const auto& param : func->params) {
        auto it = func->buffer_map.find(param);
        if (it != func->buffer_map.end()) {
          // Buffer
          const Buffer& buf = (*it).second;
          params.emplace_back(p.ToDoc<IdentifierDoc>(buf), p.GetBufferTypeDoc(buf));
        } else {
          // Var
          params.emplace_back(p.ToDoc<IdentifierDoc>(param), p.GetVarTypeDoc(param));
        }
      }
      func_doc->args = std::move(params);
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

      return func_doc;
    });

SeqStmtDoc GetBlockVarsDeclarations(const BlockRealize block_realize, TVMScriptUnifiedPrinter& p) {
  SeqStmtDoc doc;
  const auto block = block_realize->block;
  ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());

  auto axis_expr = ExprDoc::TIRBuilderAttribute("axis");

  // TODO: handle remap

  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& value = block_realize->iter_values[i];
    // p.onVarEnterScope(iter_var->var);
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
    ExprDoc arg_dom;
    if (is_zero(dom->min)) {
      arg_dom = p.ToExprDoc(dom->extent);
    } else {
      arg_dom = TupleDoc{p.ToExprDoc(dom->min), p.ToExprDoc(dom->min + dom->extent)};
    }
    assign_stmt->value = axis_expr.AccessAttr(axis_type).CallWith(
        std::initializer_list<ExprDoc>{arg_dom, p.ToExprDoc(value)});

    doc.Add(assign_stmt);
  }
  return doc;
}

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BlockRealizeNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto block_realize = Downcast<BlockRealize>(n);
      const auto block = block_realize->block;

      ScopeDoc scope_doc;
      // TODO: optional info
      // print block name and block vars
      scope_doc->scope = ExprDoc::TIRBuilderAttribute("block").CallWith(
          std::initializer_list<ExprDoc>{LiteralStringDoc(block->name_hint)});

      SeqStmtDoc body;

      body.Extend(GetBlockVarsDeclarations(block_realize, p));
      body.Add(p.ToDoc<StmtDoc>(block));

      scope_doc->body = std::move(body);

      return scope_doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BlockNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto block = Downcast<Block>(n);
      // TODO: T.alloc_buffer and match_buffer and init
      return p.ToDoc<StmtDoc>(block->body);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<ForNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto for_ref = Downcast<For>(n);
      ForDoc doc;
      // p.onVarEnterScope(for_ref->loop_var);
      doc->target = p.ToDoc<IdentifierDoc>(for_ref->loop_var);
      auto for_kind = ExprDoc::TIRBuilderAttribute(ForKind2String(for_ref->kind));
      if (is_zero(for_ref->min)) {
        doc->iter = for_kind.CallWith(std::initializer_list<ExprDoc>{p.ToExprDoc(for_ref->extent)});
      } else {
        doc->iter = for_kind.CallWith(std::initializer_list<ExprDoc>{
            p.ToExprDoc(for_ref->min), p.ToExprDoc(for_ref->min + for_ref->extent)});
      }
      // TODO: annotation, thread binding
      doc->body = p.ToDoc<StmtDoc>(for_ref->body);
      // p.onVarExitScope(for_ref->loop_var);
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<PrimTypeNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto type = Downcast<PrimType>(n);
      return TypeDoc::TIRPrimitive(runtime::DLDataType2String(type->dtype));
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<TupleTypeNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto type = Downcast<TupleType>(n);
      if (type->fields.empty()) {
        return TypeDoc::NoneType();
      } else {
        std::vector<TypeDoc> fields;
        for (auto& field : type->fields) {
          fields.push_back(p.ToDoc<TypeDoc>(field));
        }
        return TypeDoc::TIRPrimitive("Tuple").CallWith(fields);
      }
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BufferNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const Buffer buffer = Downcast<Buffer>(n);
      // p.onBufferUsed(buffer);
      return IdentifierDoc(buffer->name);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BufferStoreNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const BufferStore op = Downcast<BufferStore>(n);
      AssignDoc doc;
      auto buf_var = p.ToExprDoc(op->buffer);
      doc->target = buf_var.IndexWith(p.ToExprDocArray(op->indices));
      doc->value = p.ToExprDoc(op->value);
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<VarNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const Var var = Downcast<Var>(n);
      // p.onVarUsed(var);
      return IdentifierDoc(var->name_hint);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BufferLoadNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto buffer_load = Downcast<BufferLoad>(n);
      return p.ToExprDoc(buffer_load->buffer).IndexWith(p.ToExprDocArray(buffer_load->indices));
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<FloatImmNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto node_ref = Downcast<FloatImm>(n);
      return LiteralNumberDoc(node_ref);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<IntImmNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto node_ref = Downcast<IntImm>(n);
      return LiteralNumberDoc(node_ref);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<StringObj>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc {
      const auto s = Downcast<String>(n);
      return LiteralStringDoc(s);
    });

#define TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(OpNode, OpKind)                         \
  TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)                                    \
      .set_dispatch<OpNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> CodeDoc { \
        const auto* node = n.as<OpNode>();                                                  \
        BinOpDoc doc;                                                                       \
        doc->kind = BinOpDocNode::BinOpKind::OpKind;                                        \
        doc->lhs = p.ToExprDoc(node->a);                                                    \
        doc->rhs = p.ToExprDoc(node->b);                                                    \
        return doc;                                                                         \
      });

TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(MulNode, Mul)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(DivNode, Div)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(FloorDivNode, FloorDiv)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(AddNode, Add)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(SubNode, Sub)

String AsTVMScriptUnified(const ObjectRef& node, const String& tir_prefix) {
  auto printer = TVMScriptUnifiedPrinter(std::make_unique<PythonDocPrinter>());
  return printer.PrintNode(node);
}

TVM_REGISTER_GLOBAL("experiment.AsTVMScript").set_body_typed(AsTVMScriptUnified);

}  // namespace tvm
