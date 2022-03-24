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

#include "tvm/ir/expr.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/container/string.h"
#include "tvm/runtime/object.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/function.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace script {

using namespace tir;

class DocPrinter {
 public:
  virtual ~DocPrinter() = default;

  virtual String Print(std::initializer_list<Doc> docs) {
    output_.str("");
    for (auto& doc : docs) {
      indent_ = 0;
      PrintDoc(doc);
      output_ << "\n";
    }
    auto result = output_.str();
    output_.str("");
    return result;
  };

 protected:
  virtual void PrintDoc(const Doc& doc) {
    if (doc->IsInstance<LiteralValueDocNode>()) {
      PrintDoc(Downcast<LiteralValueDoc>(doc));
    } else if (doc->IsInstance<ConstDocNode>()) {
      PrintDoc(Downcast<ConstDoc>(doc));
    } else if (doc->IsInstance<IdentifierDocNode>()) {
      PrintDoc(Downcast<IdentifierDoc>(doc));
    } else if (doc->IsInstance<AttrAccessDocNode>()) {
      PrintDoc(Downcast<AttrAccessDoc>(doc));
    } else if (doc->IsInstance<IndexDocNode>()) {
      PrintDoc(Downcast<IndexDoc>(doc));
    } else if (doc->IsInstance<OperationDocNode>()) {
      PrintDoc(Downcast<OperationDoc>(doc));
    } else if (doc->IsInstance<CallDocNode>()) {
      PrintDoc(Downcast<CallDoc>(doc));
    } else if (doc->IsInstance<TupleDocNode>()) {
      PrintDoc(Downcast<TupleDoc>(doc));
    } else if (doc->IsInstance<SeqStmtDocNode>()) {
      PrintDoc(Downcast<SeqStmtDoc>(doc));
    } else if (doc->IsInstance<ScopeDocNode>()) {
      PrintDoc(Downcast<ScopeDoc>(doc));
    } else if (doc->IsInstance<ForDocNode>()) {
      PrintDoc(Downcast<ForDoc>(doc));
    } else if (doc->IsInstance<AssignDocNode>()) {
      PrintDoc(Downcast<AssignDoc>(doc));
    } else if (doc->IsInstance<ExprTypeDocNode>()) {
      PrintDoc(Downcast<ExprTypeDoc>(doc));
    } else if (doc->IsInstance<TypeCallDocNode>()) {
      PrintDoc(Downcast<TypeCallDoc>(doc));
    } else if (doc->IsInstance<FunctionDocNode>()) {
      PrintDoc(Downcast<FunctionDoc>(doc));
    } else if (doc->IsInstance<FunctionArgDocNode>()) {
      PrintDoc(Downcast<FunctionArgDoc>(doc));
    } else {
      LOG(FATAL) << "Do not know how to print " << doc->GetTypeKey();
    }
  };

  virtual void PrintDoc(const LiteralValueDoc& doc) = 0;
  virtual void PrintDoc(const ConstDoc& doc) = 0;
  virtual void PrintDoc(const IdentifierDoc& doc) = 0;
  virtual void PrintDoc(const AttrAccessDoc& doc) = 0;
  virtual void PrintDoc(const IndexDoc& doc) = 0;
  virtual void PrintDoc(const OperationDoc& doc) = 0;
  virtual void PrintDoc(const CallDoc& doc) = 0;
  virtual void PrintDoc(const TupleDoc& doc) = 0;
  virtual void PrintDoc(const SeqStmtDoc& doc) = 0;
  virtual void PrintDoc(const ScopeDoc& doc) = 0;
  virtual void PrintDoc(const ForDoc& doc) = 0;
  virtual void PrintDoc(const AssignDoc& doc) = 0;
  virtual void PrintDoc(const ExprTypeDoc& doc) = 0;
  virtual void PrintDoc(const TypeCallDoc& doc) = 0;
  virtual void PrintDoc(const FunctionDoc& doc) = 0;
  virtual void PrintDoc(const FunctionArgDoc& doc) = 0;

  using OutputStream = std::ostringstream;

  OutputStream& NewLine() {
    output_ << "\n" << std::string(indent_, ' ');
    return output_;
  }

  OutputStream output_;
  int indent_ = 0;
};

class PythonDocPrinter : public DocPrinter {
 public:
  PythonDocPrinter(String tir_prefix) : tir_prefix_(tir_prefix) {}

 protected:
  using DocPrinter::PrintDoc;

  virtual void PrintDoc(const LiteralValueDoc& doc) override {
    auto& value = doc->value;
    if (value->IsInstance<FloatImmNode>() || value->IsInstance<IntImmNode>()) {
      PrintNumberNode(Downcast<PrimExpr>(doc->value));
    } else if (value->IsInstance<StringObj>()) {
      PrintStringLiteral(Downcast<String>(value));
    } else if (value->IsInstance<tir::StringImmNode>()) {
      PrintStringLiteral(Downcast<StringImm>(value)->value);
    }
  };

  virtual void PrintDoc(const ConstDoc& doc) override {
    switch (doc->kind) {
      case ConstDocNode::ConstKind::TIRBuilder:
        output_ << tir_prefix_;
        break;
      case ConstDocNode::ConstKind::RelaxBuilder:
        LOG(FATAL) << "RelaxBuilder constant not supported yet";
        break;
      case ConstDocNode::ConstKind::None:
        output_ << "None";
        break;
    }
  };

  virtual void PrintDoc(const IdentifierDoc& doc) override { output_ << doc->name; };

  virtual void PrintDoc(const AttrAccessDoc& doc) override {
    PrintDoc(doc->value);
    output_ << ".";
    PrintDoc(doc->attr);
  };

  virtual void PrintDoc(const IndexDoc& doc) override {
    PrintDoc(doc->value);
    PrintJoinedElements("[", doc->indices, ", ", "]", "()");
  };

  virtual void PrintDoc(const OperationDoc& doc) override {
    PrintDoc(doc->operands[0]);
    switch (doc->kind) {
      case OperationDocNode::OperationKind::Add:
        output_ << " + ";
        break;
      case OperationDocNode::OperationKind::Sub:
        output_ << " - ";
        break;
      case OperationDocNode::OperationKind::Mul:
        output_ << " * ";
        break;
      case OperationDocNode::OperationKind::Div:
        output_ << " / ";
        break;
      case OperationDocNode::OperationKind::FloorDiv:
        output_ << " // ";
        break;
    }
    PrintDoc(doc->operands[1]);
  };

  virtual void PrintDoc(const CallDoc& doc) override {
    PrintDoc(doc->callee);
    PrintJoinedElements("(", doc->args, ", ", ")");
  };

  virtual void PrintDoc(const TupleDoc& doc) override {
    auto size = doc->elements.size();
    if (size == 0) {
      output_ << "tuple()";
    } else if (size == 1) {
      output_ << "(";
      PrintDoc(doc->elements[0]);
      output_ << ",)";
    } else {
      PrintJoinedElements("(", doc->elements, ", ", ")");
    }
  };

  virtual void PrintDoc(const SeqStmtDoc& doc) override {
    for (auto& stmt : doc->seq) {
      PrintDoc(stmt);
    }
  };

  virtual void PrintDoc(const ScopeDoc& doc) override {
    NewLine() << "with ";
    PrintDoc(doc->scope);
    output_ << ":";

    PrintWithIncreasedIndent(doc->body);
  };

  virtual void PrintDoc(const ForDoc& doc) override {
    NewLine() << "for ";
    PrintDoc(doc->target);
    output_ << " in ";
    PrintDoc(doc->iter);
    output_ << ":";

    PrintWithIncreasedIndent(doc->body);
  };

  virtual void PrintDoc(const AssignDoc& doc) override {
    // TODO: Assign Kind, Type Annotation
    NewLine();
    PrintDoc(doc->target);
    if (doc->type) {
      output_ << ": ";
      PrintDoc(doc->type.value());
    }
    if (doc->value) {
      output_ << " = ";
      PrintDoc(doc->value.value());
    }
  };

  virtual void PrintDoc(const ExprTypeDoc& doc) override { PrintDoc(doc->expr); };

  virtual void PrintDoc(const TypeCallDoc& doc) override {
    PrintDoc(doc->base);
    PrintJoinedElements("[", doc->args, ", ", "]");
  };

  virtual void PrintDoc(const FunctionDoc& doc) override {
    NewLine() << "@" << tir_prefix_ << ".prim_func";
    NewLine() << "def ";
    PrintDoc(doc->name);
    PrintJoinedElements("(", doc->args, ", ", ")");
    output_ << " -> ";
    PrintDoc(doc->return_type);
    output_ << ":";
    PrintWithIncreasedIndent(doc->body);
  };

  virtual void PrintDoc(const FunctionArgDoc& doc) override {
    PrintDoc(doc->name);
    output_ << ": ";
    PrintDoc(doc->type);
  };

 private:
  String tir_prefix_;

  int indent_spaces_ = 4;

  void IncreaseIndent() { indent_ += indent_spaces_; }

  void DecreaseIndent() { indent_ -= indent_spaces_; }

  void PrintWithIncreasedIndent(const Doc& doc) {
    IncreaseIndent();
    PrintDoc(doc);
    DecreaseIndent();
  }

  template <typename ContainerType>
  void PrintJoinedElements(const std::string& left, const ContainerType& elements,
                           const std::string& separator, const std::string& right,
                           const std::string& default_content = "") {
    output_ << left;
    bool is_first = true;
    for (auto& element : elements) {
      if (is_first) {
        is_first = false;
      } else {
        output_ << separator;
      }
      PrintObject(element);
    }
    if (is_first) {
      // This means that `elements` is empty
      output_ << default_content;
    }
    output_ << right;
  }

  template <typename... Arg>
  void PrintTIRPrimitiveCall(const std::string& primitive, Arg... args) {
    output_ << tir_prefix_ << "." << primitive;
    PrintJoinedElements("(", std::vector<typename std::common_type_t<Arg...>>{args...}, ", ", ")");
  }

  template <typename ObjType>
  std::enable_if_t<std::is_base_of<Doc, ObjType>::value, void> PrintObject(const ObjType& obj) {
    PrintDoc(obj);
  }

  template <typename ObjType>
  std::enable_if_t<!std::is_base_of<Doc, ObjType>::value, void> PrintObject(const ObjType& obj) {
    output_ << obj;
  }

  void PrintStringLiteral(const String& string) {
    // TODO: Escape and smart quote (choose ' or " automatically)
    output_ << "\"" << string << "\"";
  }

  void PrintNumberNode(const PrimExpr& expr) {
    auto& dtype = expr->dtype;
    std::ostringstream number_value;

    if (expr->IsInstance<IntImmNode>()) {
      number_value << Downcast<IntImm>(expr)->value;
    } else if (expr->IsInstance<FloatImmNode>()) {
      number_value.precision(17);
      number_value << Downcast<FloatImm>(expr)->value;
    } else {
      LOG(FATAL) << "Do not know how to process " << expr->GetTypeKey() << " as literal number";
    }

    if (dtype == DataType::Int(32)) {
      output_ << number_value.str();
    } else if (dtype == DataType::Bool()) {
      output_ << (Downcast<IntImm>(expr)->value ? "True" : "False");
    } else {
      PrintTIRPrimitiveCall(runtime::DLDataType2String(dtype), number_value.str());
    }
  };
};

class TVMScriptUnifiedPrinter {
 public:
  explicit TVMScriptUnifiedPrinter(std::unique_ptr<DocPrinter> element_printer)
      : doc_printer_(std::move(element_printer)){};

  using FType = NodeFunctor<Doc(const ObjectRef&, TVMScriptUnifiedPrinter&)>;
  static FType& vtable();

  String PrintNode(const ObjectRef& ref);

  template <typename T, typename = std::enable_if_t<std::is_base_of<Doc, T>::value>>
  T ToDoc(const ObjectRef& ref);

  template <typename DocType, typename NodeType,
            typename = std::enable_if_t<std::is_base_of<Doc, DocType>::value>,
            typename = std::enable_if_t<std::is_base_of<ObjectRef, NodeType>::value>>
  Array<DocType> ToDocArray(const Array<NodeType>& refs);

  ExprDoc ToExprDoc(const ObjectRef& ref) { return ToDoc<ExprDoc>(ref); }

  template <typename NodeType,
            typename = std::enable_if_t<std::is_base_of<ObjectRef, NodeType>::value>>
  Array<ExprDoc> ToExprDocArray(const Array<NodeType>& refs) {
    return ToDocArray<ExprDoc>(refs);
  }

  Doc PrintExtraVarDeclaration();

  TypeDoc GetBufferTypeDoc(const Buffer& buf);
  TypeDoc GetVarTypeDoc(const Var& var);

  void OnVarUsed(const Var& var) {
    auto& name = var->name_hint;
    if (!context_manager->GetVar(name) && !HasFreeVar(name, var)) {
      AssignDoc declaration;
      declaration->target = IdentifierDoc(name);
      declaration->type = GetVarTypeDoc(var);
      prelude_.push_back(std::move(declaration));
      free_vars_.Set(name, var);
    }
  }

  void OnBufferUsed(const Buffer& buffer) {
    auto& name = buffer->name;
    if (!context_manager->GetVar(name)) {
      AssignDoc declaration;
      declaration->target = IdentifierDoc(name);
      declaration->type = GetBufferTypeDoc(buffer);
      prelude_.push_back(std::move(declaration));
      free_vars_.Set(name, buffer);
    }
  }

  PrinterContextManager context_manager;

 protected:
  std::unique_ptr<DocPrinter> doc_printer_;
  Array<StmtDoc> prelude_;
  Map<String, ObjectRef> free_vars_;

  bool HasFreeVar(const String& name, const ObjectRef& var) {
    auto free_var = free_vars_.Get(name);
    return free_var && free_var.value() == var;
  }
};

TVMScriptUnifiedPrinter::FType& TVMScriptUnifiedPrinter::vtable() {
  static FType inst;
  return inst;
}

String TVMScriptUnifiedPrinter::PrintNode(const ObjectRef& ref) {
  auto element = ToDoc<Doc>(ref);
  if (prelude_.empty()) {
    return doc_printer_->Print({element});
  } else {
    return doc_printer_->Print({SeqStmtDoc(prelude_), element});
  }
}

template <typename T, typename>
T TVMScriptUnifiedPrinter::ToDoc(const ObjectRef& ref) {
  Doc element = vtable()(ref, *this);
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
  type_doc->base = TypeDoc::TIRPrimitive("Buffer");

  if (buf->shape.size() > 1) {
    TupleDoc shape_doc;
    shape_doc->elements = ToExprDocArray(buf->shape);
    type_doc->args.push_back(ExprTypeDoc(shape_doc));
  } else {
    type_doc->args.push_back(ExprTypeDoc(ToExprDoc(buf->shape[0])));
  }
  type_doc->args.push_back(ExprTypeDoc(LiteralValueDoc(runtime::DLDataType2String(buf->dtype))));
  return type_doc;
}

TypeDoc TVMScriptUnifiedPrinter::GetVarTypeDoc(const Var& var) {
  return ToDoc<TypeDoc>(GetType(var));
}

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<PrimFuncNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const auto func = Downcast<PrimFunc>(n);
      FunctionDoc func_doc;
      auto context = p.context_manager->EnterContext<PrinterFunctionContext>();

      String func_name = "func";
      const auto& it = func->attrs->dict.find("global_symbol");
      if (it != func->attrs->dict.end()) {
        func_name = Downcast<String>((*it).second);
      }
      func_doc->name = func_name;

      auto& params = func_doc->args;
      for (const auto& param : func->params) {
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
  const auto block = block_realize->block;
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
    assign_stmt->value = ExprDoc::TIRBuilderAttribute("axis", std::move(axis_type))
                             .CallWith(std::move(dom_arg), p.ToExprDoc(value));

    doc.Add(assign_stmt);
  }
  return doc;
}

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BlockRealizeNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const auto block_realize = Downcast<BlockRealize>(n);
      const auto block = block_realize->block;
      ScopeDoc scope_doc;
      auto context = p.context_manager->EnterContext<PrinterBlockContext>();

      // TODO: optional info
      // print block name and block vars
      scope_doc->scope =
          ExprDoc::TIRBuilderAttribute("block").CallWith(LiteralValueDoc(block->name_hint));

      SeqStmtDoc body;

      body.Extend(GetBlockVarsDeclarations(block_realize, p));
      body.Add(p.ToDoc<StmtDoc>(block));

      scope_doc->body = std::move(body);

      p.context_manager->ExitContext(std::move(context));
      return scope_doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BlockNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const auto block = Downcast<Block>(n);
      // TODO: T.alloc_buffer and match_buffer and init
      return p.ToDoc<StmtDoc>(block->body);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<ForNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const auto for_ref = Downcast<For>(n);
      ForDoc doc;
      auto context = p.context_manager->EnterContext<PrinterLoopContext>();
      p.context_manager->AddVar(for_ref->loop_var);

      doc->target = p.ToDoc<IdentifierDoc>(for_ref->loop_var);
      auto for_kind = ExprDoc::TIRBuilderAttribute(ForKind2String(for_ref->kind));
      if (is_zero(for_ref->min)) {
        doc->iter = for_kind.CallWith(p.ToExprDoc(for_ref->extent));
      } else {
        doc->iter = for_kind.CallWith(p.ToExprDoc(for_ref->min),
                                      p.ToExprDoc(for_ref->min + for_ref->extent));
      }
      // TODO: annotation, thread binding
      doc->body = p.ToDoc<StmtDoc>(for_ref->body);

      p.context_manager->ExitContext(std::move(context));
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<PrimTypeNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const auto type = Downcast<PrimType>(n);
      return TypeDoc::TIRPrimitive(runtime::DLDataType2String(type->dtype));
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<TupleTypeNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
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
    .set_dispatch<BufferNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const Buffer buffer = Downcast<Buffer>(n);
      p.OnBufferUsed(buffer);
      return IdentifierDoc(buffer->name);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BufferStoreNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const BufferStore op = Downcast<BufferStore>(n);
      AssignDoc doc;
      auto buf_var = p.ToExprDoc(op->buffer);
      doc->target = buf_var.IndexWith(p.ToExprDocArray(op->indices));
      doc->value = p.ToExprDoc(op->value);
      return doc;
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<VarNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const Var var = Downcast<Var>(n);
      p.OnVarUsed(var);
      return IdentifierDoc(var->name_hint);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<BufferLoadNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const auto buffer_load = Downcast<BufferLoad>(n);
      return p.ToExprDoc(buffer_load->buffer).IndexWith(p.ToExprDocArray(buffer_load->indices));
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<FloatImmNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const auto node_ref = Downcast<FloatImm>(n);
      return LiteralValueDoc(node_ref);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<IntImmNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const auto node_ref = Downcast<IntImm>(n);
      return LiteralValueDoc(node_ref);
    });

TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)
    .set_dispatch<StringObj>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc {
      const auto s = Downcast<String>(n);
      return LiteralValueDoc(s);
    });

#define TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(OpNode, OpKind)                     \
  TVM_STATIC_IR_FUNCTOR(TVMScriptUnifiedPrinter, vtable)                                \
      .set_dispatch<OpNode>([](const ObjectRef& n, TVMScriptUnifiedPrinter& p) -> Doc { \
        const auto* node = n.as<OpNode>();                                              \
        OperationDoc doc;                                                               \
        doc->kind = OperationDocNode::OperationKind::OpKind;                            \
        doc->operands = {p.ToExprDoc(node->a), p.ToExprDoc(node->b)};                   \
        return doc;                                                                     \
      });

TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(MulNode, Mul)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(DivNode, Div)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(FloorDivNode, FloorDiv)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(AddNode, Add)
TVM_DECLARE_TVMSCRIPT_UNIFIED_PRINTER_BINOP(SubNode, Sub)

String AsTVMScriptUnified(const ObjectRef& node, const String& tir_prefix) {
  auto printer = TVMScriptUnifiedPrinter(std::make_unique<PythonDocPrinter>(tir_prefix));
  return printer.PrintNode(node);
}

TVM_REGISTER_GLOBAL("experiment.AsTVMScript").set_body_typed(AsTVMScriptUnified);

}  // namespace script
}  // namespace tvm
