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
/*!
 * \brief Printer class to print TVMScript from Relax and TIR nodes.
 */
#ifndef TVM_SCRIPT_PRINTER_TVMSCRIPT_UNIFIED_PRINTER_H_
#define TVM_SCRIPT_PRINTER_TVMSCRIPT_UNIFIED_PRINTER_H_

#include "doc_printer.h"

namespace tvm {
namespace script {
namespace printer {

// Printer Context

class PrinterBaseContextNode : public Object {
 public:
  static constexpr const char* _type_key = "script.PrinterBaseContext";
  TVM_DECLARE_BASE_OBJECT_INFO(PrinterBaseContextNode, Object);
};

class PrinterBaseContext : public ObjectRef {
 public:
  PrinterBaseContext() { data_ = make_object<PrinterBaseContextNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterBaseContext, runtime::ObjectRef,
                                                    PrinterBaseContextNode);
};

class PrinterFunctionContextNode : public PrinterBaseContextNode {
 public:
  static constexpr const char* _type_key = "script.PrinterFunctionContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterFunctionContextNode, Object);
};

class PrinterFunctionContext : public PrinterBaseContext {
 public:
  PrinterFunctionContext() { data_ = make_object<PrinterFunctionContextNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterFunctionContext, PrinterBaseContext,
                                                    PrinterFunctionContextNode);
};

class PrinterLoopContextNode : public PrinterBaseContextNode {
 public:
  static constexpr const char* _type_key = "script.PrinterLoopContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterLoopContextNode, Object);
};

class PrinterLoopContext : public PrinterBaseContext {
 public:
  PrinterLoopContext() { data_ = make_object<PrinterLoopContextNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterLoopContext, PrinterBaseContext,
                                                    PrinterLoopContextNode);
};

class PrinterBlockContextNode : public PrinterBaseContextNode {
 public:
  static constexpr const char* _type_key = "script.PrinterBlockContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterBlockContextNode, Object);
};

class PrinterBlockContext : public PrinterBaseContext {
 public:
  PrinterBlockContext() { data_ = make_object<PrinterBlockContextNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterBlockContext, PrinterBaseContext,
                                                    PrinterBlockContextNode);
};

// TODO: Needs more thought on this
class PrinterContextManagerNode : public Object {
 public:
  Array<PrinterBaseContext> contexts;
  std::vector<Map<String, ObjectRef>> symbol_tables;

  template <typename ContextType>
  ContextType EnterContext() {
    ContextType context;
    contexts.push_back(context);
    PushNewSymbolTable();
    return context;
  }

  void ExitContext(PrinterBaseContext&& context) {
    // ICHECK_EQ(context, contexts.back())
    contexts.pop_back();
    symbol_tables.pop_back();
  }

  void AddVar(const tir::Buffer& buffer) { symbol_tables.back().Set(buffer->name, buffer); }

  void AddVar(const tir::Var& var) { symbol_tables.back().Set(var->name_hint, var); }

  Optional<ObjectRef> GetVar(const String& name) {
    if (symbol_tables.empty()) {
      return Optional<ObjectRef>();
    }
    Map<String, ObjectRef> current_table = symbol_tables.back();
    return current_table.Get(name);
  }

  static constexpr const char* _type_key = "script.PrinterContextManager";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterContextManagerNode, Object);

 private:
  void PushNewSymbolTable() {
    if (symbol_tables.empty()) {
      symbol_tables.emplace_back();
    } else {
      symbol_tables.emplace_back(symbol_tables.back());
    }
  }
};

class PrinterContextManager : public ObjectRef {
 public:
  PrinterContextManager() { data_ = make_object<PrinterContextManagerNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterContextManager, runtime::ObjectRef,
                                                    PrinterContextManagerNode);
};

class TVMScriptUnifiedPrinter;

// A modified version of NodeFunctor which supports auto downcast
class DocProducerRegistry {
 private:
  using InternalFType = std::function<Doc(const ObjectRef&, TVMScriptUnifiedPrinter&)>;

  template <typename NodeRef, typename DocType>
  using ProducerType = DocType (*)(const NodeRef&, TVMScriptUnifiedPrinter&);

  std::vector<InternalFType> producers_;

 public:
  bool can_dispatch(const ObjectRef& n) const {
    uint32_t type_index = n->type_index();
    return type_index < producers_.size() && producers_[type_index] != nullptr;
  }

  Doc operator()(const ObjectRef& n, TVMScriptUnifiedPrinter& p) const {
    ICHECK(can_dispatch(n)) << "DocProducerRegistry calls un-registered function on type "
                            << n->GetTypeKey();
    return producers_[n->type_index()](n, p);
  }

  template <typename NodeRef, typename DocType>
  DocProducerRegistry& register_producer(ProducerType<NodeRef, DocType> producer) {
    using NodeType = typename NodeRef::ContainerType;
    uint32_t tindex = NodeType::RuntimeTypeIndex();
    if (producers_.size() <= tindex) {
      producers_.resize(tindex + 1, nullptr);
    }
    ICHECK(producers_[tindex] == nullptr)
        << "Dispatch for " << NodeType::_type_key << " is already set";

    producers_[tindex] = [producer = std::move(producer)](const ObjectRef& ref,
                                                          TVMScriptUnifiedPrinter& p) {
      Doc doc = producer(Downcast<NodeRef>(ref), p);
      return doc;
    };
    return *this;
  }
};

class TVMScriptUnifiedPrinter {
 public:
  explicit TVMScriptUnifiedPrinter(std::unique_ptr<DocPrinter> element_printer)
      : doc_printer_(std::move(element_printer)){};

  static DocProducerRegistry& registry();

  String Print(const ObjectRef& ref);
  Doc PrintExtraVarDeclaration();

  template <typename T, typename = std::enable_if_t<std::is_base_of<Doc, T>::value>>
  T ToDoc(const ObjectRef& ref);

  template <typename DocType, typename NodeType>
  Array<DocType> ToDocArray(const Array<NodeType>& refs);

  ExprDoc ToExprDoc(const ObjectRef& ref) { return ToDoc<ExprDoc>(ref); }

  template <typename NodeType>
  Array<ExprDoc> ToExprDocArray(const Array<NodeType>& refs) {
    return ToDocArray<ExprDoc>(refs);
  }

  TypeDoc GetBufferTypeDoc(const tir::Buffer& buf);
  TypeDoc GetVarTypeDoc(const tir::Var& var);

  void OnVarUsed(const tir::Var& var) {
    const String& name = var->name_hint;
    if (!context_manager->GetVar(name) && !HasFreeVar(name, var)) {
      AssignDoc declaration;
      declaration->target = IdentifierDoc(name);
      declaration->type = GetVarTypeDoc(var);
      prelude_.push_back(std::move(declaration));
      free_vars_.Set(name, var);
    }
  }

  void OnBufferUsed(const tir::Buffer& buffer) {
    const String& name = buffer->name;
    if (!context_manager->GetVar(name) && !HasFreeVar(name, buffer)) {
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

  bool HasFreeVar(const String& name, const ObjectRef& var);
};

template <typename T, typename>
T TVMScriptUnifiedPrinter::ToDoc(const ObjectRef& ref) {
  Doc element = registry()(ref, *this);
  element->origin_ir_node = ref;
  return Downcast<T>(element);
}

template <typename DocType, typename NodeType >
Array<DocType> TVMScriptUnifiedPrinter::ToDocArray(const Array<NodeType>& refs) {
  Array<DocType> result;
  for (auto& n : refs) {
    result.push_back(ToDoc<DocType>(n));
  }
  return result;
}

#define TVMSCRIPT_PRINTER_DOC_PRODUCER(Producer)                               \
  TVM_STR_CONCAT(TVM_REG_FUNC_VAR_DEF(TVMScriptUnifiedPrinter), __COUNTER__) = \
      TVMScriptUnifiedPrinter::registry().register_producer(+Producer)

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TVMSCRIPT_UNIFIED_PRINTER_H_
