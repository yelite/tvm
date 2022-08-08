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

#include <tvm/node/object_path.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/script/printer/var_table.h>

namespace tvm {
namespace script {
namespace printer {

String GenerateUniqueName(const String& name_hint, std::unordered_set<String>* defined_names) {
  String name = name_hint;
  for (int i = 1; !defined_names->insert(name).second; ++i) {
    name = name_hint + "_" + std::to_string(i);
  }
  return name;
}

ExprDoc VarTableNode::Define(const ObjectRef& obj, DocFactory doc_factory,
                             const ObjectPath& object_path) {
  ExprDoc doc = doc_factory();
  doc->source_paths.push_back(object_path);

  ICHECK(obj2info.find(obj) == obj2info.end()) << "Duplicated object: " << obj;

  // If doc_factory returns an IdDoc, add it to this->defined_names.
  Optional<String> name(NullOpt);
  if (const auto* id_doc = doc.as<IdDocNode>()) {
    name = id_doc->name;
    auto result = defined_names.insert(id_doc->name);
    ICHECK(result.second) << "Duplicated name: " << id_doc->name
                          << ". Please pass the variable name as String to `Define` directly so "
                             "that it can be auto renamed.";
  }

  obj2info.insert({obj, VariableInfo{std::move(doc_factory), name}});

  return doc;
}

IdDoc VarTableNode::Define(const ObjectRef& obj, const String& name_hint,
                           const ObjectPath& object_path) {
  String name = GenerateUniqueName(name_hint, &this->defined_names);
  DocFactory doc_factory = [name]() { return IdDoc(name); };

  auto result = obj2info.insert({obj, VariableInfo{std::move(doc_factory), name}});
  ICHECK(result.second) << "Duplicated object: " << obj;

  IdDoc def_doc(name);
  def_doc->source_paths.push_back(object_path);
  return def_doc;
}

void VarTableNode::Remove(const ObjectRef& obj) {
  auto it = obj2info.find(obj);
  ICHECK(it != obj2info.end()) << "No such object: " << obj;

  if (it->second.name.defined()) {
    defined_names.erase(it->second.name.value());
  }
  obj2info.erase(it);
}

Optional<ExprDoc> VarTableNode::GetVarDoc(const ObjectRef& obj,
                                          const ObjectPath& object_path) const {
  auto it = obj2info.find(obj);
  if (it == obj2info.end()) {
    return NullOpt;
  }
  ExprDoc doc = it->second.doc_factory();
  doc->source_paths.push_back(object_path);
  return doc;
}

bool VarTableNode::IsVarDefined(const ObjectRef& obj) const { return obj2info.count(obj); }

VarTable::VarTable() { data_ = make_object<VarTableNode>(); }

TVM_REGISTER_NODE_TYPE(VarTableNode);
TVM_REGISTER_GLOBAL("script.printer.VarTable").set_body_typed([]() { return VarTable(); });
TVM_REGISTER_GLOBAL("script.printer.VarTableDefineByName")
    .set_body_method<VarTable, VarTableNode, IdDoc, const ObjectRef&, const String&,
                     const ObjectPath&>(&VarTableNode::Define);
TVM_REGISTER_GLOBAL("script.printer.VarTableDefineByFactory")
    .set_body_typed([](VarTable var_table, const ObjectRef& obj, runtime::PackedFunc factory,
                       const ObjectPath& object_path) {
      return var_table->Define(
          obj, [f = std::move(factory)]() { return f(); }, object_path);
    });
TVM_REGISTER_GLOBAL("script.printer.VarTableRemove")
    .set_body_method<VarTable>(&VarTableNode::Remove);
TVM_REGISTER_GLOBAL("script.printer.VarTableGetVarDoc")
    .set_body_method<VarTable>(&VarTableNode::GetVarDoc);
TVM_REGISTER_GLOBAL("script.printer.VarTableIsVarDefined")
    .set_body_method<VarTable>(&VarTableNode::IsVarDefined);

}  // namespace printer
}  // namespace script
}  // namespace tvm
