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
#ifndef TVM_SCRIPT_PRINTER_SYMBOL_TABLE_H_
#define TVM_SCRIPT_PRINTER_SYMBOL_TABLE_H_

#include <tvm/node/node.h>

#include "./doc.h"

namespace tvm {
namespace script {
namespace printer {

class SymbolTableNode : public Object {
 public:
  void VisitAttrs(AttrVisitor*) {}

  IdDoc DefByName(const ObjectRef& obj, const TracedObject<String>& name_prefix) {
    String name = GetUniqueName(name_prefix.Get());
    DocFactory doc_factory = [name](ObjectPath path) {
      IdDoc doc(name);
      doc->paths.push_back(path);
      return doc;
    };

    auto result = obj2info.insert({obj, ObjectInfo{std::move(doc_factory), name}});
    ICHECK(result.second) << "Duplicated object: " << obj;

    IdDoc def_doc(name);
    def_doc->paths.push_back(name_prefix.GetPath());
    return def_doc;
  }

  using DocFactory = std::function<ExprDoc(ObjectPath)>;

  void DefByDoc(const ObjectRef& obj, DocFactory doc_factory) {
    auto result = obj2info.insert({obj, ObjectInfo{std::move(doc_factory), NullOpt}});
    ICHECK(result.second) << "Duplicated object: " << obj;
  }

  void UndefByObject(const ObjectRef& obj) {
    auto it = obj2info.find(obj);
    ICHECK(it != obj2info.end()) << "No such object: " << obj;

    if (it->second.name.defined()) {
      names.erase(it->second.name.value());
    }
    obj2info.erase(it);
  }

  Optional<ExprDoc> GetObjectDoc(const TracedObject<ObjectRef>& obj) const {
    auto it = obj2info.find(obj.Get());
    if (it == obj2info.end()) {
      return NullOpt;
    }
    return it->second.doc_factory(obj.GetPath());
  }

  bool IsObjectDefined(const ObjectRef& obj) { return obj2info.count(obj); }

  String GetObjectName(const ObjectRef& obj) const { return obj2info.at(obj).name.value(); }

  static constexpr const char* _type_key = "script.SymbolTable";
  TVM_DECLARE_FINAL_OBJECT_INFO(SymbolTableNode, Object);

  String GetUniqueName(const String& prefix) {
    String name = prefix;
    for (int i = 1; !names.insert(name).second; ++i) {
      name = prefix + "_" + std::to_string(i);
    }
    return name;
  }

 private:
  struct ObjectInfo {
    DocFactory doc_factory;
    Optional<String> name;
  };
  std::unordered_set<String> names;
  std::unordered_map<ObjectRef, ObjectInfo, ObjectPtrHash, ObjectPtrEqual> obj2info;
};

class SymbolTable : public ObjectRef {
 public:
  SymbolTable();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(SymbolTable, ObjectRef, SymbolTableNode);
};

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_SYMBOL_TABLE_H_
