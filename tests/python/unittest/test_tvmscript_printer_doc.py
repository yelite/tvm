# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
In this test file, we want to make sure the Python code can construct
Doc objects, then access and modify their attributes correctly.
"""

import pytest

from tvm.script.printer.doc import (
    LiteralDoc,
    IdDoc,
    AttrAccessDoc,
    IndexDoc,
    CallDoc,
    OperationKind,
    OperationDoc,
    LambdaDoc,
    TupleDoc,
    ListDoc,
    DictDoc,
    SliceDoc,
    StmtBlockDoc,
    AssignDoc,
    IfDoc,
    WhileDoc,
    ForDoc,
    ScopeDoc,
    ExprStmtDoc,
    FunctionDoc,
    ClassDoc,
)


@pytest.mark.parametrize(
    "value",
    [None, "test", 0, 1, -2, 0.0, 1.5, -1.3, True, False],
)
def test_literal_doc_construction(value):
    doc = LiteralDoc(value)

    if isinstance(value, float):
        # FloatImm cannot be compared with Python's float directly
        assert float(doc.value) == pytest.approx(value)
    else:
        assert doc.value == value


def test_id_doc():
    doc = IdDoc("name")

    assert doc.name == "name"


def test_attr_access_doc():
    target = IdDoc("x")

    doc = AttrAccessDoc(target, "attribute")

    assert doc.value == target
    assert doc.attr == "attribute"


@pytest.mark.parametrize(
    "indices",
    [
        [],
        [LiteralDoc(1)],
        [LiteralDoc(2), IdDoc("x")],
        [SliceDoc(LiteralDoc(1), LiteralDoc(2))],
        [SliceDoc(LiteralDoc(1)), IdDoc("y")],
    ],
)
def test_index_doc(indices):
    target = IdDoc("x")

    doc = IndexDoc(target, indices)

    assert doc.value == target
    assert list(doc.indices) == indices


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ([], {}),
        ([LiteralDoc("arg")], {}),
        ([LiteralDoc("arg"), IdDoc("x")], {}),
        ([], {"x": LiteralDoc("x")}),
        ([], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
        ([LiteralDoc("arg")], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
        ([LiteralDoc("arg"), IdDoc("x")], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
    ],
)
def test_call_doc(args, kwargs):
    target = IdDoc("x")

    doc = CallDoc(target, *args, **kwargs)

    assert doc.callee == target
    assert list(doc.args) == args
    assert dict(zip(doc.kwargs_keys, doc.kwargs_values)) == kwargs


@pytest.mark.parametrize(
    "operands",
    [
        [],
        [LiteralDoc(1)],
        [LiteralDoc(2), IdDoc("x")],
        [LiteralDoc(2), IdDoc("x"), LiteralDoc("y")],
    ],
)
def test_operation_doc(operands):
    # Here we just test the contructor and attr visitor of OperationDoc
    # so the choice of OperationKind doesn't matter
    operator = OperationKind.Add

    doc = OperationDoc(OperationKind.Add, operands)

    assert doc.kind == operator
    assert list(doc.operands) == operands


@pytest.mark.parametrize(
    "args",
    [
        [],
        [IdDoc("x")],
        [IdDoc("x"), IdDoc("y")],
    ],
)
def test_lambda_doc(args):
    body = LiteralDoc(1)

    doc = LambdaDoc(args, body)

    assert doc.body == body
    assert list(doc.args) == args


@pytest.mark.parametrize(
    "elements",
    [
        [],
        [IdDoc("x")],
        [IdDoc("x"), IdDoc("y")],
    ],
)
def test_tuple_doc(elements):
    doc = TupleDoc(elements)

    assert list(doc.elements) == elements


@pytest.mark.parametrize(
    "elements",
    [
        [],
        [IdDoc("x")],
        [IdDoc("x"), IdDoc("y")],
    ],
)
def test_list_doc(elements):
    doc = ListDoc(elements)

    assert list(doc.elements) == elements


@pytest.mark.parametrize(
    "content",
    [
        {},
        {LiteralDoc("k"): IdDoc("v")},
        {LiteralDoc("k"): IdDoc("v"), LiteralDoc("k2"): IdDoc("v2")},
    ],
)
def test_dict_doc(content):
    doc = DictDoc(content)

    assert dict(zip(doc.keys, doc.values)) == content


@pytest.mark.parametrize(
    "start,stop",
    [
        (LiteralDoc(1), LiteralDoc(2)),
        (LiteralDoc(1), None),
        (None, LiteralDoc(2)),
        (None, None),
    ],
)
def test_slice_doc(start, stop):
    doc = SliceDoc(start, stop)

    assert doc.start == start
    assert doc.stop == stop


def test_expr_doc_attr_access():
    target = IdDoc("x")
    attr = "test"

    doc = target.attr_access(attr)

    assert doc.value == target
    assert doc.attr == attr


@pytest.mark.parametrize(
    "indices",
    [
        [],
        [LiteralDoc(1)],
        [LiteralDoc(2), IdDoc("x")],
        [SliceDoc(LiteralDoc(1), LiteralDoc(2))],
        [SliceDoc(LiteralDoc(1)), IdDoc("y")],
    ],
)
def test_expr_doc_index_access(indices):
    target = IdDoc("x")

    doc = target.index_access(indices)

    assert doc.value == target
    assert list(doc.indices) == indices


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ([], {}),
        ([LiteralDoc("arg")], {}),
        ([LiteralDoc("arg"), IdDoc("x")], {}),
        ([], {"x": LiteralDoc("x")}),
        ([], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
        ([LiteralDoc("arg")], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
        ([LiteralDoc("arg"), IdDoc("x")], {"x": LiteralDoc("x"), "y": LiteralDoc("y")}),
    ],
)
def test_expr_doc_call_with(args, kwargs):
    target = IdDoc("x")

    doc = target.call_with(*args, **kwargs)

    assert doc.callee == target
    assert list(doc.args) == args
    assert dict(zip(doc.kwargs_keys, doc.kwargs_values)) == kwargs


@pytest.mark.parametrize(
    "stmts",
    [
        [],
        [ExprStmtDoc(IdDoc("x"))],
        [ExprStmtDoc(IdDoc("x")), ExprStmtDoc(IdDoc("y"))],
    ],
)
def test_stmt_block_doc(stmts):
    doc = StmtBlockDoc(stmts)

    assert list(doc.stmts) == stmts


@pytest.mark.parametrize(
    "lhs, rhs, annotation",
    [
        (IdDoc("x"), IdDoc("y"), None),
        (IdDoc("x"), None, IdDoc("int")),
        (IdDoc("x"), IdDoc("y"), IdDoc("int")),
    ],
)
def test_assign_doc(lhs, rhs, annotation):
    doc = AssignDoc(lhs, rhs, annotation)

    assert doc.lhs == lhs
    assert doc.rhs == rhs
    assert doc.annotation == annotation


@pytest.mark.parametrize(
    "lhs, rhs, annotation",
    [
        (IdDoc("x"), None, None),
        (TupleDoc([IdDoc("x"), IdDoc("y")]), None, IdDoc("int")),
        (TupleDoc([IdDoc("x"), IdDoc("y")]), IdDoc("u"), IdDoc("int")),
    ],
)
def test_invalid_assign_doc(lhs, rhs, annotation):
    with pytest.raises(ValueError) as e:
        AssignDoc(lhs, rhs, annotation)
    assert "AssignDoc" in str(e.value)


@pytest.mark.parametrize(
    "else_branch",
    [
        [],
        [ExprStmtDoc(IdDoc("x"))],
        [ExprStmtDoc(IdDoc("x")), ExprStmtDoc(IdDoc("y"))],
    ],
)
@pytest.mark.parametrize(
    "then_branch",
    [
        [],
        [ExprStmtDoc(IdDoc("x"))],
        [ExprStmtDoc(IdDoc("x")), ExprStmtDoc(IdDoc("y"))],
    ],
)
def test_if_doc(then_branch, else_branch):
    predicate = IdDoc("x")

    if not then_branch and not else_branch:
        with pytest.raises(ValueError) as e:
            IfDoc(predicate, then_branch, else_branch)
        assert "IfDoc" in str(e.value)
        return
    else:
        doc = IfDoc(predicate, then_branch, else_branch)

    assert doc.predicate == predicate
    assert list(doc.then_branch) == then_branch
    assert list(doc.else_branch) == else_branch


@pytest.mark.parametrize(
    "body",
    [
        [],
        [ExprStmtDoc(IdDoc("x"))],
        [ExprStmtDoc(IdDoc("x")), ExprStmtDoc(IdDoc("y"))],
    ],
)
def test_while_doc(body):
    predicate = IdDoc("x")

    doc = WhileDoc(predicate, body)

    assert doc.predicate == predicate
    assert list(doc.body) == body


@pytest.mark.parametrize(
    "body",
    [
        [],
        [ExprStmtDoc(IdDoc("x"))],
        [ExprStmtDoc(IdDoc("x")), ExprStmtDoc(IdDoc("y"))],
    ],
)
def test_for_doc(body):
    lhs = IdDoc("x")
    rhs = IdDoc("y")

    doc = ForDoc(lhs, rhs, body)

    assert doc.lhs == lhs
    assert doc.rhs == rhs
    assert list(doc.body) == body


@pytest.mark.parametrize(
    "lhs",
    [
        None,
        IdDoc("x"),
    ],
)
@pytest.mark.parametrize(
    "body",
    [
        [],
        [ExprStmtDoc(IdDoc("x"))],
        [ExprStmtDoc(IdDoc("x")), ExprStmtDoc(IdDoc("y"))],
    ],
)
def test_scope_doc(lhs, body):
    rhs = IdDoc("y")

    doc = ScopeDoc(lhs, rhs, body)

    assert doc.lhs == lhs
    assert doc.rhs == rhs
    assert list(doc.body) == body


def test_expr_stmt_doc():
    expr = IdDoc("x")

    doc = ExprStmtDoc(expr)

    assert doc.expr == expr


@pytest.mark.parametrize(
    "args",
    [
        [],
        [AssignDoc(IdDoc("x"), None, IdDoc("int"))],
        [
            AssignDoc(IdDoc("x"), None, IdDoc("int")),
            AssignDoc(IdDoc("y"), LiteralDoc(1), IdDoc("int")),
        ],
    ],
)
@pytest.mark.parametrize(
    "decorators",
    [
        [],
        [IdDoc("test")],
        [IdDoc("test"), IdDoc("test2")],
    ],
)
@pytest.mark.parametrize(
    "body",
    [
        [],
        [ExprStmtDoc(IdDoc("x"))],
        [ExprStmtDoc(IdDoc("x")), ExprStmtDoc(IdDoc("y"))],
    ],
)
def test_function_doc(args, decorators, body):
    name = IdDoc("name")
    return_type = LiteralDoc(None)

    doc = FunctionDoc(name, args, decorators, return_type, body)

    assert doc.name == name
    assert list(doc.args) == args
    assert list(doc.decorators) == decorators
    assert doc.return_type == return_type
    assert list(doc.body) == body


@pytest.mark.parametrize(
    "decorators",
    [
        [],
        [IdDoc("test")],
        [IdDoc("test"), IdDoc("test2")],
    ],
)
@pytest.mark.parametrize(
    "body",
    [
        [],
        [ExprStmtDoc(IdDoc("x"))],
        [ExprStmtDoc(IdDoc("x")), ExprStmtDoc(IdDoc("y"))],
    ],
)
def test_class_doc(decorators, body):
    name = IdDoc("name")

    doc = ClassDoc(name, decorators, body)

    assert doc.name == name
    assert list(doc.decorators) == decorators
    assert list(doc.body) == body


def test_stmt_doc_comment():
    doc = ExprStmtDoc(IdDoc("x"))
    assert doc.comment is None

    comment = "test comment"
    doc.comment = comment
    assert doc.comment == comment
