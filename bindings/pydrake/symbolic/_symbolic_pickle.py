import operator

import pydrake.symbolic as sym
from pydrake.symbolic import (
    Expression,
    ExpressionKind,
    Formula,
    FormulaKind,
    Variable,
    Variables,
)


def _deconstruct_variable(var: sym.Variable) -> list:
    return [var.get_id(), var.get_name(), var.get_type()]


var_map_t = dict[int, sym.Variable]


def _reconstruct_variable(
    var_id: int, var_name: str, var_type: int, var_map: var_map_t
) -> sym.Variable:
    if var_id not in var_map:
        var_map[var_id] = Variable(var_name, var_type)
    return var_map[var_id]


def _deconstruct_formula(f: sym.Formula) -> tuple:
    result = list()
    args = list()
    result.append(f.get_kind())
    match f.get_kind():
        case FormulaKind.Var:
            args = _deconstruct_variable(list(f.GetFreeVariables())[0])
        case (
            sym.FormulaKind.Eq
            | sym.FormulaKind.Neq
            | sym.FormulaKind.Gt
            | sym.FormulaKind.Geq
            | sym.FormulaKind.Lt
            | sym.FormulaKind.Leq
            | sym.FormulaKind.Isnan
        ):
            _, exprs = f.Unapply()
            args = [_deconstruct_expression(expr) for expr in exprs]
        case sym.FormulaKind.And | sym.FormulaKind.Or | sym.FormulaKind.Not:
            _, formulas = f.Unapply()
            args = [_deconstruct_formula(formula) for formula in formulas]
        case sym.FormulaKind.Forall:
            _, forall_args = f.Unapply()
            vars: sym.Variables = forall_args[0]
            formula: sym.Formula = forall_args[1]
            assert isinstance(vars, Variables) and isinstance(formula, Formula)
            args.append([_deconstruct_variable(var) for var in vars])
            args.append(_deconstruct_formula(formula))
        case sym.FormulaKind.PositiveSemidefinite:
            _, matrix_list = f.Unapply()
            matrix = matrix_list[0]
            deconstructed_rows = [
                [_deconstruct_expression(expr) for expr in row]
                for row in matrix
            ]
            args.append(deconstructed_rows)
    result.append(args)
    return tuple(result)


def _reconstruct_formula(
    formula_kind: sym.FormulaKind, args: list, var_map: var_map_t
) -> sym.Formula:
    match formula_kind:
        case sym.FormulaKind.Var:
            assert len(args) == 3
            return Formula(_reconstruct_variable(*args, var_map=var_map))
        case sym.FormulaKind.Isnan:
            assert len(args) == 1
            return sym.isnan(
                _recur_reconstruct_expression(*args[0], var_map=var_map)
            )
        case sym.FormulaKind.Forall:
            assert len(args) == 2
            pickled_vars = args[0]
            vars = sym.Variables()
            for pickled_var in pickled_vars:
                vars.insert(
                    _reconstruct_variable(*pickled_var, var_map=var_map)
                )
            pickled_f = args[1]
            formula: sym.Formula = _reconstruct_formula(*pickled_f, var_map=var_map)
            return sym.forall(vars, formula)
        case sym.FormulaKind.PositiveSemidefinite:
            assert len(args) == 1
            deconstructed_m = args[0]
            rows = len(deconstructed_m)
            cols = len(deconstructed_m[0]) if rows > 0 else 0
            from numpy import empty

            m = empty((rows, cols), dtype=object)
            for i in range(rows):
                for j in range(cols):
                    m[i][j] = _recur_reconstruct_expression(
                        *deconstructed_m[i][j], var_map=var_map
                    )
            return sym.positive_semidefinite(m)

    logical_ops = {
        sym.FormulaKind.And: sym.logical_and,
        sym.FormulaKind.Or: sym.logical_or,
        sym.FormulaKind.Not: sym.logical_not,
    }
    if formula_kind in logical_ops:
        formulas = [
            _reconstruct_formula(*pickled_f, var_map=var_map)
            for pickled_f in args
        ]
        return logical_ops[formula_kind](*formulas)
    binary_ops = {
        sym.FormulaKind.Eq: operator.eq,
        sym.FormulaKind.Neq: operator.ne,
        sym.FormulaKind.Gt: operator.gt,
        sym.FormulaKind.Geq: operator.ge,
        sym.FormulaKind.Lt: operator.lt,
        sym.FormulaKind.Leq: operator.le,
    }
    if formula_kind in binary_ops:
        assert len(args) == 2
        expr1, expr2 = [
            _recur_reconstruct_expression(*arg, var_map=var_map) for arg in args
        ]
        return binary_ops[formula_kind](expr1, expr2)


def _deconstruct_expression(e: Expression) -> tuple:
    result = list()
    args = list()
    result.append(e.get_kind())
    match e.get_kind():
        case sym.ExpressionKind.Constant:
            args.append(e.Evaluate())
        case sym.ExpressionKind.Var:
            args = _deconstruct_variable(list(e.GetVariables())[0])
        case sym.ExpressionKind.NaN:
            pass
        case sym.ExpressionKind.Add | sym.ExpressionKind.Mul:
            _, exprs = e.Unapply()
            # exprs[0] is a number not an expression
            args = [exprs[0]] + [
                _deconstruct_expression(expr) for expr in exprs[1:]
            ]
        case (
            sym.ExpressionKind.Pow
            | sym.ExpressionKind.Atan2
            | sym.ExpressionKind.Div
            | sym.ExpressionKind.Max
            | sym.ExpressionKind.Min
            | sym.ExpressionKind.Abs
            | sym.ExpressionKind.Acos
            | sym.ExpressionKind.Asin
            | sym.ExpressionKind.Atan
            | sym.ExpressionKind.Ceil
            | sym.ExpressionKind.Cos
            | sym.ExpressionKind.Cosh
            | sym.ExpressionKind.Exp
            | sym.ExpressionKind.Floor
            | sym.ExpressionKind.Log
            | sym.ExpressionKind.Sin
            | sym.ExpressionKind.Sinh
            | sym.ExpressionKind.Sqrt
            | sym.ExpressionKind.Tan
            | sym.ExpressionKind.Tanh
        ):
            _, exprs = e.Unapply()
            args = [_deconstruct_expression(expr) for expr in exprs]
        case ExpressionKind.IfThenElse:
            _, symbolics = e.Unapply()
            args = [_deconstruct_formula(symbolics[0])] + [
                _deconstruct_expression(expr) for expr in symbolics[1:]
            ]
        case sym.ExpressionKind.UninterpretedFunction:
            _, func_signature = e.Unapply()
            func_name: str = func_signature[0]
            func_args: list[Expression] = func_signature[1]
            args = [func_name] + [
                [_deconstruct_expression(expr) for expr in func_args]
            ]
    result.append(args)
    return tuple(result)


def _recur_reconstruct_expression(
    expr_kind: sym.ExpressionKind, args: list, var_map: var_map_t
) -> sym.Expression:
    match expr_kind:
        case sym.ExpressionKind.Constant:
            assert len(args) == 1
            return Expression(*args)
        case sym.ExpressionKind.Var:
            assert len(args) == 3
            return Expression(_reconstruct_variable(*args, var_map=var_map))
        case sym.ExpressionKind.NaN:
            assert len(args) == 0
            return Expression(float("nan"))
        case sym.ExpressionKind.IfThenElse:
            assert len(args) == 3
            formula = _reconstruct_formula(*args[0], var_map=var_map)
            expr_then, expr_else = [
                _recur_reconstruct_expression(*arg, var_map=var_map)
                for arg in args[1:]
            ]
            return sym.if_then_else(formula, expr_then, expr_else)
        case sym.ExpressionKind.UninterpretedFunction:
            assert len(args) == 2
            return sym.uninterpreted_function(
                name=args[0],
                arguments=[
                    _recur_reconstruct_expression(*arg, var_map=var_map)
                    for arg in args[1]
                ],
            )

    unary_ops = {
        sym.ExpressionKind.Abs: sym.abs,
        sym.ExpressionKind.Acos: sym.acos,
        sym.ExpressionKind.Asin: sym.asin,
        sym.ExpressionKind.Atan: sym.atan,
        sym.ExpressionKind.Ceil: sym.ceil,
        sym.ExpressionKind.Cos: sym.cos,
        sym.ExpressionKind.Cosh: sym.cosh,
        sym.ExpressionKind.Exp: sym.exp,
        sym.ExpressionKind.Floor: sym.floor,
        sym.ExpressionKind.Log: sym.log,
        sym.ExpressionKind.Sin: sym.sin,
        sym.ExpressionKind.Sinh: sym.sinh,
        sym.ExpressionKind.Sqrt: sym.sqrt,
        sym.ExpressionKind.Tan: sym.tan,
        sym.ExpressionKind.Tanh: sym.tanh,
    }
    if expr_kind in unary_ops:
        assert len(args) == 1
        expr = _recur_reconstruct_expression(*args[0], var_map=var_map)
        return unary_ops[expr_kind](expr)

    binary_ops = {
        sym.ExpressionKind.Pow: sym.pow,
        sym.ExpressionKind.Atan2: sym.atan2,
        sym.ExpressionKind.Div: operator.truediv,
        sym.ExpressionKind.Max: sym.max,
        sym.ExpressionKind.Min: sym.min,
    }
    if expr_kind in binary_ops:
        assert len(args) == 2
        expr1, expr2 = [
            _recur_reconstruct_expression(*arg, var_map=var_map) for arg in args
        ]
        return binary_ops[expr_kind](expr1, expr2)

    add_mul_ops = {
        ExpressionKind.Add: sym._reduce_add,
        ExpressionKind.Mul: sym._reduce_mul,
    }
    if expr_kind in add_mul_ops:
        assert isinstance(args[0], float)
        exprs = [args[0]] + [
            _recur_reconstruct_expression(*arg, var_map=var_map)
            for arg in args[1:]
        ]
        return add_mul_ops[expr_kind](*exprs)


def _reconstruct_expression(
    expr_kind: ExpressionKind, args: list
) -> Expression:
    var_map: var_map_t = dict()
    return _recur_reconstruct_expression(expr_kind, args, var_map)


def _reduce_expression(self):
    return (_reconstruct_expression, _deconstruct_expression(self))
