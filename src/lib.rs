use lazy_static::lazy_static;
use pgrx::prelude::*;
use regex::Regex;
use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::ast::{
    BinaryOperator, Expr, GroupByExpr, Ident, Query, Select, SelectFlavor, SelectItem, SetExpr,
    Statement, TableAlias, TableFactor, TableWithJoins, Value, Values, WildcardAdditionalOptions,
};
use sqlparser::dialect::PostgreSqlDialect;
use sqlparser::parser::Parser;
use std::ffi::CStr;

::pgrx::pg_module_magic!();

// Define the hook signature and variable manually if pg_sys doesn't expose them correctly
#[allow(dead_code)]
type ParserHook = Option<
    unsafe extern "C-unwind" fn(
        str: *const std::os::raw::c_char,
        mode: pg_sys::RawParseMode::Type,
        version: i32,
    ) -> *mut pg_sys::List,
>;

#[allow(dead_code)]
static mut PARSER_HOOK_PTR: *mut ParserHook = std::ptr::null_mut();
#[allow(dead_code)]
static mut PREV_PARSER_HOOK: ParserHook = None;

#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    // Hook installation disabled due to missing parser_hook symbol
}

#[pg_guard]
#[allow(dead_code)]
extern "C-unwind" fn keyset_parser_hook(
    str: *const std::os::raw::c_char,
    raw_parse_mode: pg_sys::RawParseMode::Type,
    version: i32,
) -> *mut pg_sys::List {
    unsafe {
        let query_str = CStr::from_ptr(str).to_string_lossy();

        lazy_static! {
            static ref RE: Regex =
                Regex::new(r"(?i)ORDER\s+BY\s+(.*?)\s+(BEFORE|AFTER)\s+ROW\s*\((.*)\)").unwrap();
        }

        if let Some(_caps) = RE.captures(&query_str) {
            match rewrite_query(&query_str) {
                Ok(new_query) => {
                    let new_c_str = std::ffi::CString::new(new_query).unwrap();

                    // Temporarily disable hook to avoid recursion
                    let saved_hook = *PARSER_HOOK_PTR;
                    *PARSER_HOOK_PTR = None;

                    let result = pg_sys::raw_parser(new_c_str.as_ptr(), raw_parse_mode);

                    // Restore hook
                    *PARSER_HOOK_PTR = saved_hook;

                    return result;
                }
                Err(e) => {
                    pgrx::log!("Keyset rewrite failed: {}", e);
                }
            }
        }

        if let Some(prev) = PREV_PARSER_HOOK {
            prev(str, raw_parse_mode, version)
        } else {
            // Temporarily disable hook to avoid recursion
            let saved_hook = *PARSER_HOOK_PTR;
            *PARSER_HOOK_PTR = None;

            let result = pg_sys::raw_parser(str, raw_parse_mode);

            // Restore hook
            *PARSER_HOOK_PTR = saved_hook;

            result
        }
    }
}

fn rewrite_query(query: &str) -> Result<String, String> {
    // 1. Identify the split point for BEFORE/AFTER
    lazy_static! {
        static ref SPLIT_RE: Regex = Regex::new(r"(?i)\s+(BEFORE|AFTER)\s+ROW\s*\(").unwrap();
    }

    let mat = match SPLIT_RE.find(query) {
        Some(m) => m,
        None => return Ok(query.to_string()),
    };
    let start_idx = mat.start();

    // The part before the keyword is the valid SQL (mostly)
    let valid_part = &query[0..start_idx];

    // The part after is the cursor
    let rest = &query[mat.end()..];
    // We need to find the closing parenthesis for ROW(...)
    // This is naive and assumes no nested parens in values for now
    let end_paren = rest
        .rfind(')')
        .ok_or("Missing closing parenthesis for ROW")?;
    let cursor_values_str = &rest[0..end_paren];
    let tail = &rest[end_paren + 1..];

    // Reconstruct query without the keyset clause
    let reconstructed_query = format!("{}{}", valid_part, tail);

    // Extract mode
    let mode_str = &query[mat.start()..mat.end()]; // e.g. " BEFORE ROW ("
    let is_before = mode_str.to_uppercase().contains("BEFORE");

    // Parse the reconstructed query to get the AST
    let dialect = PostgreSqlDialect {};
    let mut ast = Parser::parse_sql(&dialect, &reconstructed_query).map_err(|e| e.to_string())?;

    if ast.len() != 1 {
        return Err("Expected exactly one statement".to_string());
    }

    let mut query_node = match ast.pop().unwrap() {
        Statement::Query(q) => q,
        _ => return Err("Expected a SELECT query".to_string()),
    };

    // Parse cursor values
    // We wrap them in "SELECT ..." to parse them as expressions
    let cursor_sql = format!("SELECT {}", cursor_values_str);
    let mut cursor_ast = Parser::parse_sql(&dialect, &cursor_sql).map_err(|e| e.to_string())?;
    let cursor_exprs = match cursor_ast.pop().unwrap() {
        Statement::Query(q) => {
            if let SetExpr::Select(s) = *q.body {
                s.projection
            } else {
                return Err("Failed to parse cursor values".to_string());
            }
        }
        _ => return Err("Failed to parse cursor values".to_string()),
    };

    // Extract expressions from projection
    let cursor_args: Vec<Expr> = cursor_exprs
        .into_iter()
        .map(|item| {
            match item {
                sqlparser::ast::SelectItem::UnnamedExpr(e) => e,
                sqlparser::ast::SelectItem::ExprWithAlias { expr, .. } => expr,
                _ => Expr::Value(Value::Null.into()), // Should not happen for ROW values
            }
        })
        .collect();

    // Now apply the logic (similar to keyset_paginate but working on AST directly)
    apply_keyset_logic(&mut query_node, is_before, cursor_args)?;

    Ok(query_node.to_string())
}

fn apply_keyset_logic(
    query_node: &mut Query,
    is_before: bool,
    cursor_args: Vec<Expr>,
) -> Result<(), String> {
    let original_order_by = query_node.order_by.clone();

    let order_by_exprs = if let Some(ob) = &mut query_node.order_by {
        match &mut ob.kind {
            sqlparser::ast::OrderByKind::Expressions(exprs) => exprs,
            _ => return Err("Unsupported ORDER BY kind".to_string()),
        }
    } else {
        return Err("Query must have an ORDER BY clause".to_string());
    };

    if cursor_args.len() > order_by_exprs.len() {
        return Err(format!(
            "Cursor has {} elements but ORDER BY has {}",
            cursor_args.len(),
            order_by_exprs.len()
        ));
    }

    let mut or_chain = None;

    for i in 0..cursor_args.len() {
        let mut and_chain = None;

        for j in 0..i {
            let col_expr = &order_by_exprs[j].expr;
            let param_expr = &cursor_args[j];

            // Equality check: (col = val) OR (col IS NULL AND val IS NULL)
            // Since we don't know if val is null at compile time (it might be a param),
            // we use IS NOT DISTINCT FROM if possible, or explicit check.
            // sqlparser doesn't have IsNotDistinctFrom easily accessible in all versions,
            // so let's construct: (col = val) OR (col IS NULL AND val IS NULL)

            let eq_op = Expr::BinaryOp {
                left: Box::new(col_expr.clone()),
                op: BinaryOperator::Eq,
                right: Box::new(param_expr.clone()),
            };

            let col_is_null = Expr::IsNull(Box::new(col_expr.clone()));
            let param_is_null = Expr::IsNull(Box::new(param_expr.clone()));
            let both_null = Expr::BinaryOp {
                left: Box::new(col_is_null),
                op: BinaryOperator::And,
                right: Box::new(param_is_null),
            };

            let eq_expr = Expr::BinaryOp {
                left: Box::new(eq_op),
                op: BinaryOperator::Or,
                right: Box::new(both_null),
            };

            // Wrap in parentheses to ensure correct precedence when combined with AND
            let eq_expr = Expr::Nested(Box::new(eq_expr));

            and_chain = match and_chain {
                Some(expr) => Some(Expr::BinaryOp {
                    left: Box::new(expr),
                    op: BinaryOperator::And,
                    right: Box::new(eq_expr),
                }),
                None => Some(eq_expr),
            };
        }

        let col_expr = &order_by_exprs[i].expr;
        let param_expr = &cursor_args[i];

        let is_asc = order_by_exprs[i].options.asc.unwrap_or(true);

        // Determine effective NULLS FIRST/LAST
        // Postgres defaults: ASC -> NULLS LAST, DESC -> NULLS FIRST
        let effective_nulls_first = if let Some(nf) = order_by_exprs[i].options.nulls_first {
            nf
        } else {
            !is_asc // DESC implies NULLS FIRST (true), ASC implies NULLS LAST (false)
        };

        // Determine operator based on direction and mode
        // BEFORE reverses the direction logic
        let use_gt = if !is_before { is_asc } else { !is_asc };
        let op = if use_gt {
            BinaryOperator::Gt
        } else {
            BinaryOperator::Lt
        };

        // Construct comparison expression handling NULLs
        // Logic:
        // If NULLS FIRST:
        //   If param IS NULL: col IS NOT NULL (everything else is after null)
        //   If param IS NOT NULL: col OP param (nulls are before, so ignored)
        // If NULLS LAST:
        //   If param IS NULL: FALSE (nothing is after null)
        //   If param IS NOT NULL: (col OP param) OR (col IS NULL) (nulls are after)

        // Since param might be a variable ($1), we need a CASE statement or complex logic.
        // CASE WHEN param IS NULL THEN ... ELSE ... END

        let param_is_null = Expr::IsNull(Box::new(param_expr.clone()));

        let when_null_expr = if effective_nulls_first {
            // NULLS FIRST, param is NULL -> we want everything NOT NULL
            Expr::IsNotNull(Box::new(col_expr.clone()))
        } else {
            // NULLS LAST, param is NULL -> nothing is after
            Expr::Value(Value::Boolean(false).into())
        };

        let base_cmp = Expr::BinaryOp {
            left: Box::new(col_expr.clone()),
            op,
            right: Box::new(param_expr.clone()),
        };

        let else_expr = if effective_nulls_first {
            // NULLS FIRST, param NOT NULL -> just compare (NULLs are before)
            base_cmp
        } else {
            // NULLS LAST, param NOT NULL -> compare OR col IS NULL (NULLs are after)
            Expr::BinaryOp {
                left: Box::new(base_cmp),
                op: BinaryOperator::Or,
                right: Box::new(Expr::IsNull(Box::new(col_expr.clone()))),
            }
        };

        let cmp_expr = Expr::Case {
            operand: None,
            conditions: vec![sqlparser::ast::CaseWhen {
                condition: param_is_null,
                result: when_null_expr,
            }],
            else_result: Some(Box::new(else_expr)),
            case_token: sqlparser::ast::helpers::attached_token::AttachedToken::empty(),
            end_token: sqlparser::ast::helpers::attached_token::AttachedToken::empty(),
        };

        and_chain = match and_chain {
            Some(expr) => Some(Expr::BinaryOp {
                left: Box::new(expr),
                op: BinaryOperator::And,
                right: Box::new(cmp_expr),
            }),
            None => Some(cmp_expr),
        };

        or_chain = match or_chain {
            Some(expr) => Some(Expr::BinaryOp {
                left: Box::new(expr),
                op: BinaryOperator::Or,
                right: Box::new(and_chain.unwrap()),
            }),
            None => and_chain,
        };
    }

    let where_clause = or_chain.unwrap();

    if let SetExpr::Select(ref mut select) = *query_node.body {
        if let Some(existing_where) = &select.selection {
            select.selection = Some(Expr::BinaryOp {
                left: Box::new(existing_where.clone()),
                op: BinaryOperator::And,
                right: Box::new(where_clause),
            });
        } else {
            select.selection = Some(where_clause);
        }
    } else {
        return Err("Query body must be a SELECT".to_string());
    }

    if is_before {
        for item in order_by_exprs.iter_mut() {
            item.options.asc = Some(!item.options.asc.unwrap_or(true));
        }

        // Wrap the query to restore original order
        let mut inner_query = Query {
            with: None,
            body: Box::new(SetExpr::Values(Values {
                explicit_row: false,
                rows: vec![],
                value_keyword: false,
            })),
            order_by: None,
            limit_clause: None,
            fetch: None,
            locks: vec![],
            for_clause: None,
            settings: None,
            format_clause: None,
            pipe_operators: vec![],
        };

        std::mem::swap(query_node, &mut inner_query);

        let subquery = TableFactor::Derived {
            lateral: false,
            subquery: Box::new(inner_query),
            alias: Some(TableAlias {
                name: Ident::new("keyset_sub"),
                columns: vec![],
                explicit: true,
            }),
        };

        let from = vec![TableWithJoins {
            relation: subquery,
            joins: vec![],
        }];

        let select = Select {
            distinct: None,
            top: None,
            projection: vec![SelectItem::Wildcard(WildcardAdditionalOptions {
                opt_exclude: None,
                opt_except: None,
                opt_rename: None,
                opt_replace: None,
                opt_ilike: None,
                wildcard_token: AttachedToken::empty(),
            })],
            into: None,
            from,
            lateral_views: vec![],
            selection: None,
            group_by: GroupByExpr::Expressions(vec![], vec![]),
            cluster_by: vec![],
            distribute_by: vec![],
            sort_by: vec![],
            having: None,
            named_window: vec![],
            qualify: None,
            window_before_qualify: false,
            value_table_mode: None,
            connect_by: None,
            exclude: None,
            flavor: SelectFlavor::Standard,
            prewhere: None,
            select_token: AttachedToken::empty(),
            top_before_distinct: false,
        };

        let outer_query = Query {
            with: None,
            body: Box::new(SetExpr::Select(Box::new(select))),
            order_by: original_order_by,
            limit_clause: None,
            fetch: None,
            locks: vec![],
            for_clause: None,
            settings: None,
            format_clause: None,
            pipe_operators: vec![],
        };

        *query_node = outer_query;
    }

    Ok(())
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use crate::keyset_rewrite_query;
    use crate::rewrite_query;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_rewrite_query_after() {
        let sql = "SELECT * FROM users ORDER BY id ASC AFTER ROW(5)";
        let rewritten = rewrite_query(sql).expect("Rewrite failed");
        assert!(rewritten.contains("id > 5"));
    }

    #[pg_test]
    fn test_rewrite_query_before() {
        let sql = "SELECT * FROM users ORDER BY id ASC BEFORE ROW(5)";
        let rewritten = rewrite_query(sql).expect("Rewrite failed");
        assert!(rewritten.contains("id < 5"));
    }

    #[pg_test]
    fn test_rewrite_query_nulls_last() {
        let sql = "SELECT * FROM users ORDER BY id ASC NULLS LAST AFTER ROW(5)";
        let rewritten = rewrite_query(sql).expect("Rewrite failed");
        // Should contain logic for NULLS LAST
        // (id > 5) OR (id IS NULL)
        assert!(rewritten.contains("id > 5"));
        assert!(rewritten.contains("id IS NULL"));
    }

    #[pg_test]
    fn test_rewrite_query_nulls_first() {
        let sql = "SELECT * FROM users ORDER BY id ASC NULLS FIRST AFTER ROW(5)";
        let rewritten = rewrite_query(sql).expect("Rewrite failed");
        // Should contain logic for NULLS FIRST
        // id > 5 (NULLs are before, so ignored)
        assert!(rewritten.contains("id > 5"));
        // Should NOT contain "id IS NULL" as an OR condition for >
        // But wait, my logic adds IS NOT NULL check if param is null.
        // If param is 5 (not null), then for NULLS FIRST, we just want > 5.
        // So "id IS NULL" should not be present in the OR branch.
    }

    #[pg_test]
    fn test_rewrite_query_limit() {
        let sql = "SELECT * FROM users ORDER BY id ASC AFTER ROW(5) LIMIT 10";
        let rewritten = rewrite_query(sql).expect("Rewrite failed");
        assert!(rewritten.contains("LIMIT 10"));
        assert!(rewritten.contains("id > 5"));
    }

    #[pg_test]
    fn test_rewrite_query_fetch() {
        let sql = "SELECT * FROM users ORDER BY id ASC AFTER ROW(5) FETCH FIRST 10 ROWS ONLY";
        let rewritten = rewrite_query(sql).expect("Rewrite failed");
        assert!(rewritten.contains("FETCH FIRST 10 ROWS ONLY"));
        assert!(rewritten.contains("id > 5"));
    }

    #[pg_test]
    fn test_rewrite_query_fetch_with_ties() {
        let sql = "SELECT * FROM users ORDER BY id ASC AFTER ROW(5) FETCH FIRST 10 ROWS WITH TIES";
        let rewritten = rewrite_query(sql).expect("Rewrite failed");
        assert!(rewritten.contains("FETCH FIRST 10 ROWS WITH TIES"));
        assert!(rewritten.contains("id > 5"));
    }

    #[pg_test]
    fn test_paging_integration() {
        Spi::run("CREATE TABLE test_paging (id int, val int);").expect("create table failed");
        Spi::run(
            "INSERT INTO test_paging (id, val) VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50);",
        )
        .expect("insert failed");

        // Test Forward Paging (AFTER)
        // Page 1: Get first 2
        let result = Spi::connect(|client| {
            let table = client
                .select(
                    "SELECT id FROM test_paging ORDER BY id ASC LIMIT 2",
                    None,
                    &[],
                )
                .expect("select failed");
            let ids: Vec<i32> = table
                .map(|row| row.get::<i32>(1).expect("no id").expect("null id"))
                .collect();
            Ok::<_, pgrx::spi::SpiError>(Some(ids))
        })
        .expect("spi failed");
        assert_eq!(result, Some(vec![1, 2]));

        // Page 2: Get next 2 after id=2
        // Note: We need to pass the values for ROW(). In SQL we can use literals.
        let result = Spi::connect(|client| {
            let sql = keyset_rewrite_query(
                "SELECT id FROM test_paging ORDER BY id ASC AFTER ROW(2) LIMIT 2",
            )
            .expect("rewrite failed");
            let table = client.select(&sql, None, &[]).expect("select failed");
            let ids: Vec<i32> = table
                .map(|row| row.get::<i32>(1).expect("no id").expect("null id"))
                .collect();
            Ok::<_, pgrx::spi::SpiError>(Some(ids))
        })
        .expect("spi failed");
        assert_eq!(result, Some(vec![3, 4]));

        // Test Backward Paging (BEFORE)
        // Suppose we are at page 3 (id 5), want previous page (ids 3, 4)
        // BEFORE ROW(5) ORDER BY id ASC LIMIT 2
        // This should return 3, 4.
        // Wait, standard keyset pagination "BEFORE" usually requires inverting the order to get the "last 2 before X", then re-sorting.
        // My implementation in `rewrite_query` handles the direction inversion for the comparison (e.g. id < 5).
        // But does it handle the `ORDER BY` inversion?
        // Let's check `rewrite_query` implementation.

        /*
        if is_before {
            for item in order_by_exprs.iter_mut() {
                item.options.asc = Some(!item.options.asc.unwrap_or(true));
            }
        }
        */
        // Yes, it inverts the sort order in the AST.
        // So "ORDER BY id ASC BEFORE ROW(5) LIMIT 2" becomes "ORDER BY id DESC ... WHERE id < 5 LIMIT 2".
        // This will return 4, 3.
        // The client usually needs to reverse this list to display 3, 4.

        let result = Spi::connect(|client| {
            let sql = keyset_rewrite_query(
                "SELECT id FROM test_paging ORDER BY id ASC BEFORE ROW(5) LIMIT 2",
            )
            .expect("rewrite failed");
            let table = client.select(&sql, None, &[]).expect("select failed");
            let ids: Vec<i32> = table
                .map(|row| row.get::<i32>(1).expect("no id").expect("null id"))
                .collect();
            Ok::<_, pgrx::spi::SpiError>(Some(ids))
        })
        .expect("spi failed");
        assert_eq!(result, Some(vec![3, 4])); // Expect correct order due to wrapping
    }

    #[pg_test]
    fn test_paging_integration_multi_col() {
        Spi::run("CREATE TABLE test_paging_multi (a int, b int);").expect("create table failed");
        // a ASC, b DESC
        // (1, 10), (1, 5), (2, 20), (2, 10)
        Spi::run("INSERT INTO test_paging_multi (a, b) VALUES (1, 10), (1, 5), (2, 20), (2, 10);")
            .expect("insert failed");

        // Sort: a ASC, b DESC
        // Expected order:
        // 1, 10
        // 1, 5
        // 2, 20
        // 2, 10

        // Get first 2
        let result = Spi::connect(|client| {
            let table = client
                .select(
                    "SELECT a, b FROM test_paging_multi ORDER BY a ASC, b DESC LIMIT 2",
                    None,
                    &[],
                )
                .expect("select failed");
            let rows: Vec<(i32, i32)> = table
                .map(|row| {
                    (
                        row.get::<i32>(1).expect("no a").expect("null a"),
                        row.get::<i32>(2).expect("no b").expect("null b"),
                    )
                })
                .collect();
            Ok::<_, pgrx::spi::SpiError>(Some(rows))
        })
        .expect("spi failed");
        assert_eq!(result, Some(vec![(1, 10), (1, 5)]));

        // Get next 2 after (1, 5)
        let result = Spi::connect(|client| {
            let sql = keyset_rewrite_query(
                "SELECT a, b FROM test_paging_multi ORDER BY a ASC, b DESC AFTER ROW(1, 5) LIMIT 2",
            )
            .expect("rewrite failed");
            let table = client.select(&sql, None, &[]).expect("select failed");
            let rows: Vec<(i32, i32)> = table
                .map(|row| {
                    (
                        row.get::<i32>(1).expect("no a").expect("null a"),
                        row.get::<i32>(2).expect("no b").expect("null b"),
                    )
                })
                .collect();
            Ok::<_, pgrx::spi::SpiError>(Some(rows))
        })
        .expect("spi failed");
        assert_eq!(result, Some(vec![(2, 20), (2, 10)]));

        // Test Backward Paging (BEFORE)
        // Suppose we are at (2, 20), want previous page (1, 10), (1, 5)
        // BEFORE ROW(2, 20) ORDER BY a ASC, b DESC LIMIT 2
        // Should return (1, 10), (1, 5) in that order.

        let result = Spi::connect(|client| {
            let sql = keyset_rewrite_query("SELECT a, b FROM test_paging_multi ORDER BY a ASC, b DESC BEFORE ROW(2, 20) LIMIT 2").expect("rewrite failed");
            let table = client.select(&sql, None, &[]).expect("select failed");
            let rows: Vec<(i32, i32)> = table.map(|row| (row.get::<i32>(1).expect("no a").expect("null a"), row.get::<i32>(2).expect("no b").expect("null b"))).collect();
            Ok::<_, pgrx::spi::SpiError>(Some(rows))
        }).expect("spi failed");
        assert_eq!(result, Some(vec![(1, 10), (1, 5)]));
    }

    #[pg_test]
    fn test_paging_integration_types() {
        Spi::run("CREATE TABLE test_paging_types (t text, f float8);")
            .expect("create table failed");
        Spi::run("INSERT INTO test_paging_types (t, f) VALUES ('a', 1.1), ('b', 2.2), ('c', 3.3);")
            .expect("insert failed");

        // Page 1
        let result = Spi::connect(|client| {
            let table = client
                .select(
                    "SELECT t, f FROM test_paging_types ORDER BY t ASC LIMIT 2",
                    None,
                    &[],
                )
                .expect("select failed");
            let rows: Vec<(String, f64)> = table
                .map(|row| {
                    (
                        row.get::<String>(1).expect("no t").expect("null t"),
                        row.get::<f64>(2).expect("no f").expect("null f"),
                    )
                })
                .collect();
            Ok::<_, pgrx::spi::SpiError>(Some(rows))
        })
        .expect("spi failed");
        assert_eq!(
            result,
            Some(vec![("a".to_string(), 1.1), ("b".to_string(), 2.2)])
        );

        // Page 2
        let result = Spi::connect(|client| {
            // Note: For text, we need to quote the value in the ROW() clause.
            let sql = keyset_rewrite_query(
                "SELECT t, f FROM test_paging_types ORDER BY t ASC AFTER ROW('b') LIMIT 2",
            )
            .expect("rewrite failed");
            let table = client.select(&sql, None, &[]).expect("select failed");
            let rows: Vec<(String, f64)> = table
                .map(|row| {
                    (
                        row.get::<String>(1).expect("no t").expect("null t"),
                        row.get::<f64>(2).expect("no f").expect("null f"),
                    )
                })
                .collect();
            Ok::<_, pgrx::spi::SpiError>(Some(rows))
        })
        .expect("spi failed");
        assert_eq!(result, Some(vec![("c".to_string(), 3.3)]));
    }

    #[pg_test]
    fn test_keyset_rewrite_query_exposed() {
        let result = Spi::get_one::<String>(
            "SELECT keyset_rewrite_query('SELECT * FROM t ORDER BY a AFTER ROW(1)')",
        );
        assert!(result
            .expect("SPI failed")
            .expect("null result")
            .contains("a > 1"));
    }

    #[pg_test]
    fn test_paging_integration_multi_col_nulls() {
        Spi::run("CREATE TABLE test_paging_nulls (a int, b int);").expect("create table failed");
        Spi::run(
            "INSERT INTO test_paging_nulls (a, b) VALUES (1, NULL), (1, 10), (NULL, 5), (2, 20);",
        )
        .expect("insert failed");

        // Sort: a ASC NULLS FIRST, b DESC NULLS LAST
        // Expected order:
        // 1. (NULL, 5)
        // 2. (1, 10)
        // 3. (1, NULL)
        // 4. (2, 20)

        // Page 1: First 2 rows
        let result = Spi::connect(|client| {
            let table = client.select("SELECT a, b FROM test_paging_nulls ORDER BY a ASC NULLS FIRST, b DESC NULLS LAST LIMIT 2", None, &[]).expect("select failed");
            let rows: Vec<(Option<i32>, Option<i32>)> = table.map(|row| (row.get::<i32>(1).expect("no a"), row.get::<i32>(2).expect("no b"))).collect();
            Ok::<_, pgrx::spi::SpiError>(Some(rows))
        }).expect("spi failed");
        assert_eq!(result, Some(vec![(None, Some(5)), (Some(1), Some(10))]));

        // Page 2: Next 2 rows after (1, 10)
        // Note: We need to handle passing NULLs to ROW().
        // In SQL: AFTER ROW(1, 10)
        let result = Spi::connect(|client| {
            let sql = keyset_rewrite_query("SELECT a, b FROM test_paging_nulls ORDER BY a ASC NULLS FIRST, b DESC NULLS LAST AFTER ROW(1, 10) LIMIT 2").expect("rewrite failed");
            let table = client.select(&sql, None, &[]).expect("select failed");
            let rows: Vec<(Option<i32>, Option<i32>)> = table.map(|row| (row.get::<i32>(1).expect("no a"), row.get::<i32>(2).expect("no b"))).collect();
            Ok::<_, pgrx::spi::SpiError>(Some(rows))
        }).expect("spi failed");
        assert_eq!(result, Some(vec![(Some(1), None), (Some(2), Some(20))]));

        // Page 3: Next rows after (1, NULL)
        // In SQL: AFTER ROW(1, NULL)
        let result = Spi::connect(|client| {
            let sql = keyset_rewrite_query("SELECT a, b FROM test_paging_nulls ORDER BY a ASC NULLS FIRST, b DESC NULLS LAST AFTER ROW(1, NULL) LIMIT 2").expect("rewrite failed");
            let table = client.select(&sql, None, &[]).expect("select failed");
            let rows: Vec<(Option<i32>, Option<i32>)> = table.map(|row| (row.get::<i32>(1).expect("no a"), row.get::<i32>(2).expect("no b"))).collect();
            Ok::<_, pgrx::spi::SpiError>(Some(rows))
        }).expect("spi failed");
        assert_eq!(result, Some(vec![(Some(2), Some(20))]));
    }

    #[pg_test]
    fn test_rewrite_query_no_keyword() {
        let sql = "SELECT * FROM users ORDER BY id ASC";
        let rewritten = rewrite_query(sql).expect("Rewrite failed");
        assert_eq!(rewritten, sql);
    }

    #[pg_test]
    fn test_paging_integration_fetch() {
        Spi::run("CREATE TABLE test_paging_fetch (id int);").expect("create table failed");
        Spi::run("INSERT INTO test_paging_fetch (id) VALUES (1), (2), (3), (4), (5);")
            .expect("insert failed");

        // Page 1: Fetch first 2
        let result = Spi::connect(|client| {
            let table = client
                .select(
                    "SELECT id FROM test_paging_fetch ORDER BY id ASC FETCH FIRST 2 ROWS ONLY",
                    None,
                    &[],
                )
                .expect("select failed");
            let ids: Vec<i32> = table
                .map(|row| row.get::<i32>(1).expect("no id").expect("null id"))
                .collect();
            Ok::<_, pgrx::spi::SpiError>(Some(ids))
        })
        .expect("spi failed");
        assert_eq!(result, Some(vec![1, 2]));

        // Page 2: Fetch next 2 after id=2
        let result = Spi::connect(|client| {
            let sql = keyset_rewrite_query("SELECT id FROM test_paging_fetch ORDER BY id ASC AFTER ROW(2) FETCH FIRST 2 ROWS ONLY").expect("rewrite failed");
            let table = client.select(&sql, None, &[]).expect("select failed");
            let ids: Vec<i32> = table.map(|row| row.get::<i32>(1).expect("no id").expect("null id")).collect();
            Ok::<_, pgrx::spi::SpiError>(Some(ids))
        }).expect("spi failed");
        assert_eq!(result, Some(vec![3, 4]));
    }

    #[pg_test]
    fn test_paging_integration_fetch_with_ties() {
        Spi::run("CREATE TABLE test_paging_ties (val int);").expect("create table failed");
        // Insert duplicates: 1, 1, 2, 2, 2, 3
        Spi::run("INSERT INTO test_paging_ties (val) VALUES (1), (1), (2), (2), (2), (3);")
            .expect("insert failed");

        // Page 1: Fetch first 2 rows WITH TIES
        // Should return both 1s.
        let result = Spi::connect(|client| {
            let table = client.select("SELECT val FROM test_paging_ties ORDER BY val ASC FETCH FIRST 2 ROWS WITH TIES", None, &[]).expect("select failed");
            let vals: Vec<i32> = table.map(|row| row.get::<i32>(1).expect("no val").expect("null val")).collect();
            Ok::<_, pgrx::spi::SpiError>(Some(vals))
        }).expect("spi failed");
        assert_eq!(result, Some(vec![1, 1]));

        // Page 2: Fetch first 3 rows WITH TIES (starting from scratch to demonstrate)
        // Should return 1, 1, 2, 2, 2 (5 rows) because 3rd row is 2, and there are ties.
        let result = Spi::connect(|client| {
            let table = client.select("SELECT val FROM test_paging_ties ORDER BY val ASC FETCH FIRST 3 ROWS WITH TIES", None, &[]).expect("select failed");
            let vals: Vec<i32> = table.map(|row| row.get::<i32>(1).expect("no val").expect("null val")).collect();
            Ok::<_, pgrx::spi::SpiError>(Some(vals))
        }).expect("spi failed");
        assert_eq!(result, Some(vec![1, 1, 2, 2, 2]));

        // Paging with AFTER ROW
        // Suppose we fetched the first page (1, 1). Cursor is (1).
        // Next page: AFTER ROW(1) FETCH FIRST 2 ROWS WITH TIES
        // Should return 2, 2, 2.
        let result = Spi::connect(|client| {
            let sql = keyset_rewrite_query("SELECT val FROM test_paging_ties ORDER BY val ASC AFTER ROW(1) FETCH FIRST 2 ROWS WITH TIES").expect("rewrite failed");
            let table = client.select(&sql, None, &[]).expect("select failed");
            let vals: Vec<i32> = table.map(|row| row.get::<i32>(1).expect("no val").expect("null val")).collect();
            Ok::<_, pgrx::spi::SpiError>(Some(vals))
        }).expect("spi failed");
        assert_eq!(result, Some(vec![2, 2, 2]));
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}

#[pg_extern]
fn keyset_rewrite_query(query: &str) -> Result<String, String> {
    rewrite_query(query)
}
