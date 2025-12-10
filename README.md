# pg_keyset

A PostgreSQL extension for keyset pagination.

## Usage

This extension allows you to use keyset pagination using standard SQL syntax extended with `BEFORE` and `AFTER` keywords in the `ORDER BY` clause.

### Syntax

```sql
SELECT ...
ORDER BY ...
[BEFORE | AFTER] ROW(value1, value2, ...)
```

### Example

Get the next page (after the last seen row):

```sql
SELECT id, name
FROM users
ORDER BY id ASC
AFTER ROW(5);
```

This is automatically rewritten to:

```sql
SELECT id, name
FROM users
WHERE id > 5
ORDER BY id ASC;
```

Get the previous page (before the first seen row):

```sql
SELECT id, name
FROM users
ORDER BY id ASC
BEFORE ROW(5);
```

This is rewritten to fetch rows where `id < 5`, with the sort order temporarily reversed to get the closest rows.

### Function API

You can also use the `keyset_paginate` function directly if you prefer not to use the syntax extension or need dynamic query generation.

```sql
SELECT * FROM keyset_paginate(
    'SELECT id, name FROM users ORDER BY id ASC LIMIT 10',
    'AFTER',
    ROW(5)
);
```

### Handling Non-Unique Sort Keys

Keyset pagination requires a deterministic sort order to function correctly. If your `ORDER BY` columns are not unique, standard keyset pagination may skip rows.

To handle non-unique sort keys, you must ensure that you fetch all rows with the same sort key values in a single batch. You can achieve this using the `FETCH FIRST N ROWS WITH TIES` clause.

```sql
SELECT * FROM items
ORDER BY category ASC
AFTER ROW('books')
FETCH FIRST 10 ROWS WITH TIES;
```

Using `WITH TIES` ensures that if the 10th row has duplicates (e.g., multiple items in the 'books' category), all of them are returned. This guarantees that the cursor for the next page will be at the end of the group of duplicates, preventing skipped rows.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
