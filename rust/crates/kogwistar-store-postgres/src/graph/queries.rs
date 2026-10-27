use super::super::*;

pub(crate) async fn graph_projection_records<C>(
    client: &C,
    tables: &Tables,
    read: GraphProjectionRead,
    after_id: Option<String>,
    keyset_page: bool,
) -> PostgresStoreResult<Vec<GraphRecord>>
where
    C: GenericClient + Sync,
{
    graph_projection_records_with_lock(client, tables, read, after_id, keyset_page, false).await
}

pub(crate) async fn graph_projection_records_with_lock<C>(
    client: &C,
    tables: &Tables,
    read: GraphProjectionRead,
    after_id: Option<String>,
    keyset_page: bool,
    for_update: bool,
) -> PostgresStoreResult<Vec<GraphRecord>>
where
    C: GenericClient + Sync,
{
    let table = graph_table(tables, &read.table)?;
    let (filter_sql, mut values) = graph_filter_sql(&read.scope, &read.metadata, 1)?;
    let mut clauses = vec![filter_sql];
    if let Some(ids) = read.ids {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let start = values.len() + 1;
        clauses.push(format!(
            "id IN ({})",
            (0..ids.len())
                .map(|index| format!("${}", start + index))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        values.extend(ids);
    }
    if let Some(after_id) = after_id {
        let cursor_index = values.len() + 1;
        clauses.push(format!(
            "id COLLATE \"C\" > ${cursor_index}::TEXT COLLATE \"C\""
        ));
        values.push(after_id);
    }
    let limit = i64::try_from(read.limit).unwrap_or(i64::MAX).to_string();
    values.push(limit);
    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = values
        .iter()
        .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();
    let ordering = if keyset_page {
        "id COLLATE \"C\" ASC"
    } else {
        "id ASC"
    };
    let locking = if for_update { " FOR UPDATE" } else { "" };
    let rows = client
        .query(
            &format!(
                "SELECT id, document, metadata::TEXT, embedding::TEXT FROM {table} WHERE {} ORDER BY {ordering} LIMIT ${}::TEXT::BIGINT{locking}",
                clauses.join(" AND "),
                values.len()
            ),
            &params,
        )
        .await
        .map_err(backend)?;
    rows.iter().map(graph_record_from_row).collect()
}

pub(crate) async fn graph_projection_vector_query<C>(
    client: &C,
    tables: &Tables,
    query: GraphProjectionVectorQuery,
    metadata_not_equals: &BTreeMap<String, Value>,
) -> PostgresStoreResult<Vec<VectorMatch>>
where
    C: GenericClient + Sync,
{
    if query.query.embedding.len() != query.embedding_dim {
        return Err(StoreError::VectorDimensionMismatch {
            expected: query.embedding_dim,
            actual: query.query.embedding.len(),
        }
        .into());
    }
    finite_vector(&query.query.embedding)?;
    let table = graph_table(tables, &query.table)?;
    let (filter_sql, mut values) = graph_filter_sql_with_exclusions(
        &query.scope,
        &query.query.metadata,
        metadata_not_equals,
        1,
    )?;
    let vector_sql = format!(
        "{}::vector",
        sql_text_literal(&vector_literal(&query.query.embedding))
    );
    let limit_index = values.len() + 1;
    values.push(
        i64::try_from(query.query.limit)
            .unwrap_or(i64::MAX)
            .to_string(),
    );
    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = values
        .iter()
        .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();
    let operator = match query.query.metric {
        kogwistar_store::DistanceMetric::Cosine => "<=>",
        kogwistar_store::DistanceMetric::L2 => "<->",
        kogwistar_store::DistanceMetric::InnerProduct => "<#>",
    };
    let rows = client
        .query(
            &format!(
                "SELECT id, document, metadata::TEXT, embedding::TEXT, embedding {operator} {vector_sql} AS distance \
                 FROM {table} WHERE embedding IS NOT NULL AND {filter_sql} \
                 ORDER BY distance ASC, id ASC LIMIT ${limit_index}::TEXT::BIGINT"
            ),
            &params,
        )
        .await
        .map_err(backend)?;
    rows.iter()
        .map(|row| {
            Ok(VectorMatch {
                record: graph_record_from_row(row)?,
                distance: row.try_get(4).map_err(backend)?,
            })
        })
        .collect()
}

pub(crate) async fn graph_projection_vector_query_batch<C>(
    client: &C,
    tables: &Tables,
    queries: Vec<GraphProjectionVectorQuery>,
    metadata_not_equals: &BTreeMap<String, Value>,
) -> PostgresStoreResult<Vec<Vec<VectorMatch>>>
where
    C: GenericClient + Sync,
{
    let Some(first) = queries.first() else {
        return Ok(Vec::new());
    };
    require_namespace(&first.scope.namespace)?;
    let table = graph_table(tables, &first.table)?;
    let metric = first.query.metric;
    let mut query_vectors = Vec::with_capacity(queries.len());
    for (index, query) in queries.iter().enumerate() {
        if query.scope != first.scope
            || query.table != first.table
            || query.query.limit != first.query.limit
            || query.query.metadata != first.query.metadata
            || query.query.metric != metric
            || query.embedding_dim != first.embedding_dim
        {
            return Err(PostgresStoreError::Backend(
                "batched graph vector queries must share scope and search options".to_owned(),
            ));
        }
        if query.query.embedding.len() != query.embedding_dim {
            return Err(StoreError::VectorDimensionMismatch {
                expected: query.embedding_dim,
                actual: query.query.embedding.len(),
            }
            .into());
        }
        finite_vector(&query.query.embedding)?;
        let index = i64::try_from(index).map_err(|_| {
            PostgresStoreError::Backend("vector query batch exceeds PostgreSQL limits".to_owned())
        })?;
        query_vectors.push(format!(
            "({index}::BIGINT, {}::vector)",
            sql_text_literal(&vector_literal(&query.query.embedding))
        ));
    }
    let (filter_sql, mut values) = graph_filter_sql_with_exclusions(
        &first.scope,
        &first.query.metadata,
        metadata_not_equals,
        1,
    )?;
    let limit_index = values.len() + 1;
    values.push(
        i64::try_from(first.query.limit)
            .unwrap_or(i64::MAX)
            .to_string(),
    );
    let params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = values
        .iter()
        .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
        .collect();
    let operator = match metric {
        kogwistar_store::DistanceMetric::Cosine => "<=>",
        kogwistar_store::DistanceMetric::L2 => "<->",
        kogwistar_store::DistanceMetric::InnerProduct => "<#>",
    };
    let rows = client
        .query(
            &format!(
                "WITH query_vectors(query_index, query_embedding) AS (VALUES {}) \
                 SELECT query_vectors.query_index, matches.id, matches.document, matches.metadata::TEXT, \
                        matches.embedding::TEXT, matches.distance \
                 FROM query_vectors CROSS JOIN LATERAL ( \
                     SELECT id, document, metadata, embedding, embedding {operator} query_vectors.query_embedding AS distance \
                     FROM {table} WHERE embedding IS NOT NULL AND {filter_sql} \
                     ORDER BY distance ASC, id ASC LIMIT ${limit_index}::TEXT::BIGINT \
                 ) AS matches ORDER BY query_vectors.query_index ASC, matches.distance ASC, matches.id ASC",
                query_vectors.join(", ")
            ),
            &params,
        )
        .await
        .map_err(backend)?;
    let mut result: Vec<Vec<VectorMatch>> = (0..queries.len()).map(|_| Vec::new()).collect();
    for row in &rows {
        let query_index: i64 = row.try_get(0).map_err(backend)?;
        let query_index = usize::try_from(query_index).map_err(|_| {
            PostgresStoreError::Backend("database returned invalid vector query index".to_owned())
        })?;
        let Some(matches) = result.get_mut(query_index) else {
            return Err(PostgresStoreError::Backend(
                "database returned out-of-range vector query index".to_owned(),
            ));
        };
        let metadata_json: String = row.try_get(3).map_err(backend)?;
        let metadata = serde_json::from_str(&metadata_json)?;
        let embedding_json: Option<String> = row.try_get(4).map_err(backend)?;
        let embedding = embedding_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()?;
        matches.push(VectorMatch {
            record: GraphRecord {
                id: row.try_get(1).map_err(backend)?,
                document: row.try_get(2).map_err(backend)?,
                metadata,
                embedding,
            },
            distance: row.try_get(5).map_err(backend)?,
        });
    }
    Ok(result)
}

pub(crate) async fn graph_acl_records_for_entities<C>(
    client: &C,
    tables: &Tables,
    scope: &kogwistar_store::GraphScope,
    truth_graph: &str,
    entity_ids: &[String],
) -> PostgresStoreResult<Vec<GraphRecord>>
where
    C: GenericClient + Sync,
{
    if entity_ids.is_empty() {
        return Ok(Vec::new());
    }
    if entity_ids.len() > 10_000 {
        return Err(PostgresStoreError::Backend(
            "ACL entity lookup exceeds 10000 IDs".to_owned(),
        ));
    }
    let table = graph_table(tables, "gke_nodes")?;
    let filter = kogwistar_store::MetadataFilter {
        equals: [
            (
                "entity_type".to_owned(),
                Value::String("acl_record".to_owned()),
            ),
            (
                "acl_truth_graph".to_owned(),
                Value::String(truth_graph.to_owned()),
            ),
        ]
        .into_iter()
        .collect(),
    };
    let (scope_filter, values) = graph_filter_sql(scope, &filter, 1)?;
    let targets = entity_ids
        .iter()
        .map(|entity_id| {
            let json_id = serde_json::to_string(entity_id)?;
            Ok(format!(
                "metadata -> 'acl_target_entity_id' = {}::JSONB",
                sql_text_literal(&json_id)
            ))
        })
        .collect::<PostgresStoreResult<Vec<String>>>()?
        .join(" OR ");
    let rows = client
        .query(
            &format!(
                "SELECT id, document, metadata::TEXT, embedding::TEXT FROM {table} WHERE {scope_filter} AND ({targets}) ORDER BY id"
            ),
            &values
                .iter()
                .map(|value| value as &(dyn tokio_postgres::types::ToSql + Sync))
                .collect::<Vec<_>>(),
        )
        .await
        .map_err(backend)?;
    rows.iter().map(graph_record_from_row).collect()
}
