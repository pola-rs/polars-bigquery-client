use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(input_fields[0].clone())
}

#[polars_expr(output_type_func=same_output_type)]
fn unsupported(inputs: &[Series]) -> PolarsResult<Series> {
    Ok(inputs[0].clone())
}
