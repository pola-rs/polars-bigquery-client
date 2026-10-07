use pyo3::prelude::*;

use crate::stream::ArrowStreamExporter;

#[pyclass]
#[derive(Clone)]
pub struct DropFlag {
    value: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[pymethods]
impl DropFlag {
    fn is_set(&self) -> bool {
        self.value.load(std::sync::atomic::Ordering::SeqCst)
    }
}

fn placeholder_schema() -> polars_arrow::datatypes::ArrowSchemaRef {
    let field = polars_arrow::datatypes::Field::new(
        "placeholder".into(),
        polars_arrow::datatypes::ArrowDataType::Int32,
        true,
    );
    let schema = polars_arrow::datatypes::ArrowSchema::from_iter(vec![field]);
    std::sync::Arc::new(schema)
}

#[pyfunction]
pub fn _create_test_exporter() -> ArrowStreamExporter {
    let rt = pyo3_async_runtimes::tokio::get_runtime();
    rt.block_on(async {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    });

    let (tx, rx) = tokio::sync::mpsc::channel(10);
    // Keep the sender alive inside a task owned by the receiver so rx.recv() blocks
    // until interrupted or dropped, without leaking heap memory via Box::leak.
    let handle = rt.spawn(async move {
        let _keep_tx_alive = tx;
        std::future::pending::<()>().await;
    });

    let receiver =
        arrow_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, vec![handle]);
    ArrowStreamExporter::new(placeholder_schema(), receiver)
}

#[pyfunction]
pub fn _test_create_exporter_with_drop_flag() -> (ArrowStreamExporter, DropFlag) {
    let rt = pyo3_async_runtimes::tokio::get_runtime();

    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag_clone = flag.clone();
    let (tx, rx) = tokio::sync::mpsc::channel(10);

    // Spawn a placeholder task that holds `tx` and runs until aborted, setting `flag` on drop.
    let handle = rt.spawn(async move {
        struct SetOnDrop(std::sync::Arc<std::sync::atomic::AtomicBool>);
        impl Drop for SetOnDrop {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let _cleanup = SetOnDrop(flag_clone);
        let _keep_tx_alive = tx;

        std::future::pending::<()>().await;
    });

    let receiver =
        arrow_bigquery_lib::BigQueryRecordBatchReceiver::new_for_testing(rx, vec![handle]);
    let exporter = ArrowStreamExporter::new(placeholder_schema(), receiver);
    let drop_flag = DropFlag { value: flag };

    (exporter, drop_flag)
}
