import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.hf_model_manager import (
    MANAGED_MODELS,
    all_model_status,
    delete_model,
    download_model,
    get_cache_root,
    hub_cache_dir,
    set_cache_root,
)


class _ModelTaskWorker(QObject):
    message = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, action: str, model_keys: list[str], cache_root: str):
        super().__init__()
        self.action = action
        self.model_keys = model_keys
        self.cache_root = cache_root

    def run(self):
        lookup = {model.key: model for model in MANAGED_MODELS}
        try:
            for key in self.model_keys:
                model = lookup[key]
                if self.action == "download":
                    self.message.emit(f"Downloading {model.label}...")
                    path = download_model(model, self.cache_root)
                    self.message.emit(f"Downloaded {model.label} to {path}")
                elif self.action == "delete":
                    self.message.emit(f"Deleting {model.label}...")
                    delete_model(model, self.cache_root)
                    self.message.emit(f"Deleted {model.label}")
            self.finished.emit(True, "Done")
        except Exception:
            self.finished.emit(False, traceback.format_exc())


class ModelManagerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._worker = None

        layout = QVBoxLayout(self)

        location_row = QHBoxLayout()
        location_row.addWidget(QLabel("HF cache:"))
        self.cache_edit = QLineEdit(str(get_cache_root()))
        self.cache_edit.setMinimumWidth(420)
        location_row.addWidget(self.cache_edit, stretch=1)

        self.choose_btn = QPushButton("Choose...")
        self.choose_btn.clicked.connect(self._choose_cache)
        location_row.addWidget(self.choose_btn)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._apply_cache)
        location_row.addWidget(self.apply_btn)
        layout.addLayout(location_row)

        self.hub_label = QLabel()
        layout.addWidget(self.hub_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Model", "Purpose", "Repo", "Status", "Size"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=1)

        button_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        button_row.addWidget(self.refresh_btn)

        self.download_btn = QPushButton("Download Selected")
        self.download_btn.clicked.connect(self._download_selected)
        button_row.addWidget(self.download_btn)

        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        button_row.addWidget(self.delete_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        layout.addWidget(self.log)

        self.refresh()

    def _choose_cache(self):
        path = QFileDialog.getExistingDirectory(self, "Select Hugging Face cache folder", self.cache_edit.text())
        if path:
            self.cache_edit.setText(path)

    def _apply_cache(self):
        try:
            root = set_cache_root(self.cache_edit.text().strip())
        except Exception as exc:
            QMessageBox.critical(self, "Cache Location", str(exc))
            return False
        self.cache_edit.setText(str(root))
        self._log(f"Cache root set to {root}")
        self.refresh()
        return True

    def _selected_keys(self) -> list[str]:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self.table.item(row, 0).data(Qt.UserRole) for row in rows]

    def _download_selected(self):
        keys = self._selected_keys()
        if not keys:
            QMessageBox.information(self, "Download Models", "Select one or more models first.")
            return
        if not self._apply_cache():
            return
        self._run_task("download", keys)

    def _delete_selected(self):
        keys = self._selected_keys()
        if not keys:
            QMessageBox.information(self, "Delete Models", "Select one or more models first.")
            return
        reply = QMessageBox.question(
            self,
            "Delete Models",
            "Delete the selected model caches from disk?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        if not self._apply_cache():
            return
        self._run_task("delete", keys)

    def _run_task(self, action: str, keys: list[str]):
        self._set_busy(True)
        self._thread = QThread(self)
        self._worker = _ModelTaskWorker(action, keys, self.cache_edit.text().strip())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self._log)
        self._worker.finished.connect(self._task_finished)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _task_finished(self, ok: bool, message: str):
        if ok:
            self._log(message)
        else:
            self._log(message)
            QMessageBox.critical(self, "Model Manager", message)
        self._set_busy(False)
        self._thread = None
        self._worker = None
        self.refresh()

    def _set_busy(self, busy: bool):
        for widget in (
            self.choose_btn,
            self.apply_btn,
            self.refresh_btn,
            self.download_btn,
            self.delete_btn,
            self.table,
        ):
            widget.setEnabled(not busy)

    def _log(self, message: str):
        self.log.append(message)

    def refresh(self):
        root = Path(self.cache_edit.text().strip() or str(get_cache_root()))
        statuses = all_model_status(root)
        self.hub_label.setText(f"Hub cache: {hub_cache_dir(root)}")

        self.table.setRowCount(len(MANAGED_MODELS))
        for row, model in enumerate(MANAGED_MODELS):
            status = statuses[model.key]
            values = [
                model.label,
                model.purpose,
                model.repo_id,
                "Downloaded" if status["present"] else "Not downloaded",
                status["size"],
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == 0:
                    item.setData(Qt.UserRole, model.key)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
