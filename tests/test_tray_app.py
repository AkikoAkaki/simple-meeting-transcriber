"""Tests for UI logic and interactions in tray_app.py."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Ensure offscreen Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

import tray_app


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _mock_app_and_service():
    service = SimpleNamespace(
        settings=SimpleNamespace(
            watch_dir="/fake/watch",
            watcher_enabled=True,
            model="large-v3",
            device="auto",
            transcript_dir="/fake/transcripts",
        ),
        worker=SimpleNamespace(
            active=None,
            cancel_active=MagicMock(),
        ),
        store=SimpleNamespace(
            get=lambda _id: None,
            recent=lambda _limit=20: [
                {
                    "job_id": "job-1",
                    "source_path": "/fake/video1.mp4",
                    "status": "completed",
                    "updated_at": "2026-09-03T10:00:00",
                    "message": "Done",
                },
                {
                    "job_id": "job-2",
                    "source_path": "/fake/video2.mp4",
                    "status": "completed",
                    "updated_at": "2026-09-03T10:05:00",
                    "message": "Done",
                },
            ],
        ),
        token_store=SimpleNamespace(
            get=lambda: "fake-token",
            set=MagicMock(),
        ),
    )
    app = SimpleNamespace(service=service)
    return app


def test_manual_format_options_do_not_contain_srt(qapp):
    app = _mock_app_and_service()
    dashboard = tray_app.Dashboard(app)
    items = [dashboard.manual_format.itemText(i) for i in range(dashboard.manual_format.count())]
    assert "srt" not in items
    assert items == ["md", "txt"]


def test_refresh_preserves_selected_job_id(qapp):
    app = _mock_app_and_service()
    dashboard = tray_app.Dashboard(app)
    dashboard.refresh(update_recent=True)

    assert dashboard.recent_list.count() == 2
    # Select the second item (job-2)
    dashboard.recent_list.setCurrentRow(1)
    assert dashboard.recent_list.currentItem().data(Qt.ItemDataRole.UserRole) == "job-2"

    # Refresh again with update_recent=True
    dashboard.refresh(update_recent=True)
    # Selection should still be job-2
    current = dashboard.recent_list.currentItem()
    assert current is not None
    assert current.data(Qt.ItemDataRole.UserRole) == "job-2"


def test_refresh_without_update_recent_does_not_touch_recent_list(qapp):
    app = _mock_app_and_service()
    dashboard = tray_app.Dashboard(app)
    dashboard.refresh(update_recent=True)
    dashboard.recent_list.setCurrentRow(0)

    # Monkeypatch clear to detect if it gets called
    clear_called = False
    orig_clear = dashboard.recent_list.clear

    def mock_clear():
        nonlocal clear_called
        clear_called = True
        orig_clear()

    dashboard.recent_list.clear = mock_clear

    dashboard.refresh(update_recent=False)
    assert not clear_called, "refresh(update_recent=False) must not clear or rebuild recent_list"
    assert dashboard.recent_list.currentItem().data(Qt.ItemDataRole.UserRole) == "job-1"


def test_handle_event_skips_recent_refresh_on_progress_and_heartbeat(qapp, monkeypatch):
    controller = tray_app.TrayApp(qapp)
    try:
        controller.dashboard = tray_app.Dashboard(controller)
        controller.dashboard.setVisible(True)

        refresh_args = []
        orig_refresh = controller.dashboard.refresh

        def mock_refresh(update_recent=True):
            refresh_args.append(update_recent)
            orig_refresh(update_recent=update_recent)

        monkeypatch.setattr(controller.dashboard, "refresh", mock_refresh)

        # Progress event
        controller._handle_event("progress", {"progress": 0.5, "message": "50%"})
        assert refresh_args[-1] is False, "progress events must not refresh recent_list"

        # Heartbeat event
        controller._handle_event("heartbeat", {"message": "ping"})
        assert refresh_args[-1] is False, "heartbeat events must not refresh recent_list"

        # Completed event (state transition)
        controller._handle_event("completed", {"message": "Transcription complete"})
        assert refresh_args[-1] is True, "completed events must refresh recent_list"

        # Queued event (state transition)
        controller._handle_event("queued", {"message": "Queued"})
        assert refresh_args[-1] is True, "queued events must refresh recent_list"

        # Failed event (state transition)
        controller._handle_event("failed", {"message": "Error occurred"})
        assert refresh_args[-1] is True, "failed events must refresh recent_list"
    finally:
        controller.service.stop()


def test_make_icon_generates_multi_resolution_antialiased_qicon(qapp):
    icon = tray_app._make_icon("#3b82f6")
    sizes = [(sz.width(), sz.height()) for sz in icon.availableSizes()]
    assert (32, 32) in sizes
    assert (64, 64) in sizes
    assert (128, 128) in sizes


def test_model_box_contains_large_v3_turbo_as_recommended(qapp):
    app = _mock_app_and_service()
    dashboard = tray_app.Dashboard(app)
    items = [dashboard.model_box.itemText(i) for i in range(dashboard.model_box.count())]
    assert "large-v3-turbo" in items
    assert items[0] == "large-v3-turbo"
    for expected in ["large-v3", "medium", "small", "base", "tiny"]:
        assert expected in items


def test_dashboard_has_preview_label_and_cache_controls(qapp):
    app = _mock_app_and_service()
    dashboard = tray_app.Dashboard(app)

    assert hasattr(dashboard, "preview_label")
    assert dashboard.preview_label.text() == ""

    assert hasattr(dashboard, "clear_cache_btn")
    assert dashboard.clear_cache_btn.text() == "Clear Audio Cache"
    assert hasattr(dashboard, "clear_cache_button")

    assert hasattr(dashboard, "cache_size_label")
    assert hasattr(dashboard, "cache_label")
    assert "Cache" in dashboard.cache_size_label.text()


def test_preview_label_updates_on_progress_and_clears_on_completion_or_idle(qapp):
    controller = tray_app.TrayApp(qapp)
    try:
        controller.dashboard = tray_app.Dashboard(controller)
        assert controller.dashboard.preview_label.text() == ""

        # Progress with preview text
        controller._handle_event("progress", {"progress": 0.3, "preview": "Live transcript text"})
        assert controller.dashboard.preview_label.text() == "Live transcript text"

        # Subsequent progress updates preview
        controller._handle_event("progress", {"progress": 0.6, "preview": "Second phrase"})
        assert controller.dashboard.preview_label.text() == "Second phrase"

        # Completed clears preview
        controller._handle_event("completed", {"message": "Done"})
        assert controller.dashboard.preview_label.text() == ""

        # Another progress then failed clears preview
        controller._handle_event("progress", {"progress": 0.1, "preview": "Failing segment"})
        assert controller.dashboard.preview_label.text() == "Failing segment"
        controller._handle_event("failed", {"message": "Error"})
        assert controller.dashboard.preview_label.text() == ""
    finally:
        controller.service.stop()


def test_clear_audio_cache_dialog_confirm_and_cancel(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    app = _mock_app_and_service()
    clear_mock = MagicMock(return_value={"deleted_count": 3, "reclaimed_bytes": 3145728, "reclaimed_size": "3.0 MB"})
    app.service.clear_audio_cache = clear_mock
    app.service.get_cache_size = MagicMock(return_value="0 B")

    dashboard = tray_app.Dashboard(app)

    # 1. User cancels confirmation -> should NOT clear
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.No)
    dashboard._clear_audio_cache()
    clear_mock.assert_not_called()

    # 2. User confirms -> should call clear_audio_cache and append log
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes)
    dashboard._clear_audio_cache()
    clear_mock.assert_called_once()
    assert "3 file(s) removed (3.0 MB reclaimed)" in dashboard.log.toPlainText()
    assert "0 B" in dashboard.cache_size_label.text()

