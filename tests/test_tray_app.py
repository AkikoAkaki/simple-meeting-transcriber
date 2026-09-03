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
