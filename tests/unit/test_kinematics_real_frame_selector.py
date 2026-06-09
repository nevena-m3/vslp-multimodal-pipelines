from pathlib import Path


def test_kinematics_real_frame_selector_strings_present():
    app = Path('src/vslp/gui/kinematics/app.py').read_text(encoding='utf-8')
    assert 'Load Selected Frame' in app
    assert 'selected_landmark_video_overlay_preview.png' in app
    assert 'Region quick-select' in app
    assert 'Real video frame + Google MediaPipe overlay' in app
