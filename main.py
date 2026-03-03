from player_ball_assigner import PlayerBallAssigner
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assignment import TeamAssigner
from camera_movement import CameraMovementEstimator
from viewtransformer import ViewTransformer
from speed_and_distance_etimator import Speed_and_Distance_Estimator
from pathlib import Path
import os


# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent


def run_pipeline(
    input_video_path: str = 'inputs/video1.mp4',
    output_video_path: str = 'output_videos/output_video_final.mp4',
    use_stubs: bool = True,
):
    # Convert relative paths to absolute paths based on project root
    if not os.path.isabs(input_video_path):
        input_video_path = str(PROJECT_ROOT / input_video_path)
    if not os.path.isabs(output_video_path):
        output_video_path = str(PROJECT_ROOT / output_video_path)

    # Validate input video exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    video_frames = read_video(input_video_path)
    if not video_frames or len(video_frames) == 0:
        raise ValueError(f"No frames could be read from video: {input_video_path}")

    model_path = str(PROJECT_ROOT / 'models/weights/best.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Please ensure the model file is in the repository."
        )
    tracker = Tracker(model_path)
    stub_path = str(PROJECT_ROOT / 'stubs/track_stubs.pkl')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=stub_path,
    )
    tracker.add_position_to_tracks(tracks)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_stub_path = str(PROJECT_ROOT / 'stubs/camera_movement.pkl')
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=camera_movement_stub_path,
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )


    view_transformer = ViewTransformer(reference_frame=video_frames[0], use_keypoint_model=True)
    view_transformer.add_transformed_position_to_tracks(tracks)

    speed_and_distance_estimator = Speed_and_Distance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_number], track['bbox'], player_id
            )
            tracks['players'][frame_number][player_id]['team'] = team
            tracks['players'][frame_number][player_id]['team_color'] = (
                team_assigner.team_colors[team]
            )

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    last_team_with_ball = 0
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bbox
        )

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            last_team_with_ball = (
                tracks['players'][frame_num][assigned_player].get('team', 0) or 0
            )

        team_ball_control.append(last_team_with_ball)

    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )
    output_video_frames = speed_and_distance_estimator.draw_speed_and_distance(
        tracks, output_video_frames
    )

    save_video(output_video_frames, output_video_path)

    return output_video_path


def main():
    
    run_pipeline(
        input_video_path='inputs/video1.mp4',
        output_video_path='output_videos/output_video_final.mp4',
        use_stubs=True,
    )


if __name__ == '__main__':
    main()
