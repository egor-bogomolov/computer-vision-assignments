#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Tuple

import numpy as np
import cv2

from corners import CornerStorage, FrameCorners
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import *


class CameraTracker:

    def __init__(self, corner_storage: CornerStorage, intrinsic_mat: np.ndarray, parameters: TriangulationParameters):
        self._corner_storage = corner_storage
        self._intrinsic_mat = intrinsic_mat
        self._triangulation_parameters = parameters

        self._n_frames = len(corner_storage)
        self._track = [None] * self._n_frames
        self._point_positions = [None] * (corner_storage.max_corner_id() + 1)

        frame_ind1, frame_ind2 = self._initialization()
        print(frame_ind1, frame_ind2)
        print(self._n_frames)
        self._add_cloud_points(frame_ind1, frame_ind2)
        self._tracking()

    def _initialization(self):
        poses = []
        poses_cloud_size = []
        indices = []
        for i in range(1, len(self._corner_storage)):
            pose, pose_cloud_size = self._two_frame_initialization(self._corner_storage[0], self._corner_storage[i])
            poses.append(pose)
            poses_cloud_size.append(pose_cloud_size)
            indices.append(i)

        index = np.argmax(poses_cloud_size)

        self._track[0] = eye3x4()
        self._track[indices[index]] = pose_to_view_mat3x4(poses[index])

        return 0, indices[index]

    def _two_frame_initialization(self, frame1: FrameCorners, frame2: FrameCorners):
        correspondences = build_correspondences(frame1, frame2)

        if correspondences.points_1.shape[0] < 5:
            return None, 0

        essential_mat, mask_essential = cv2.findEssentialMat(correspondences.points_1, correspondences.points_2,
                                                             self._intrinsic_mat,
                                                             method=cv2.RANSAC, prob=0.999, threshold=1.)

        _, mask_homography = cv2.findHomography(correspondences.points_1, correspondences.points_2,
                                                method=cv2.RANSAC, confidence=0.999,
                                                ransacReprojThreshold=self._triangulation_parameters.max_reprojection_error)

        # zeros in mask correspond to outliers
        essential_inliers = np.count_nonzero(mask_essential)
        homography_inliers = np.count_nonzero(mask_homography)

        if essential_inliers < 1. * homography_inliers:
            return None, 0

        correspondences_filtered = remove_correspondences_with_ids(correspondences, np.argwhere(mask_essential == 0))

        R1, R2, t = cv2.decomposeEssentialMat(essential_mat)
        possible_poses = [Pose(R1, t), Pose(R2, t), Pose(R1, -t), Pose(R2, -t)]
        poses2points = [0] * 4

        for i, pose in enumerate(possible_poses):
            points, ids = triangulate_correspondences(correspondences_filtered, eye3x4(), pose_to_view_mat3x4(pose),
                                                      self._intrinsic_mat, self._triangulation_parameters)
            poses2points[i] = points.shape[0]

        best_pose = np.argmax(poses2points)
        return possible_poses[best_pose], poses2points[best_pose]

    def _add_points(self, points, ids):
        cnt = 0
        for point, ind in zip(points, ids):
            if self._point_positions[ind] is None:
                self._point_positions[ind] = point
                cnt += 1
        return cnt

    def _add_cloud_points(self, frame_ind1, frame_ind2):
        frame1, frame2 = self._corner_storage[frame_ind1], self._corner_storage[frame_ind2]
        matrix1, matrix2 = self._track[frame_ind1], self._track[frame_ind2]
        correspondences = build_correspondences(frame1, frame2)
        points, ids = triangulate_correspondences(correspondences, matrix1, matrix2,
                                                  self._intrinsic_mat, self._triangulation_parameters)
        self._log_added_points(frame_ind1, frame_ind2, self._add_points(points, ids))

    def _tracking(self):
        added = True
        while added:
            added = False
            for i in range(self._n_frames):
                if self._track[i] is not None:
                    continue

                self._track[i] = self._compute_frame_matrix(self._corner_storage[i])
                if self._track[i] is not None:
                    for j in range(self._n_frames):
                        if self._track[j] is not None and i != j:
                            self._add_cloud_points(i, j)
                    print(f"Processed frame {i}")
                    added = True

        prev = -1
        for i in range(self._n_frames):
            if self._track[i] is None:
                self._track[i] = self._track[prev]
            else:
                prev = i



    def _compute_frame_matrix(self, frame: FrameCorners):
        frame_ids = frame.ids.squeeze(-1)
        mask = np.ones_like(frame_ids)
        for i, ind in enumerate(frame_ids):
            if self._point_positions[ind] is None:
                mask[i] = 0

        ids = frame_ids[mask == 1]
        points = frame.points[mask == 1]
        n_points = points.shape[0]

        if n_points < 4:
            return None

        object_points = np.array([self._point_positions[ind] for ind in ids if self._point_positions[ind] is not None])
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, points, self._intrinsic_mat, None)

        if not retval:
            return None

        inliers = inliers.squeeze(-1)
        for ind in frame_ids:
            if ind not in inliers:
                self._point_positions[ind] = None

        return rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

    def _log_added_points(self, frame1, frame2, added_points):
        if added_points > 0:
            print(f'Added {added_points} points from frames {frame1} and {frame2}')

    def point_cloud_builder(self) -> PointCloudBuilder:
        return PointCloudBuilder(ids=np.array([i for i, point in enumerate(self._point_positions) if point is not None]),
                                 points=np.array([point for point in self._point_positions if point is not None]))

    def track(self) -> np.ndarray:
        return np.array(self._track)


def _track_camera(corner_storage: CornerStorage,
                  intrinsic_mat: np.ndarray) \
        -> Tuple[List[np.ndarray], PointCloudBuilder]:
    parameters = TriangulationParameters(max_reprojection_error=0.1, min_triangulation_angle_deg=5., min_depth=1.)
    tracker = CameraTracker(corner_storage, intrinsic_mat, parameters)
    return tracker.track(), tracker.point_cloud_builder()


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
