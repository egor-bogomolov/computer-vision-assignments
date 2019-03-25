#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _image_to_uint8(image):
    return (image * 255).astype(np.uint8)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder,
                max_corners: int = 500) -> None:

    # Parameters for Shi Tomasi corner detection
    feature_params = dict(maxCorners=max_corners,
                          qualityLevel=0.01,
                          minDistance=7,
                          blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Maximal distance over single axis between original and reversed point
    forward_backward_threshold = 0.8

    # Find corners and reshape them
    def get_corners(img, _mask=None):
        # Can be none since mask is used
        features = cv2.goodFeaturesToTrack(img, mask=_mask, **feature_params)
        if features is not None:
            return features.squeeze(axis=1)
        return None

    # Set corners at frame
    def set_corners(frame, _ids, _corner_points):
        builder.set_corners_at_frame(frame, FrameCorners(
            ids=_ids,
            points=_corner_points,
            sizes=np.full(len(_ids), feature_params['blockSize'])
        ))

    # Find initial corners
    image = frame_sequence[0]
    corners = get_corners(image)
    cur_id = len(corners)
    ids = np.arange(cur_id)
    set_corners(0, ids, corners)

    for i, (old_image, image) in enumerate(zip(frame_sequence[:-1], frame_sequence[1:]), 1):
        old_image_uint8 = _image_to_uint8(old_image)
        image_uint8 = _image_to_uint8(image)
        # To reduce number of false-positives perform forward-backward check
        moved_corners, st_f, err = cv2.calcOpticalFlowPyrLK(old_image_uint8, image_uint8, corners, None, **lk_params)
        backed_corners, st_b, err = cv2.calcOpticalFlowPyrLK(image_uint8, old_image_uint8, moved_corners, None, **lk_params)

        # Keep tracked corners
        tracked = np.logical_and(st_f.squeeze(-1) == 1, st_b.squeeze(-1) == 1)
        tracked = np.logical_and(
            tracked,
            abs(corners - backed_corners).sum(axis=-1) < forward_backward_threshold
        )
        corners = moved_corners[tracked]
        ids = ids[tracked]

        deficit = feature_params['maxCorners'] - len(corners)

        # Find new corners
        if deficit > 0:
            mask = np.full_like(image, 255, dtype=np.uint8)
            for corner in corners:
                cv2.circle(mask, (corner[0], corner[1]), feature_params['minDistance'], 0, -1)
            candidate_corners = get_corners(image, _mask=mask)
            if candidate_corners is not None:
                if len(candidate_corners) > deficit:
                    candidate_corners = candidate_corners[:deficit]
                addition = min(deficit, len(candidate_corners))
                ids = np.concatenate([ids, np.arange(cur_id, cur_id + addition)])
                corners = np.concatenate([corners, candidate_corners], axis=0)
                cur_id += deficit

        set_corners(i, ids, corners)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True,
          max_corners: int = 500) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :param max_corners: maximal number of corners per frame.
    :return: corners for all frames of given sequence.
    """
    max_corners = max(max_corners, len(frame_sequence) * 10)
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder, max_corners)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder, max_corners)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
