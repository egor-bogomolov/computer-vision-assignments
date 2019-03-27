from typing import List, Dict

from collections import namedtuple
import numpy as np
from scipy.optimize import approx_fprime as derivative

from corners import FrameCorners
from _camtrack import *

ProjectionError = namedtuple(
    'ProjectionError',
    ('frame_id', 'id_3d', 'id_2d')
)


def run_bundle_adjustment(intrinsic_mat: np.ndarray,
                          list_of_corners: List[FrameCorners],
                          max_inlier_reprojection_error: float,
                          view_mats: List[np.ndarray],
                          pc_builder: PointCloudBuilder) -> List[np.ndarray]:

    proj_mats = [intrinsic_mat @ view_mat for view_mat in view_mats]
    projection_errors = []
    used_3d_points_inds = set()

    for k, (proj_mat, corners) in enumerate(zip(proj_mats, list_of_corners)):
        indices = np.array([ind for ind in corners.ids if ind in pc_builder.ids], dtype=np.int32)
        indices_2d_local = np.array([k for k, ind in enumerate(corners.ids) if ind in indices], np.int32)
        indices_3d_local = np.array([k for k, ind in enumerate(pc_builder.ids) if ind in indices], np.int32)
        inlier_indices = calc_inlier_indices(pc_builder.points[indices_3d_local],
                                             corners.points[indices_2d_local],
                                             proj_mat,
                                             max_inlier_reprojection_error)
        for ind in inlier_indices:
            id_3d = indices_3d_local[ind]
            id_2d = indices_2d_local[ind]
            used_3d_points_inds.add(id_3d)
            projection_errors.append(ProjectionError(frame_id=k, id_3d=id_3d, id_2d=id_2d))

    used_3d_points_inds = list(sorted(used_3d_points_inds))
    point_ind_to_position = {}
    n_matrix_params = 6 * len(view_mats)
    p = np.concatenate([_view_mats3x4_to_rt(view_mats),
                        _points_to_flat_coordinates(pc_builder.points[used_3d_points_inds])])
    for k, point_ind in enumerate(used_3d_points_inds):
        point_ind_to_position[point_ind] = n_matrix_params + 3 * k

    _run_optimization(projection_errors, list_of_corners, point_ind_to_position, p, intrinsic_mat, 4)

    for k in range(len(view_mats)):
        r_vec = p[6 * k : 6 * k + 3].reshape(3, 1)
        t_vec = p[6 * k + 3 : 6 * k + 6].reshape(3, 1)
        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        view_mats[k] = view_mat

    pc_builder.update_points(pc_builder.ids[used_3d_points_inds], p[n_matrix_params:].reshape(-1, 3))

    return view_mats


def _view_mats3x4_to_rt(view_mats: List[np.ndarray]) -> np.ndarray:
    result = np.zeros(6 * len(view_mats))
    for k, mat in enumerate(view_mats):
        pos = 6 * k
        r, t = view_mat3x4_to_rodrigues_and_translation(mat)
        result[pos: pos + 3] = r[:, 0]
        result[pos + 3: pos + 6] = t[:]
    return result


def _points_to_flat_coordinates(points: np.ndarray) -> np.ndarray:
    return points.reshape(-1)


def _vec_to_proj_mat(vec: np.ndarray, intrinsic_mat: np.ndarray) -> np.ndarray:
    r_vec = vec[0:3].reshape(3, 1)
    t_vec = vec[3:6].reshape(3, 1)
    view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
    return np.dot(intrinsic_mat, view_mat)


def _reprojection_error(vec: np.ndarray, point2d: np.ndarray, intrinsic_mat: np.ndarray) -> np.float32:
    return compute_reprojection_error(vec[6:9], point2d, _vec_to_proj_mat(vec, intrinsic_mat))


def _reprojection_errors(projection_errors: List[ProjectionError],
                         list_of_corners: List[FrameCorners],
                         mapping: Dict[int, int],
                         p: np.ndarray,
                         intrinsic_mat: np.ndarray) -> np.ndarray:
    errors = np.zeros(len(projection_errors))
    for i, proj_err in enumerate(projection_errors):
        vec = np.zeros(9)

        mat_pos = 6 * proj_err.frame_id
        vec[:6] = p[mat_pos: mat_pos + 6]

        point_pos = mapping[proj_err.id_3d]
        vec[6:] = p[point_pos: point_pos + 3]

        point2d = list_of_corners[proj_err.frame_id].points[proj_err.id_2d]

        errors[i] = _reprojection_error(vec, point2d, intrinsic_mat)

    return errors


def _compute_jacobian(projection_errors: List[ProjectionError],
                      list_of_corners: List[FrameCorners],
                      mapping: Dict[int, int],
                      p: np.ndarray,
                      intrinsic_mat: np.ndarray) -> np.ndarray:

    J = np.zeros((len(projection_errors), len(p)))
    for row, proj_err in enumerate(projection_errors):
        vec = np.zeros(9)

        mat_pos = 6 * proj_err.frame_id
        vec[:6] = p[mat_pos: mat_pos + 6]

        point_pos = mapping[proj_err.id_3d]
        vec[6:] = p[point_pos: point_pos + 3]

        point2d = list_of_corners[proj_err.frame_id].points[proj_err.id_2d]

        partial_derivatives = derivative(vec,
                                         lambda v: _reprojection_error(v, point2d, intrinsic_mat),
                                         np.full_like(vec, 1e-9))

        for i in range(6):
            J[row, mat_pos + i] = partial_derivatives[i]

        for i in range(3):
            J[row, point_pos + i] = partial_derivatives[6 + i]

    return J


def _run_optimization(projection_errors: List[ProjectionError],
                      list_of_corners: List[FrameCorners],
                      mapping: Dict[int, int],
                      p: np.ndarray,
                      intrinsic_mat: np.ndarray,
                      n_steps: int):
    n = 6 * len(list_of_corners)
    lmbd = 10
    errors = _reprojection_errors(projection_errors, list_of_corners, mapping, p, intrinsic_mat)
    print(errors.sum())
    for step in range(n_steps):
        J = _compute_jacobian(projection_errors, list_of_corners, mapping, p, intrinsic_mat)
        JJ = J.T @ J
        JJ += lmbd * np.diag(np.diag(JJ))
        U = JJ[:n, :n]
        W = JJ[:n, n:]
        V = JJ[n:, n:]
        V_inv = np.zeros_like(V)
        for i in range(0, len(V), 3):
            s = 3 * i
            t = 3 * i + 3
            V_inv[s:t, s:t] = np.linalg.inv(V[s:t, s:t])

        g = J.T @ _reprojection_errors(projection_errors, list_of_corners, mapping, p, intrinsic_mat)
        A = U - W @ V_inv @ W.T
        b = W @ V_inv @ g[n:] - g[:n]
        dc = np.linalg.solve(A, b)
        dx = V_inv @ (-g[n:] - W.T @ dc)

        p[:n] += dc
        p[n:] += dx

    errors = _reprojection_errors(projection_errors, list_of_corners, mapping, p, intrinsic_mat)
    print(errors.sum())
