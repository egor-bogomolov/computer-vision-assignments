#! /usr/bin/env python3

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo

import data3d


def _build_vertex_color_program():
    vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;

        in vec3 position;
        in vec3 color;

        out vec3 inter_color;
        
        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
            inter_color = color;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 130
        in vec3 inter_color;
        
        out vec3 out_color;

        void main() {
            out_color = inter_color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


def _build_fragment_color_program():
    vertex_shader = shaders.compileShader(
        """
        #version 130
        uniform mat4 mvp;

        in vec3 position;
       
        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
        }""",
        GL.GL_VERTEX_SHADER
    )
    fragment_shader = shaders.compileShader(
        """
        #version 130
        uniform vec3 color;
        
        out vec3 out_color;

        void main() {
            out_color = color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        vertex_shader, fragment_shader
    )


class CameraTrackRenderer:
    _yellow = np.array([1., 1., 0], dtype=np.float32)
    _white = np.array([1., 1., 1.], dtype=np.float32)
    _blue = np.array([0., 0., 1.], dtype=np.float32)

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """

        self._camera_parameters = tracked_cam_parameters
        self._camera_track_positions = np.array([pose.t_vec for pose in tracked_cam_track], dtype=np.float32)
        self._camera_track_rotations = np.array([pose.r_mat for pose in tracked_cam_track], dtype=np.float32)

        self._cloud_ids = point_cloud.ids
        self._n_cloud_points = len(point_cloud.ids)
        self._cloud_points_buffer = self._bufferize(point_cloud.points)
        self._cloud_colors_buffer = self._bufferize(point_cloud.colors)

        self._vertex_program = _build_vertex_color_program()
        self._fragment_program = _build_fragment_color_program()

        camera_vertices = self._load_obj_file(cam_model_files[0])
        self._camera_vertices = camera_vertices
        self._n_camera_vertices = len(camera_vertices)

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        model = np.diag([1, -1, -1, 1]).astype(np.float32)
        view = self._view_matrix(camera_tr_vec, camera_rot_mat)
        projection = self._projection_matrix_from_fovy(camera_fov_y)
        mvp = projection.dot(view.dot(model))

        # Cloud of points
        self._render_object(mvp, self._cloud_points_buffer, self._cloud_colors_buffer,
                            self._vertex_program, self._n_cloud_points)

        # Camera track
        # print(self._camera_track_positions[:tracked_cam_track_pos + 1])
        self._render_object(mvp, self._bufferize(self._camera_track_positions[:tracked_cam_track_pos + 1]),
                            self._white, self._fragment_program, tracked_cam_track_pos + 1,
                            uniform_color=True, drawing_object=GL.GL_LINE_STRIP)

        # Frustum
        frustum_corners = self._frustum_corners(camera_fov_y,
                                               self._camera_track_positions[tracked_cam_track_pos],
                                               self._camera_track_rotations[tracked_cam_track_pos])

        self._render_object(mvp, self._bufferize(frustum_corners),
                            self._yellow, self._fragment_program, 4,
                            uniform_color=True, drawing_object=GL.GL_LINE_LOOP)

        self._render_object(mvp, self._bufferize(np.array(
            [[self._camera_track_positions[tracked_cam_track_pos], p] for p in frustum_corners]
        )),
                            self._yellow, self._fragment_program, 8,
                            uniform_color=True, drawing_object=GL.GL_LINES)

        camera_pos = self._camera_track_positions[tracked_cam_track_pos]
        camera_rot = self._camera_track_rotations[tracked_cam_track_pos]
        rot_matrix = np.diag([-1., -1., -1.])
        camera_vertices = np.array([rot_matrix.dot(camera_rot.dot(v)) + camera_pos
                                    for v in self._camera_vertices], dtype=np.float32)
        self._render_object(mvp, self._bufferize(camera_vertices),
                            self._blue, self._fragment_program, self._n_camera_vertices,
                            uniform_color=True, drawing_object=GL.GL_TRIANGLES)

        GLUT.glutSwapBuffers()

    @staticmethod
    def _view_matrix(camera_tr_vec, camera_rot_mat):
        translation = np.eye(4, dtype=np.float32)
        translation[:3, 3] = -camera_tr_vec

        rotation = np.eye(4, dtype=np.float32)
        rotation[:3, :3] = np.linalg.inv(camera_rot_mat)

        return np.dot(rotation, translation)

    def _projection_matrix_from_fovy(self, fovy, znear=0.5, zfar=100.):
        t = np.tan(fovy) * znear
        r = t * self._aspect_ratio()
        return self._projection_matrix(znear, zfar, r, t)

    @staticmethod
    def _projection_matrix(n, f, r, t):
        return np.array([[n / r, 0, 0, 0],
                         [0, n / t, 0, 0],
                         [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                         [0, 0, -1, 0]],
                        dtype=np.float32)

    def _frustum_corners(self, fovy, camera_tr_vec, camera_rot_mat):
        z = 5.

        t = np.tan(fovy) * z
        b = -t
        r = t * self._aspect_ratio()
        l = -r

        return np.array([camera_tr_vec] * 4, dtype=np.float32) + camera_rot_mat.dot(np.array([[r, t, z],
                                                                                              [r, b, z],
                                                                                              [l, b, z],
                                                                                              [l, t, z]]).T).T

    @staticmethod
    def _render_object(mvp, position, color, program, n_points, uniform_color=False, drawing_object=GL.GL_POINTS):
        shaders.glUseProgram(program)
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(program, 'mvp'),
            1, True, mvp)

        # If color is fragment-level then use uniform, pass an array otherwise
        if uniform_color:
            GL.glUniform3fv(
                GL.glGetUniformLocation(program, 'color'),
                1, color)
        else:
            color.bind()
            color_loc = GL.glGetAttribLocation(program, 'color')
            GL.glEnableVertexAttribArray(color_loc)
            GL.glVertexAttribPointer(color_loc, 3, GL.GL_FLOAT,
                                     False, 0,
                                     color)

        position.bind()
        position_loc = GL.glGetAttribLocation(program, 'position')
        GL.glEnableVertexAttribArray(position_loc)
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 position)
        GL.glDrawArrays(drawing_object, 0, n_points)

        GL.glDisableVertexAttribArray(position_loc)
        position.unbind()
        if not uniform_color:
            GL.glDisableVertexAttribArray(color_loc)
            color.unbind()
        shaders.glUseProgram(0)

    @staticmethod
    def _aspect_ratio():
        return GLUT.glutGet(GLUT.GLUT_WINDOW_WIDTH) / GLUT.glutGet(GLUT.GLUT_WINDOW_HEIGHT)

    @staticmethod
    def _bufferize(arr):
        return vbo.VBO(arr.astype(np.float32).reshape(-1))

    @staticmethod
    def _load_obj_file(filename):
        with open(filename, 'r') as f:
            vertices = []
            faces = []
            for line in f:
                t, *r = line.split()
                if t == 'v':
                    vertices.append(list(map(np.float32, r)))
                elif t == 'f':
                    faces.append(list(map(lambda ind: int(ind) - 1, r)))

            return np.array([vertices[ind] for face in faces for ind in face], dtype=np.float32)