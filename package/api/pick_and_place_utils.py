"""
Utility functions for the interactive pick-and-place notebooks.

This module intentionally keeps stateless helper code separate from the
stateful classes in ``pick_and_place_core.py``. The functions here cover scene
construction, SG2 joint grouping, RGB-D capture, depth-to-point-cloud
conversion, OWLv2+SAM2 segmentation post-processing, target generation, and
simple object attachment.

Most helpers are notebook-facing and favor explicit inputs/outputs over hidden
global state. This makes the notebooks easier to read: the outer loop can call
``run_prompt_to_segmentation`` twice, ``build_point_cloud_observation`` twice,
then hand the resulting 3D targets to the controller.

Student implementation guide:
    Search for the marker ``FILL IN`` to find the project blanks.

    Required blanks in this file:
        1. ``build_point_cloud_observation``:
           Compute the center of a masked point cloud. This turns RGB-D
           perception into a 3D pick/place target.
        2. ``get_grasp_offsets``:
           Define guide-point offsets around the gripper TCP. These guide
           points help IK preserve gripper orientation, not only position.
        3. ``make_pick_place_phase_targets``:
           Design the phase-wise targets for a pick-and-place sequence.
        4. ``get_gripper_ctrl``:
           Convert a desired open/close state into gripper actuator commands.

    These blanks are intentionally not presented as the only possible solution.
    They are places where you should reason about what information the robot
    has, what constraints the IK solver needs, and why pick-and-place is usually
    organized as a sequence of phases.
"""

from __future__ import annotations

import cv2
import mujoco
import numpy as np
from pathlib import Path

from package.kinematics.transforms import D2R, np_uv, pr2t, rpy2r, t2p, view_in_world  # type: ignore
from package.mujoco_sim.mujoco_utils import merge_mjcfs  # type: ignore
from package.mujoco_sim.parser import MuJoCoParser  # type: ignore
from package.utility.utils import group_by_prefix, sample_xyzs  # type: ignore

# -----------------------------------------------------------------------------
# Scene / robot setup
# -----------------------------------------------------------------------------

def build_pick_and_place_xml(
        obj_pairs,
        n_obj,
        bowl_item="bowl_5",
        output_xml_path="xml/scene_ffw_sg2_pick_n_place.xml",
        objaverse_path="../asset/objaverse/",
        verbose=True,
    ):
    """
    Build the pick-and-place MuJoCo scene.

    The function merges the SG2 robot, table, floor, sampled Objaverse objects,
    and bowl into one MJCF file, then returns a parsed ``MuJoCoParser``
    environment. It is the first scene-construction step used by the
    interactive notebooks.

    Parameters:
        obj_pairs (list[tuple[str, str]]): Object category/id pairs.
        n_obj (int): Number of pickable objects to include.
        bowl_item (str): Bowl asset folder name.
        output_xml_path (str): Merged XML output path.
        objaverse_path (str): Root object asset directory.
        verbose (bool): If True, print parser/model information.

    Returns:
        MuJoCoParser: Parsed scene environment.
    """
    mjcf_files = [
        "../asset/robotis/robotis_ffw/ffw_sg2_site_added.xml",
        "../asset/object/table_0.xml",
        "../asset/floor/floor_simple_white.xml",
    ]
    for obj_name, item in obj_pairs[:int(n_obj)]:
        mjcf_files.append("%s/%s/%s/model_new.xml" % (objaverse_path, obj_name, item))
    mjcf_files.append("%s/bowl/%s/model_new.xml" % (objaverse_path, bowl_item))

    xml_path = merge_mjcfs(
        included_mjcf_files = mjcf_files,
        output_xml_path     = output_xml_path,
        memory              = "4G",
        noslip              = 3,
    )
    return MuJoCoParser(rel_xml_path=xml_path, verbose=verbose)


def get_sg2_joint_groups(env):
    """
    Group FFW-SG2 joint names into semantic control sets.

    Parameters:
        env: MuJoCo parser/environment with ``rev_pri_joint_names``.

    Returns:
        dict[str, list[str]]: Joint groups used by notebooks/controllers.
    """
    joint_groups = group_by_prefix(
        items       = env.rev_pri_joint_names,
        prefix_dict = {
            "arm_l_joint_names": ["arm_l_joint"],
            "arm_r_joint_names": ["arm_r_joint"],
            "gripper_l_joint_name": ["gripper_l_joint1"],
            "gripper_r_joint_name": ["gripper_r_joint1"],
            "head_pitch_joint_name": ["head_joint1"],
            "lift_joint_name": ["lift_joint"],
        },
    )
    joint_groups["use_arm_joint_names"] = joint_groups["arm_l_joint_names"] + joint_groups["arm_r_joint_names"]
    joint_groups["use_gripper_joint_names"] = joint_groups["gripper_l_joint_name"] + joint_groups["gripper_r_joint_name"]
    joint_groups["use_lift_joint_name"] = ["lift_joint"]
    joint_groups["use_upper_body_joint_names"] = joint_groups["use_arm_joint_names"] + joint_groups["use_lift_joint_name"]
    joint_groups["use_whole_joint_names"] = (
        joint_groups["use_upper_body_joint_names"]
        + joint_groups["use_gripper_joint_names"]
        + joint_groups["head_pitch_joint_name"]
    )
    return joint_groups


def get_sg2_init_q(env, head_lookat=np.array([0.80, 0.0, 0.7])):
    """
    Build the initial FFW-SG2 whole-body joint configuration.

    Parameters:
        env: Scene containing ``head_cam``.
        head_lookat (array-like): World-frame point for head pitch.

    Returns:
        list[float]: Initial q ordered as ``get_sg2_joint_groups()["use_whole_joint_names"]``.
    """
    head_lookat = np.asarray(head_lookat, dtype=np.float64).reshape(3)
    init_q_arm = [1.07, 0, 0, -2.57, 0, 0, 0] * 2
    init_q_grippers = [0, 0]
    init_q_lift = [-0.25]
    uv_head_to_lookat = np_uv(head_lookat - env.get_p(name="head_cam", type="body"))
    init_q_head_pitch = [np.arctan2(uv_head_to_lookat[0], uv_head_to_lookat[2])]
    return init_q_arm + init_q_lift + init_q_grippers + init_q_head_pitch


def arrange_pick_and_place_scene(
        env,
        obj_pairs,
        obj_body_names,
        bowl_body_name="body_obj_bowl_5",
        n_obj=None,
        obj_x_range=(0.60, 0.65),
        obj_y_range=(-0.20, 0.20),
        bowl_x_range=(0.80, 0.85),
        bowl_y_range=(-0.05, 0.05),
    ):
    """
    Place robot base, table, pickable objects, and bowl in the scene.

    The object and bowl positions are sampled in simple tabletop ranges. The
    function writes poses directly into MuJoCo and zeros velocities so the
    scene is ready for the viewer loop.

    Parameters:
        env: MuJoCo parser/environment.
        obj_pairs (list[tuple[str, str]]): Object category/id pairs.
        obj_body_names (list[str]): Body names aligned with ``obj_pairs``.
        bowl_body_name (str): Body name of the bowl object.
        n_obj (int | None): Number of objects to place. If None, all objects in
            ``obj_pairs`` are used.
        obj_x_range, obj_y_range: Sampling ranges for pickable objects.
        bowl_x_range, bowl_y_range: Sampling ranges for the bowl.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(obj_xyzs, bowl_xyz)``.
    """
    n_obj = len(obj_pairs) if n_obj is None else int(n_obj)
    obj_xyzs = sample_xyzs(
        n_sample = n_obj,
        x_range  = obj_x_range,
        y_range  = obj_y_range,
        z_range  = (0.7, 0.7),
        min_dist = 0.2,
    )
    bowl_xyz = sample_xyzs(
        n_sample = 1,
        x_range  = bowl_x_range,
        y_range  = bowl_y_range,
        z_range  = (0.7, 0.7),
    )[0]

    env.set_p(name="base_link", p=[0.2, 0, 0.0], type="base_body")
    env.set_p(name="body_object_table_0", p=[0.8, 0, 0], type="body")
    for obj_idx in range(n_obj):
        env.set_T(name=obj_body_names[obj_idx], type="base_body", T=pr2t(p=obj_xyzs[obj_idx, :]))
    env.set_p(name=bowl_body_name, type="base_body", p=bowl_xyz)
    env.set_zero_qvel()
    return obj_xyzs, bowl_xyz


def get_object_selection_options(env, obj_pairs, obj_body_names, reverse=True):
    """
    Build radio-button labels for current object instances.

    Objects are sorted by their current world-frame y position and assigned
    readable spatial labels such as ``left`` and ``right``. The returned mapping
    lets a notebook convert a selected radio-button label back into the object
    index used by ``obj_pairs`` and ``obj_body_names``.

    Parameters:
        env: MuJoCo parser/environment exposing ``get_p``.
        obj_pairs (list[tuple[str, str]]): Object category/id pairs.
        obj_body_names (list[str]): Body names aligned with ``obj_pairs``.
        reverse (bool): Sort convention for the y-axis. The notebooks use this
            to match the rendered camera view.

    Returns:
        tuple[list[str], dict[str, int]]: Labels and label-to-object-index map.
    """
    sorted_idxs = sorted(
        range(len(obj_pairs)),
        key=lambda idx: env.get_p(obj_body_names[idx], type="body")[1],
        reverse=reverse,
    )
    n_obj = len(sorted_idxs)
    if n_obj == 1:
        region_names = ["visible"]
    elif n_obj == 2:
        region_names = ["left", "right"]
    elif n_obj == 3:
        region_names = ["left", "middle", "right"]
    else:
        region_names = [
            "left" if rank == 0 else "right" if rank == n_obj - 1 else "%d-th" % (rank + 1)
            for rank in range(n_obj)
        ]

    labels = []
    label_to_idx = {}
    for rank, obj_idx in enumerate(sorted_idxs):
        obj_name, obj_id = obj_pairs[obj_idx]
        label = "%s: %s (%s)" % (region_names[rank], obj_name, obj_id)
        labels.append(label)
        label_to_idx[label] = obj_idx
    return labels, label_to_idx


def get_object_rank_among_same_category(env, obj_pairs, obj_body_names, obj_idx, reverse=True):
    """
    Return the selected object's left/right rank among objects of the same category.

    Parameters:
        env: MuJoCo parser/environment exposing ``get_p``.
        obj_pairs (list[tuple[str, str]]): Object category/id pairs.
        obj_body_names (list[str]): Body names aligned with ``obj_pairs``.
        obj_idx (int): Selected object index.
        reverse (bool): Same y-axis sorting convention as
            ``get_object_selection_options``.

    Returns:
        dict: ``rank``, ``count``, ``same_idxs``, and ``sorted_idxs``.
    """
    obj_idx = int(obj_idx)
    obj_name = obj_pairs[obj_idx][0]
    same_idxs = [idx for idx, pair in enumerate(obj_pairs) if pair[0] == obj_name]
    sorted_idxs = sorted(
        same_idxs,
        key=lambda idx: env.get_p(obj_body_names[idx], type="body")[1],
        reverse=reverse,
    )
    return dict(
        rank       = int(sorted_idxs.index(obj_idx)),
        count      = int(len(sorted_idxs)),
        same_idxs  = same_idxs,
        sorted_idxs = sorted_idxs,
        obj_name   = obj_name,
    )


# -----------------------------------------------------------------------------
# Camera / RGB-D
# -----------------------------------------------------------------------------

def compute_viewer_transform(env, p_ego, p_trgt, up_vector=np.array([0, 0, 1])):
    """
    Compute the world-frame viewer transform for an egocentric render.

    Parameters:
        env: MuJoCo parser/environment exposing ``compute_view_params``.
        p_ego (array-like): Camera position in world frame.
        p_trgt (array-like): Look-at target in world frame.
        up_vector (array-like): Viewer up direction.

    Returns:
        np.ndarray: Shape ``(4, 4)`` viewer transform.
    """
    cam_azimuth, cam_distance, cam_elevation, cam_lookat = env.compute_view_params(
        camera_pos = np.asarray(p_ego, dtype=np.float64).reshape(3),
        target_pos = np.asarray(p_trgt, dtype=np.float64).reshape(3),
        up_vector  = np.asarray(up_vector, dtype=np.float64).reshape(3),
    )
    R_lookat = rpy2r(np.deg2rad([0, -cam_elevation, cam_azimuth]))
    T_lookat = pr2t(np.asarray(cam_lookat, dtype=np.float64).reshape(3), R_lookat)
    return T_lookat @ pr2t(np.array([-cam_distance, 0, 0]), np.eye(3))


def capture_rgbd_observation(
        env,
        cam_name="head_cam",
        p_trgt=None,
        rsz_rate_for_img=1 / 4,
        restore_view=True,
        up_vector=np.array([0, 0, 1]),
    ):
    """
    Capture RGB-D from a MuJoCo viewer camera pose.

    ``MuJoCoParser`` renders RGB-D through the active viewer camera. This helper
    temporarily moves the viewer to the egocentric camera pose defined by
    ``cam_name`` and ``p_trgt``, captures RGB and depth, and then optionally
    restores the previous viewer camera. The returned ``T_cam`` is the body
    transform of ``cam_name`` before the temporary viewer move.

    Parameters:
        env: MuJoCo parser/environment with viewer and ``grab_rgbd_img``.
        cam_name (str): Body name used as the egocentric camera position.
        p_trgt (array-like | None): Optional look-at target. If None, the
            current viewer camera is used without changing view parameters.
        rsz_rate_for_img (float | None): Optional resize factor for RGB/depth.
        restore_view (bool): Restore the previous viewer camera after capture.
        up_vector (array-like): Up direction for ``compute_view_params``.

    Returns:
        dict: Image observation with ``rgb``, ``depth``, ``p_ego``, ``p_trgt``,
            ``T_cam``, and ``cam_name``.
    """
    T_cam = env.get_T(name=cam_name, type="body")
    p_ego = t2p(T_cam)
    p_trgt = None if p_trgt is None else np.asarray(p_trgt, dtype=np.float64).reshape(3)

    if restore_view:
        viewer_azimuth, viewer_distance, viewer_elevation, viewer_lookat = env.viewer.get_cam_info()

    if p_trgt is not None:
        cam_azimuth, cam_distance, cam_elevation, cam_lookat = env.compute_view_params(
            camera_pos = p_ego,
            target_pos = p_trgt,
            up_vector  = np.asarray(up_vector, dtype=np.float64).reshape(3),
        )
        env.viewer.set_cam_info(
            azimuth   = cam_azimuth,
            distance  = cam_distance,
            elevation = cam_elevation,
            lookat    = cam_lookat,
        )
        mujoco.mj_forward(env.model, env.data)
        mujoco.mjv_updateScene(
            env.model,
            env.data,
            env.viewer.vopt,
            env.viewer.pert,
            env.viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            env.viewer.scn,
        )
        mujoco.mjr_render(env.viewer.viewport, env.viewer.scn, env.viewer.ctx)

    rgb, depth = env.grab_rgbd_img()
    if rsz_rate_for_img is not None:
        h = int(rgb.shape[0] * float(rsz_rate_for_img))
        w = int(rgb.shape[1] * float(rsz_rate_for_img))
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    if restore_view:
        env.viewer.set_cam_info(
            azimuth   = viewer_azimuth,
            distance  = viewer_distance,
            elevation = viewer_elevation,
            lookat    = viewer_lookat,
        )
        mujoco.mj_forward(env.model, env.data)
        mujoco.mjv_updateScene(
            env.model,
            env.data,
            env.viewer.vopt,
            env.viewer.pert,
            env.viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            env.viewer.scn,
        )
        mujoco.mjr_render(env.viewer.viewport, env.viewer.scn, env.viewer.ctx)

    return dict(
        rgb      = rgb,
        depth    = depth,
        p_ego    = p_ego,
        p_trgt   = p_trgt,
        T_cam    = T_cam,
        cam_name = str(cam_name),
    )


# -----------------------------------------------------------------------------
# Point cloud
# -----------------------------------------------------------------------------

def estimate_grasp_center(
        points,
        height_ratio=0.50,
        lower_percentile=10.0,
        upper_percentile=90.0,
    ):
    """
    Estimate a size-aware side-grasp point from a world-frame point cloud.

    The horizontal coordinates use the midpoint of robust point-cloud bounds.
    This is less biased by unevenly sampled visible surfaces, so bowl placement
    targets stay closer to the geometric middle. The vertical grasp coordinate
    is placed at ``height_ratio`` between robust lower and upper z bounds, so
    the grasp height scales with the observed object height instead of relying
    on a fixed offset.

    Parameters:
        points (array-like): Shape ``(N, 3)`` point cloud.
        height_ratio (float): Relative grasp height from lower to upper bound.
        lower_percentile, upper_percentile (float): Robust bounds used to
            reject sparse depth and segmentation outliers.

    Returns:
        dict: ``center``, ``grasp_center``, ``extent``, ``lower_bound``, and
            ``upper_bound``. Values are None when no finite points exist.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return dict(
            center=None,
            grasp_center=None,
            extent=None,
            lower_bound=None,
            upper_bound=None,
        )

    lower_percentile = float(lower_percentile)
    upper_percentile = float(upper_percentile)
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Expected 0 <= lower_percentile < upper_percentile <= 100.")

    lower_bound, upper_bound = np.percentile(
        points,
        [lower_percentile, upper_percentile],
        axis=0,
    )
    inlier_mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
    inliers = points[inlier_mask]
    if inliers.shape[0] < 4:
        inliers = points

    center = np.median(inliers, axis=0)
    center[:2] = 0.5 * (lower_bound[:2] + upper_bound[:2])
    grasp_center = center.copy()
    grasp_center[2] = lower_bound[2] + np.clip(float(height_ratio), 0.0, 1.0) * (
        upper_bound[2] - lower_bound[2]
    )
    return dict(
        center=center,
        grasp_center=grasp_center,
        extent=upper_bound - lower_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )


def build_point_cloud_observation(
        env,
        depth,
        mask=None,
        p_ego=None,
        p_trgt=None,
        fovy=45,
        rsz_rate_for_pcd=1 / 2,
        to_world=True,
        valid_depth_threshold=1e-8,
        grasp_height_ratio=0.55,
    ):
    """
    Convert a depth image, and optionally a mask, into point-cloud arrays.

    The full depth image is converted into a dense point cloud. If ``mask`` is
    provided, the mask is resized to the point-cloud depth resolution and used
    to extract object-specific points and their mean center. When a downsampled
    mask loses all valid pixels, the function retries once at full resolution.

    Parameters:
        env: MuJoCo parser/environment exposing ``get_pcd_from_depth_img``.
        depth (np.ndarray): Depth image in meters.
        mask (np.ndarray | None): Optional binary mask aligned with ``depth``.
        p_ego (array-like | None): Camera position used for world transform.
        p_trgt (array-like | None): Look-at target used for world transform.
        fovy (float | None): Camera field of view in degrees. The default
            ``45`` matches the older ``get_pcd_from_mask`` helper used by
            ``notebook/10_interactive_pick_and_place.ipynb``. Pass None to use
            the first MuJoCo camera fovy when available.
        rsz_rate_for_pcd (float | None): Optional depth downsampling factor.
        to_world (bool): If True, return selected points in world frame.
        valid_depth_threshold (float): Minimum valid depth.
        grasp_height_ratio (float): Relative height used for ``grasp_center``.

    Returns:
        dict: Point-cloud observation with ``full``, ``masked``, ``center``,
            size-aware ``grasp_center``, ``extent``, ``xyz_img``,
            ``xyz_img_world``, and ``mask``.
    """
    if rsz_rate_for_pcd is not None:
        h_rsz = int(depth.shape[0] * float(rsz_rate_for_pcd))
        w_rsz = int(depth.shape[1] * float(rsz_rate_for_pcd))
        depth_rsz = cv2.resize(depth, (w_rsz, h_rsz), interpolation=cv2.INTER_NEAREST)
    else:
        depth_rsz = depth

    if fovy is None:
        fovy = 45.0 if len(env.model.cam_fovy) == 0 else float(env.model.cam_fovy[0])
    _, xyz_img, _ = env.get_pcd_from_depth_img(depth_rsz, fovy=fovy)

    if p_ego is not None and p_trgt is not None:
        T_viewer = compute_viewer_transform(env=env, p_ego=p_ego, p_trgt=p_trgt)
        xyz_flat = xyz_img.reshape(-1, 3)
        xyz_h = np.concatenate([xyz_flat, np.ones((xyz_flat.shape[0], 1), dtype=xyz_flat.dtype)], axis=1)
        xyz_world = (T_viewer @ xyz_h.T).T[:, :3]
        xyz_img_world = xyz_world.reshape(xyz_img.shape[0], xyz_img.shape[1], 3)
    else:
        xyz_img_world = xyz_img.copy()

    xyz_src = xyz_img_world if to_world else xyz_img
    full_valid = (depth_rsz > valid_depth_threshold)
    full_valid &= np.isfinite(xyz_src).all(axis=2)
    full_pcd = xyz_src[full_valid]

    mask_rsz = None
    masked_pcd = None
    center = None
    grasp_center = None
    extent = None
    lower_bound = None
    upper_bound = None
    valid_mask_count = 0
    mask_area = 0
    if mask is not None:
        if mask.shape != depth_rsz.shape:
            mask_rsz = cv2.resize(
                mask.astype(np.float32),
                (depth_rsz.shape[1], depth_rsz.shape[0]),
                interpolation=cv2.INTER_AREA,
            ) > 0.0
        else:
            mask_rsz = mask.astype(bool)
        mask_area = int(mask_rsz.sum())
        valid_mask = mask_rsz & full_valid
        valid_mask_count = int(valid_mask.sum())
        masked_pcd = xyz_src[valid_mask]

        # ------------------------------------------------------------------
        # FILL IN: Estimate a 3D target point from the masked point cloud.
        #
        # Context:
        #   ``masked_pcd`` contains the valid 3D points whose pixels belong to
        #   the segmentation mask. For pick-and-place, the controller expects a
        #   compact target point such as the mean, median, bbox center, top
        #   point, or another task-specific representative.
        #
        # Things to think about:
        #   - What should happen when the mask has no valid depth points?
        #   - Is the mean always a good pick point for every object shape?
        #   - Would a robust statistic, filtered point cloud, or height-aware
        #     target improve stability?
        #
        # Expected output:
        #   Set ``center`` to either a shape-(3,) numpy array or ``None``.
        # ------------------------------------------------------------------
        if masked_pcd.shape[0] > 0:
            grasp_info = estimate_grasp_center(
                points=masked_pcd,
                height_ratio=grasp_height_ratio,
            )
            center = grasp_info["center"]
            grasp_center = grasp_info["grasp_center"]
            extent = grasp_info["extent"]
            lower_bound = grasp_info["lower_bound"]
            upper_bound = grasp_info["upper_bound"]

        if center is None and rsz_rate_for_pcd is not None and int(np.asarray(mask).astype(bool).sum()) > 0:
            return build_point_cloud_observation(
                env                   = env,
                depth                 = depth,
                mask                  = mask,
                p_ego                 = p_ego,
                p_trgt                = p_trgt,
                fovy                  = fovy,
                rsz_rate_for_pcd      = None,
                to_world              = to_world,
                valid_depth_threshold = valid_depth_threshold,
                grasp_height_ratio    = grasp_height_ratio,
            )

    return dict(
        full          = full_pcd,
        masked        = masked_pcd,
        center        = center,
        grasp_center  = grasp_center,
        extent        = extent,
        lower_bound   = lower_bound,
        upper_bound   = upper_bound,
        xyz_img       = xyz_img,
        xyz_img_world = xyz_img_world,
        mask          = mask_rsz,
        fovy          = float(fovy),
        to_world      = bool(to_world),
        debug         = dict(
            depth_shape       = tuple(depth.shape),
            depth_rsz_shape   = tuple(depth_rsz.shape),
            depth_min         = float(np.nanmin(depth_rsz)) if depth_rsz.size > 0 else None,
            depth_max         = float(np.nanmax(depth_rsz)) if depth_rsz.size > 0 else None,
            full_valid_count  = int(full_valid.sum()),
            mask_is_none      = mask is None,
            mask_original_area = 0 if mask is None else int(np.asarray(mask).astype(bool).sum()),
            mask_area         = int(mask_area),
            valid_mask_count  = int(valid_mask_count),
            masked_pcd_count  = 0 if masked_pcd is None else int(len(masked_pcd)),
            grasp_height_ratio = float(grasp_height_ratio),
        ),
    )


# -----------------------------------------------------------------------------
# Robot / object state
# -----------------------------------------------------------------------------

def collect_joint_state(env, joint_names=None, include_full=True):
    """
    Collect joint state from a MuJoCo environment.

    Parameters:
        env: MuJoCo parser/environment exposing ``get_qpos``.
        joint_names (list[str] | None): Optional named joints to query.
        include_full (bool): Include full ``qpos`` and ``qvel`` arrays.

    Returns:
        dict: Joint state dictionary.
    """
    state = dict(joint_names=None, q=None)
    if joint_names is not None:
        state["joint_names"] = list(joint_names)
        state["q"] = env.get_qpos(joint_names=joint_names)
    if include_full:
        state["qpos"] = env.data.qpos.copy()
        state["qvel"] = env.data.qvel.copy()
    return state


def collect_body_state(env, body_names=None):
    """
    Collect body pose state from a MuJoCo environment.

    Parameters:
        env: MuJoCo parser/environment exposing ``get_p`` and ``get_T``.
        body_names (list[str] | None): Body names to query.

    Returns:
        dict: Mapping from body name to position and transform.
    """
    if body_names is None:
        return {}
    body_state = {}
    for body_name in body_names:
        body_state[str(body_name)] = dict(
            p = env.get_p(name=body_name, type="body"),
            T = env.get_T(name=body_name, type="body"),
        )
    return body_state


# -----------------------------------------------------------------------------
# OWLv2 + SAM2 segmentation
# -----------------------------------------------------------------------------

def make_owlv2_sam2_segmenter(
        owl_ckpt="google/owlv2-base-patch16",
        sam2_ckpt="facebook/sam2.1-hiera-base-plus",
        owl_local_dir="../hf_models/owlv2",
        sam2_local_dir="../hf_models/sam2",
        device=None,
        verbose=True,
    ):
    """
    Build the local OWLv2 + SAM2 text-prompt segmentation backend.

    OWLv2 proposes text-conditioned bounding boxes. SAM2 then segments the
    selected image regions from point/bbox prompts. The returned object exposes
    a ``run`` method used by ``run_text_segmentation``.

    Returns:
        PromptOwlv2Sam2: Segmenter with a ``run(img_pil, text_queries, ...)`` method.
    """
    from package.hf_models.hf_vision import PointSam2, PromptOwlv2, PromptOwlv2Sam2

    owl = PromptOwlv2(
        ckpt      = owl_ckpt,
        local_dir = owl_local_dir,
        device    = device,
        verbose   = verbose,
    )
    sam2 = PointSam2(
        ckpt      = sam2_ckpt,
        local_dir = sam2_local_dir,
        device    = device,
        verbose   = verbose,
    )
    return PromptOwlv2Sam2(owl=owl, sam2=sam2, verbose=verbose)


def run_text_segmentation(segmenter, rgb, text_queries, **segmentation_kwargs):
    """
    Run text-prompt segmentation and return normalized candidate results.

    This is the low-level segmentation helper. It prepares text aliases, tries
    the configured detection threshold and optional fallback thresholds, clips
    masks to their detection boxes, applies bbox fallback for empty SAM2 masks,
    deduplicates near-identical boxes, and optionally plots the result.

    Most notebooks should call ``run_prompt_to_segmentation`` instead, because
    that helper also selects the single result to use for pick/place.

    Parameters:
        segmenter: Callable, or object with ``run``.
        rgb (np.ndarray): RGB image.
        text_queries (str | list[str]): Text query or queries.
        **segmentation_kwargs: Backend-specific segmentation kwargs.

    Returns:
        list[dict]: Segmentation result dictionaries.
    """
    from PIL import Image

    plot_seg = bool(segmentation_kwargs.pop("plot_seg", False))
    plot_seg = bool(segmentation_kwargs.pop("plot_det_seg", plot_seg))
    clip_mask_to_box = bool(segmentation_kwargs.pop("clip_mask_to_box", True))
    box_clip_margin = int(segmentation_kwargs.pop("box_clip_margin", 4))
    bbox_mask_fallback = bool(segmentation_kwargs.pop("bbox_mask_fallback", True))
    fallback_det_thresholds = segmentation_kwargs.pop("fallback_det_thresholds", None)
    use_text_aliases = bool(segmentation_kwargs.pop("use_text_aliases", True))
    if "use_bbox_neg_points" in segmentation_kwargs and "use_neg_points" not in segmentation_kwargs:
        segmentation_kwargs["use_neg_points"] = segmentation_kwargs.pop("use_bbox_neg_points")
    plot_kwargs = dict(
        fig_w = segmentation_kwargs.pop("plot_fig_w", 4),
        overlay_bright = segmentation_kwargs.pop("plot_overlay_bright", 0.8),
    )

    img_np = np.asarray(rgb).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    text_queries_use = make_owlv2_text_queries(text_queries) if use_text_aliases else text_queries
    det_threshold = float(segmentation_kwargs.get("det_threshold", 0.1))
    thresholds = [det_threshold]
    if fallback_det_thresholds is not None:
        thresholds.extend([float(th) for th in fallback_det_thresholds])
    thresholds = list(dict.fromkeys(thresholds))

    results = []
    for threshold in thresholds:
        run_kwargs = dict(segmentation_kwargs)
        run_kwargs["det_threshold"] = threshold
        if hasattr(segmenter, "run"):
            raw_results = segmenter.run(img_pil, text_queries_use, **run_kwargs)
        else:
            raw_results = segmenter(img_pil, text_queries_use, **run_kwargs)
        results = normalize_owlv2_sam2_results(
            raw_results       = raw_results,
            clip_to_box       = clip_mask_to_box,
            box_clip_margin   = box_clip_margin,
            bbox_mask_fallback = bbox_mask_fallback,
            image_shape       = img_np.shape[:2],
            det_threshold_used = threshold,
            text_queries_used = text_queries_use,
        )
        if len(results) > 0:
            break

    if plot_seg and hasattr(segmenter, "plot_det_seg"):
        segmenter.plot_det_seg(img_np=img_np, results=results, **plot_kwargs)
    return results


def run_prompt_to_segmentation(
        segmenter,
        rgb,
        prompt=None,
        text_queries=None,
        env=None,
        obj_pairs=None,
        obj_body_names=None,
        obj_idx=None,
        rank_info=None,
        same_category_rank=True,
        image_rank=None,
        image_rank_reverse=False,
        return_all=False,
        **segmentation_kwargs,
    ):
    """
    Run prompt-to-segmentation and return one selected result.

    This is the notebook-facing helper for interactive pick and place. It
    converts a text prompt into OWLv2+SAM2 candidates, optionally expands the
    number of returned boxes when the selected object has same-category
    neighbors, and selects the candidate whose image left/right rank matches
    the radio-button object choice.

    If same-category ranking metadata is provided, the function automatically
    raises ``topk_per_query`` so multiple boxes can be considered. Otherwise it
    selects the highest-scoring result. The selected result includes a
    ``candidates`` field containing the raw candidate list for plotting or
    debugging.

    Parameters:
        segmenter: Callable, or object with ``run``.
        rgb (np.ndarray): RGB image.
        prompt (str | list[str] | None): Text prompt. Used when
            ``text_queries`` is None.
        text_queries (str | list[str] | None): Explicit backend query.
        env: Optional MuJoCo environment used to rank same-category objects.
        obj_pairs (list[tuple[str, str]] | None): Object category/id pairs.
        obj_body_names (list[str] | None): Body names aligned with
            ``obj_pairs``.
        obj_idx (int | None): Selected object index from the radio button.
        rank_info (dict | None): Precomputed output from
            ``get_object_rank_among_same_category``.
        same_category_rank (bool): If True and object metadata is available,
            select by image left/right rank among same-category candidates.
        image_rank (int | None): Manual image-rank override.
        image_rank_reverse (bool): If True, rank from image-right to image-left.
        return_all (bool): If True, return ``(selected, results)``.
        **segmentation_kwargs: Forwarded to ``run_text_segmentation``.

    Returns:
        dict | None: Selected segmentation result. The returned dict includes
            ``candidates`` and optional ``rank_info`` metadata.
    """
    text_queries_use = text_queries if text_queries is not None else prompt
    if text_queries_use is None:
        raise ValueError("Either prompt or text_queries is required.")

    seg_kwargs = dict(segmentation_kwargs)
    rank_info_use = None if rank_info is None else dict(rank_info)
    if bool(same_category_rank) and rank_info_use is None:
        has_rank_inputs = env is not None and obj_pairs is not None and obj_body_names is not None and obj_idx is not None
        if has_rank_inputs:
            rank_info_use = get_object_rank_among_same_category(
                env            = env,
                obj_pairs      = obj_pairs,
                obj_body_names = obj_body_names,
                obj_idx        = obj_idx,
                reverse        = True,
            )

    if rank_info_use is not None:
        count = int(rank_info_use.get("count", 1))
        seg_kwargs["topk_per_query"] = max(int(seg_kwargs.get("topk_per_query", 1)), count)

    results = run_text_segmentation(
        segmenter    = segmenter,
        rgb          = rgb,
        text_queries = text_queries_use,
        **seg_kwargs,
    )

    selected = None
    if image_rank is not None:
        selected = select_segmentation_result_by_image_rank(
            results = results,
            rank    = image_rank,
            reverse = image_rank_reverse,
        )
    elif rank_info_use is not None and int(rank_info_use.get("count", 1)) > 1:
        selected = select_segmentation_result_by_image_rank(
            results = results,
            rank    = int(rank_info_use.get("rank", 0)),
            reverse = image_rank_reverse,
        )
    elif len(results) > 0:
        selected = dict(sorted(results, key=lambda item: float(item.get("score", 0.0)), reverse=True)[0])

    if selected is not None:
        selected["candidates"] = results
        selected["rank_info"] = rank_info_use
        selected["prompt"] = text_queries_use
    if return_all:
        return selected, results
    return selected


def make_owlv2_text_queries(text_queries):
    """
    Expand internal object names into OWLv2-friendly natural-language queries.

    Parameters:
        text_queries (str | list[str]): Raw query or query list.

    Returns:
        list[str]: Deduplicated natural-language query aliases.
    """
    if isinstance(text_queries, str):
        raw_queries = [text_queries]
    else:
        raw_queries = [str(q) for q in text_queries]

    alias_map = {
        "bottled_water": ["water bottle", "bottle", "plastic bottle", "bottled water"],
        "water_bottle": ["water bottle", "bottle", "plastic bottle"],
        "can": ["can", "soda can"],
        "bowl": ["gray bowl", "empty bowl", "bowl"],
        "round bowl": ["gray bowl", "empty bowl", "bowl"],
    }
    queries = []
    for query in raw_queries:
        query_clean = str(query).strip()
        if query_clean == "":
            continue
        aliases = alias_map.get(query_clean, [query_clean.replace("_", " "), query_clean])
        for alias in aliases:
            if alias not in queries:
                queries.append(alias)
    return queries


def clip_mask_to_box(mask, box, margin=0):
    """
    Clip a segmentation mask to a bbox with optional pixel margin.

    Parameters:
        mask (np.ndarray): Binary mask.
        box (array-like): ``[x1, y1, x2, y2]`` bbox.
        margin (int): Pixel margin around the bbox.

    Returns:
        np.ndarray: Clipped binary mask.
    """
    mask = np.asarray(mask).astype(bool)
    box = np.asarray(box, dtype=np.float64).reshape(-1)
    if box.size < 4:
        return mask
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = box[:4]
    m = int(max(0, margin))
    x1 = int(np.clip(np.floor(x1) - m, 0, w))
    y1 = int(np.clip(np.floor(y1) - m, 0, h))
    x2 = int(np.clip(np.ceil(x2) + m, 0, w))
    y2 = int(np.clip(np.ceil(y2) + m, 0, h))
    clipped = np.zeros_like(mask, dtype=bool)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def bbox_to_mask(image_shape, box, margin=0):
    """
    Convert a bbox into a binary mask.

    Parameters:
        image_shape (tuple[int, int]): ``(H, W)`` image shape.
        box (array-like): ``[x1, y1, x2, y2]`` bbox.
        margin (int): Pixel margin around the bbox.

    Returns:
        np.ndarray: Binary bbox mask.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    box = np.asarray(box, dtype=np.float64).reshape(-1)
    mask = np.zeros((h, w), dtype=bool)
    if box.size < 4:
        return mask
    x1, y1, x2, y2 = box[:4]
    m = int(margin)
    x1 = int(np.clip(np.floor(x1) - m, 0, w))
    y1 = int(np.clip(np.floor(y1) - m, 0, h))
    x2 = int(np.clip(np.ceil(x2) + m, 0, w))
    y2 = int(np.clip(np.ceil(y2) + m, 0, h))
    mask[y1:y2, x1:x2] = True
    return mask


def normalize_owlv2_sam2_results(
        raw_results,
        clip_to_box=True,
        box_clip_margin=4,
        bbox_mask_fallback=True,
        image_shape=None,
        det_threshold_used=None,
        text_queries_used=None,
    ):
    """
    Normalize OWLv2+SAM2 result dictionaries.

    Parameters:
        raw_results: Raw segmenter output.
        clip_to_box (bool): Clip masks to each selected bbox.
        box_clip_margin (int): Pixel margin for clipping.
        bbox_mask_fallback (bool): Use bbox rectangle when SAM2 returns an
            empty mask.
        image_shape (tuple[int, int] | None): Required for bbox fallback.
        det_threshold_used (float | None): Detection threshold metadata.
        text_queries_used (list[str] | None): Query metadata.

    Returns:
        list[dict]: Valid result dictionaries sorted by descending score.
    """
    if raw_results is None:
        return []
    if isinstance(raw_results, dict):
        raw_results = [raw_results]

    out = []
    for result in list(raw_results):
        if not isinstance(result, dict) or result.get("mask", None) is None:
            continue
        item = dict(result)
        item["mask"] = np.asarray(item["mask"]).astype(bool)
        item["sam_mask_area"] = int(item["mask"].sum())
        if bool(clip_to_box) and item.get("box", None) is not None:
            item["mask"] = clip_mask_to_box(item["mask"], item["box"], margin=box_clip_margin)
        item["clipped_mask_area"] = int(item["mask"].sum())
        if int(item["mask"].sum()) == 0 and bool(bbox_mask_fallback) and item.get("box", None) is not None and image_shape is not None:
            item["mask"] = bbox_to_mask(image_shape=image_shape, box=item["box"], margin=0)
            item["clipped_mask_area"] = int(item["mask"].sum())
            item["mask_source"] = "bbox_fallback"
        else:
            item["mask_source"] = "sam2"
        if det_threshold_used is not None:
            item["det_threshold_used"] = float(det_threshold_used)
        if text_queries_used is not None:
            item["text_queries_used"] = list(text_queries_used)
        out.append(item)
    out = sorted(out, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return dedupe_segmentation_results_by_box(out)


def get_result_box_center_x(result):
    """
    Return bbox center x for a segmentation/detection result.

    Parameters:
        result (dict): Result containing ``box``.

    Returns:
        float: Bbox center x, or infinity if no valid box exists.
    """
    box = np.asarray(result.get("box", []), dtype=np.float64).reshape(-1)
    if box.size < 4:
        return np.inf
    return float(0.5 * (box[0] + box[2]))


def result_box_iou(a, b):
    """
    Compute IoU between two result boxes.

    Parameters:
        a, b (dict): Results containing ``box``.

    Returns:
        float: Intersection-over-union.
    """
    box_a = np.asarray(a.get("box", []), dtype=np.float64).reshape(-1)
    box_b = np.asarray(b.get("box", []), dtype=np.float64).reshape(-1)
    if box_a.size < 4 or box_b.size < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else float(inter / denom)


def dedupe_segmentation_results_by_box(results, iou_thr=0.7):
    """
    Remove near-duplicate detection results by bbox IoU.

    Parameters:
        results (list[dict]): Result dictionaries sorted by preference.
        iou_thr (float): IoU threshold for duplicate boxes.

    Returns:
        list[dict]: Deduplicated results.
    """
    deduped = []
    for result in results:
        if all(result_box_iou(result, kept) < float(iou_thr) for kept in deduped):
            deduped.append(result)
    return deduped


def select_segmentation_result_by_image_rank(results, rank=0, reverse=False):
    """
    Select a detection/segmentation result by image left/right rank.

    Parameters:
        results (list[dict]): Candidate result dictionaries containing boxes.
        rank (int): Zero-based rank. ``0`` means image-leftmost when
            ``reverse=False``.
        reverse (bool): If True, rank from image-right to image-left.

    Returns:
        dict | None: Selected result, or None when no candidates exist.
    """
    candidates = [r for r in results if isinstance(r, dict) and r.get("box", None) is not None]
    candidates = sorted(candidates, key=get_result_box_center_x, reverse=bool(reverse))
    if len(candidates) == 0:
        return None
    rank = int(np.clip(int(rank), 0, len(candidates) - 1))
    selected = dict(candidates[rank])
    selected["image_rank"] = rank
    selected["image_rank_count"] = len(candidates)
    selected["image_rank_reverse"] = bool(reverse)
    return selected


# -----------------------------------------------------------------------------
# Pick-and-place execution helpers
# -----------------------------------------------------------------------------

def select_arm_by_position(env, p_trgt, left_site="gripper_l_tcp", right_site="gripper_r_tcp"):
    """
    Select the arm whose TCP is closer to a world-frame target.

    Parameters:
        env: MuJoCo parser/environment exposing ``get_p``.
        p_trgt (array-like): World-frame target position.
        left_site (str): Left TCP body/site name.
        right_site (str): Right TCP body/site name.

    Returns:
        str: ``"l"`` or ``"r"``.
    """
    p_trgt = np.asarray(p_trgt, dtype=np.float64).reshape(3)
    dist_l = np.linalg.norm(env.get_p(left_site, type="body") - p_trgt)
    dist_r = np.linalg.norm(env.get_p(right_site, type="body") - p_trgt)
    return "l" if dist_l < dist_r else "r"


def get_grasp_offsets(pitch_deg=0.0, axis_len=0.1):
    """
    Create guide-point offsets around a gripper TCP.

    FILL IN:
        Return a dictionary of local guide-point offsets around the active TCP.
        The downstream IK code can use these guide points to constrain gripper
        orientation with site-position targets.

    Things to think about:
        - A TCP-only target constrains position but leaves orientation weakly
          constrained.
        - Additional points such as ``top``, ``side``, and ``front`` can encode
          a small local coordinate frame around the gripper.
        - ``pitch_deg`` can be used to tilt this local frame for a different
          grasp approach.

    Parameters:
        pitch_deg (float): Local y-axis pitch angle in degrees.
        axis_len (float): Distance from TCP to guide points.

    Returns:
        dict[str, np.ndarray]: Offsets for ``tcp``, ``top``, ``side``, and
            ``front``.
    """
    axis_len = float(axis_len)
    pitch = float(pitch_deg) * D2R

    # Match the R_tcp convention used in notebooks 03 and 04: the target
    # rotation columns point toward front, side, and top, respectively.
    R = rpy2r(np.array([0.0, pitch, 0.0], dtype=np.float64))
    return {
        "tcp":   np.zeros(3, dtype=np.float64),
        "front": axis_len * R[:, 0],
        "side":  axis_len * R[:, 1],
        "top":   axis_len * R[:, 2],
    }


def make_pick_place_phase_targets(
        pick_position,
        place_position,
        grasp_offsets=None,
        pick_tcp_offset=np.array([0.0, 0.0, 0.0]),
        pick_lower_bound=None,
        pick_upper_bound=None,
        surface_clearance=0.025,
        grasp_depth_ratio=0.50,
        approach_offset=np.array([-0.10, 0.0, 0.0]),
        lift_offset=np.array([0.0, 0.0, 0.30]),
        place_hover_offset=np.array([0.0, 0.0, 0.30]),
        place_tcp_offset=np.array([0.0, 0.0, 0.15]),
        retreat_offset=np.array([0.0, 0.0, 0.50]),
    ):
    """
    Build phase targets for the pick-and-place state machine.

    Each phase target is a dictionary of guide points around the active TCP:
    ``tcp``, ``top``, ``side``, and ``front``. These guide points are passed to
    the IK API so the gripper position and orientation are constrained by
    simple site-position targets.

    FILL IN:
        Convert the raw pick/place points into a sequence of phase targets.
        A direct jump from pick to place is usually brittle. A practical
        sequence often includes approach, pregrasp, lift, place approach,
        place, and retreat targets.

    Things to think about:
        - Where should the robot move before closing the gripper?
        - How high should it lift after grasping?
        - Where should it hover before placing?
        - After release, where should the gripper retreat?
        - Should each phase preserve the same gripper guide-point offsets?

    Parameters:
        pick_position (array-like): Perceived pick point in world frame.
        place_position (array-like): Place object/region center in world frame.
        grasp_offsets (dict[str, array-like] | None): Guide-point offsets.
        pick_tcp_offset (array-like): World-frame correction applied to the
            perceived pick point before building pick-side phase targets.
        pick_lower_bound, pick_upper_bound (array-like | None): Robust masked
            point-cloud bounds. When provided, ``pregrasp`` stops before the
            near surface and ``grasp_pose`` advances into the object more
            gently.
        surface_clearance (float): Distance to stop before the near surface.
        grasp_depth_ratio (float): Fraction of the point-cloud thickness to
            enter from the near surface for the final grasp pose.
        approach_offset, lift_offset, place_hover_offset, place_tcp_offset,
            retreat_offset (array-like): World-frame phase offsets.

    Returns:
        dict[str, dict[str, np.ndarray]]: Phase targets keyed by phase name.
    """
    pick_position = np.asarray(pick_position, dtype=np.float64).reshape(3)
    place_position = np.asarray(place_position, dtype=np.float64).reshape(3)
    if grasp_offsets is None:
        grasp_offsets = get_grasp_offsets()

    grasp_offsets = {
        name: np.asarray(offset, dtype=np.float64).reshape(3)
        for name, offset in dict(grasp_offsets).items()
    }

    pick_tcp_offset = np.asarray(pick_tcp_offset, dtype=np.float64).reshape(3)
    approach_offset = np.asarray(approach_offset, dtype=np.float64).reshape(3)
    lift_offset = np.asarray(lift_offset, dtype=np.float64).reshape(3)
    place_hover_offset = np.asarray(place_hover_offset, dtype=np.float64).reshape(3)
    place_tcp_offset = np.asarray(place_tcp_offset, dtype=np.float64).reshape(3)
    retreat_offset = np.asarray(retreat_offset, dtype=np.float64).reshape(3)

    def make_target_dict(tcp_position):
        tcp_position = np.asarray(tcp_position, dtype=np.float64).reshape(3)
        return {
            name: tcp_position + offset
            for name, offset in grasp_offsets.items()
        }

    pick_tcp_position = pick_position + pick_tcp_offset
    pregrasp_tcp_position = pick_tcp_position.copy()
    grasp_tcp_position = pick_tcp_position.copy()

    if pick_lower_bound is not None and pick_upper_bound is not None and np.linalg.norm(approach_offset) > 1e-8:
        lower = np.asarray(pick_lower_bound, dtype=np.float64).reshape(3)
        upper = np.asarray(pick_upper_bound, dtype=np.float64).reshape(3)
        if np.isfinite(lower).all() and np.isfinite(upper).all():
            insertion_dir = -approach_offset / np.linalg.norm(approach_offset)
            corners = np.array([
                [x, y, z]
                for x in [lower[0], upper[0]]
                for y in [lower[1], upper[1]]
                for z in [lower[2], upper[2]]
            ], dtype=np.float64)
            projections = corners @ insertion_dir
            near_proj = float(np.min(projections))
            far_proj = float(np.max(projections))
            depth = max(0.0, far_proj - near_proj)

            current_proj = float(pick_tcp_position @ insertion_dir)
            pregrasp_proj = near_proj - max(0.0, float(surface_clearance))
            grasp_proj = near_proj + np.clip(float(grasp_depth_ratio), 0.0, 1.0) * depth
            pregrasp_tcp_position = pick_tcp_position + (pregrasp_proj - current_proj) * insertion_dir
            grasp_tcp_position = pick_tcp_position + (grasp_proj - current_proj) * insertion_dir

    return {
        "approach":       make_target_dict(pick_tcp_position + approach_offset),
        "pregrasp":       make_target_dict(pregrasp_tcp_position),
        "grasp_pose":     make_target_dict(grasp_tcp_position),
        "lift":           make_target_dict(grasp_tcp_position + lift_offset),
        "place_approach": make_target_dict(place_position + place_hover_offset),
        "place":          make_target_dict(place_position + place_tcp_offset),
        "retreat":        make_target_dict(place_position + retreat_offset),
    }


def get_gripper_ctrl(use_arm, mode):
    """
    Build a two-gripper command.

    FILL IN:
        Return actuator commands for the left and right grippers. The command
        should depend on which arm is active and whether the requested mode is
        ``open`` or ``close``.

    Things to think about:
        - The controller always sends commands for both grippers.
        - Opening usually means both grippers should be open.
        - Closing should only close the active gripper, while the other stays
          open.

    Parameters:
        use_arm (str): ``"l"`` or ``"r"``.
        mode (str): ``"open"`` or ``"close"``.

    Returns:
        np.ndarray: Shape ``(2,)`` gripper command.
    """
    use_arm = str(use_arm).lower()
    mode = str(mode).lower()
    if use_arm not in ["l", "r"]:
        raise ValueError("use_arm should be 'l' or 'r', got: %s" % str(use_arm))
    if mode not in ["open", "close"]:
        raise ValueError("mode should be 'open' or 'close', got: %s" % str(mode))

    ctrl = np.zeros(2, dtype=np.float64)
    if mode == "close":
        ctrl[0 if use_arm == "l" else 1] = 1.0
    return ctrl


def attach_body_to_body(
        env,
        obj_body_name,
        body_name,
        T_body_obj,
        obj_type="base_body",
        body_type="body",
        zero_obj_qvel=True,
    ):
    """
    Move an object body so it stays fixed relative to another body.

    This is a kinematic attachment helper used after grasping. It computes the
    object's world pose from a stored object-in-TCP transform and the current
    TCP pose, then writes that pose back to the object's free body.

    Parameters:
        env: MuJoCo parser/environment.
        obj_body_name (str): Object body to move.
        body_name (str): Parent body to follow.
        T_body_obj (np.ndarray): Object pose expressed in parent-body frame.
        obj_type (str): Pose setter type for ``obj_body_name``.
        body_type (str): Pose getter type for ``body_name``.
        zero_obj_qvel (bool): Zero object free-joint velocity when possible.

    Returns:
        None.
    """
    T_w_body = env.get_T(body_name, type=body_type)
    T_w_obj = view_in_world(T=T_body_obj, T_wl=T_w_body)
    env.set_T(obj_body_name, type=obj_type, T=T_w_obj)
    if bool(zero_obj_qvel) and obj_type in ["base", "base_body"]:
        try:
            jntadr = env.model.body(obj_body_name).jntadr[0]
            dofadr = env.model.jnt_dofadr[jntadr]
            env.data.qvel[dofadr:dofadr + 6] = 0.0
        except Exception:
            pass


def sample_obj_pairs(
        root_dir,
        obj_names=None,
        n_sample=None,
        seed=None,
        replace=False,
        sort_pairs=True,
        verbose=False,
    ):
    """
    Collect (obj_name, item) pairs under an Objaverse-style object directory.

    Expected directory structure:
        root_dir/
            obj_name/
                item/
                item/

    Parameters:
        root_dir (str | Path): Root directory containing category folders.
        obj_names (list[str] | None): Optional category filter. If None, all
            category directories under ``root_dir`` are considered.
        n_sample (int | None): Optional number of pairs to sample.
        seed (int | None): Random seed used when sampling.
        replace (bool): Whether sampling may repeat pairs.
        sort_pairs (bool): Sort all discovered pairs before sampling.
        verbose (bool): Print discovered/sampled pairs.

    Returns:
        list[tuple[str, str]]: ``(category, item_id)`` pairs.
    """
    root_dir = Path(root_dir)
    if obj_names is None:
        obj_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    else:
        obj_dirs = [root_dir / name for name in obj_names if (root_dir / name).is_dir()]

    pair_list = []
    for obj_dir in obj_dirs:
        obj_name = obj_dir.name
        for item_dir in obj_dir.iterdir():
            if item_dir.is_dir():
                pair_list.append((obj_name, item_dir.name))

    if sort_pairs:
        pair_list = sorted(pair_list)

    if n_sample is not None:
        assert len(pair_list) > 0, "[sample_obj_pairs] No matched pairs found."
        if replace:
            assert n_sample >= 0, "[sample_obj_pairs] n_sample should be non-negative."
        else:
            assert len(pair_list) >= n_sample, "[sample_obj_pairs] Not enough pairs."

        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(pair_list), size=n_sample, replace=replace)
        pair_list = [pair_list[idx] for idx in idxs]

    if verbose:
        print("[sample_obj_pairs]")
        print(" root_dir:[%s]" % str(root_dir))
        print(
            " n_total:[%d] n_sampled:[%d] seed:[%s]" %
            (len(pair_list), len(pair_list), seed)
        )
        for i, pair in enumerate(pair_list):
            print(" [%02d/%02d] [%s]" % (i, len(pair_list), str(pair)))

    return pair_list
