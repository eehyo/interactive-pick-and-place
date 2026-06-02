"""
Notebook-facing APIs for the interactive pick-and-place demos.

This module contains the small stateful classes used by the notebooks:

- ``FFWSG2IKSolverAPI`` wraps the low-level MuJoCo site-position IK solver and
  exposes a simple target-based interface.
- ``PickAndPlaceObservationAPI`` collects RGB-D images, point clouds, robot
  state, object state, and optional segmentation into one observation
  dictionary.
- ``InteractivePickAndPlaceLoopController`` owns the pick-and-place
  mode/phase state machine and returns simulation step commands to the
  notebook loop.

The classes deliberately do not hide the entire experiment inside one large
``run`` method. The notebooks still own the outer loop, rendering, and user
interface, which keeps the control flow visible for debugging and teaching.

Student implementation guide:
    Search for the marker ``FILL IN`` to find the project blanks.

    Required blanks in this file:
        1. ``FFWSG2IKSolverAPI.DEFAULT_SOLVER_CONFIG``:
           Tune the IK solver configuration and study how it affects
           convergence, smoothness, and stability. Baseline values are provided
           so the code remains runnable.
        2. ``InteractivePickAndPlaceLoopController.update_pose_phase``:
           Convert the current phase target into IK targets, solve IK, choose
           the gripper command, and decide when to advance to the next phase.
        3. ``InteractivePickAndPlaceLoopController.make_execute_step_info``:
           Pack the controller's upper-body and gripper controls into the
           dictionary consumed by the notebook's ``env.step(...)`` call.

    The lower-level ``FFWSG2IKSolverAPI`` is intentionally provided as a
    scaffold. In this project, the main focus is how phase targets become
    robot controls, not how to rewire the IK solver internals.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from package.kinematics.transforms import (  # type: ignore
    D2R,
    np_uv,
    pr2t,
    rpy2r,
    t2p,
    view_in_local,
)
from package.mujoco_sim.mujoco_ik import SitePositionIKSolverRevPriBase  # type: ignore
from package.mujoco_sim.mujoco_utils import merge_mjcfs  # type: ignore
from package.mujoco_sim.parser import MuJoCoParser  # type: ignore
from package.utility.utils import group_by_prefix, sample_xyzs  # type: ignore

from .pick_and_place_utils import (  # type: ignore
    attach_body_to_body,
    build_point_cloud_observation,
    capture_rgbd_observation,
    collect_body_state,
    collect_joint_state,
    get_grasp_offsets,
    get_gripper_ctrl,
    make_owlv2_sam2_segmenter,
    make_pick_place_phase_targets,
    run_prompt_to_segmentation,
    select_arm_by_position,
)


class FFWSG2IKSolverAPI(object):
    """
    Notebook-friendly site-position IK API for the FFW-SG2 robot.

    This class wraps ``SitePositionIKSolverRevPriBase`` and keeps the notebook
    side focused on a simple sequence: configure the IK environment, add one or
    more 3D site targets, solve IK, then send the returned qpos to MuJoCo.

    The API is intentionally explicit. In the common path, the caller computes
    all offset targets outside the class and passes them through ``p_trgt``.
    When useful, ``compute_target_position`` can also build a target from
    ``p_base + R @ offset`` using either an explicit rotation matrix or RPY
    angles.

    Typical use in the notebooks:
        1. Create or set the IK environment.
        2. Add one or more named site targets with ``set_ik_target``.
        3. Call ``solve_ik``.
        4. Use the returned qpos as actuator controls in ``env.step``.

    Example:
        ik_api = FFWSG2IKSolverAPI(env=env, use_arm="r")
        ik_api.set_ik_target("tcp", type="tcp", p_trgt=p_tcp)
        ik_api.set_ik_target("tcp_top", type="top", p_trgt=p_tcp_top)
        q_ctrl, ik_info = ik_api.solve_ik(env=env)
        env.step(ctrl=q_ctrl, joint_names=ik_api.joints_use, nstep=30)

    Attributes:
        env: External MuJoCo simulation environment used by ``solve_ik``.
        ik_env: Internal MuJoCo parser used by the IK solver.
        use_arm (str): Arm key, usually "r" or "l".
        joints_use (list[str] | None): Revolute/prismatic joints used by IK.
        ik_targets (list[dict]): Buffered site-position IK targets.
        solver_kwargs (dict): Persistent kwargs for solver construction.
        solve_kwargs (dict): Persistent kwargs passed to ``solve_ik``.
        last_info (dict | None): Last IK solver info dictionary.
        last_qpos_used_best (np.ndarray | None): Last best IK solution.
    """

    # ------------------------------------------------------------------
    # FILL IN / EXPERIMENT: Tune the IK solver configuration.
    #
    # Baseline values are provided so the notebook can run after the other
    # blanks are implemented. For the project, try changing these values and
    # observe how the robot behavior changes.
    #
    # Things to think about:
    #   - Does increasing ``max_ik_tick`` improve convergence or only slow the
    #     loop down?
    #   - Do larger step sizes make motion faster, unstable, or jittery?
    #   - How does ``k_null`` affect posture preference and smoothness?
    #   - What happens near joint limits?
    #
    # Suggested experiment:
    #   Compare at least two configurations and report their effect on IK
    #   error, motion smoothness, and pick-and-place success.
    # ------------------------------------------------------------------
    DEFAULT_SOLVER_CONFIG = dict(
        max_ik_tick      = 10,       # Number of IK update iterations per solve.
        ik_stepsize_rev  = 10 * D2R, # Revolute-joint update scale.
        ik_stepsize_pri  = 0.10,     # Prismatic-joint update scale.
        ik_update_th_rev = 10 * D2R, # Revolute update clipping threshold.
        ik_update_th_pri = 0.10,     # Prismatic update clipping threshold.
        dls_damping      = 1e-6,     # DLS damping for most IK joints.
        dls_damping_lift = 1e-5,     # DLS damping override for the lift joint.
        max_probe_rev    = 3 * D2R,  # Joint-limit probing range for revolute joints.
        max_probe_pri    = 0.01,     # Joint-limit probing range for prismatic joints.
        k_null           = 0.1,      # Nullspace posture-control gain.
    )

    DEFAULT_SOLVE_CONFIG = dict(
        joint_limit_handle_flag = True,
        nullspace_control_flag  = True,
        base_control_flag       = False,
    )

    SITE_TYPE_TO_SUFFIX = {
        "site": None,
        "tcp": "",
        "top": "_top",
        "side": "_side",
        "front": "_front",
    }

    def __init__(
            self,
            env=None,
            ik_env=None,
            rel_xml_path=None,
            use_arm="r",
            joints_use=None,
            include_other_arm=True,
            include_lift=True,
            solver_kwargs=None,
            solve_kwargs=None,
            name="FFW-SG2 IK Solver API",
        ):
        """
        Initialize the IK API.

        Parameters:
            env: Optional live MuJoCo environment. If provided, its
                ``rel_xml_path`` is used to create ``ik_env`` when ``ik_env`` and
                ``rel_xml_path`` are not provided.
            ik_env: Optional pre-built IK MuJoCo parser.
            rel_xml_path (str | None): XML path used to build a new IK parser.
            use_arm (str): Arm key used for gripper site names, usually "r" or
                "l".
            joints_use (list[str] | None): Joint names to use for IK. If None,
                joints are inferred from ``ik_env`` when available.
            include_other_arm (bool): Whether inferred joints include the arm
                opposite to ``use_arm``.
            include_lift (bool): Whether inferred joints include lift joints.
            solver_kwargs (dict | None): Persistent solver construction kwargs.
            solve_kwargs (dict | None): Persistent kwargs for solver execution.
            name (str): Human-readable API name.
        """
        self.name = str(name)
        self.env = env
        self.ik_env = None
        self.use_arm = str(use_arm)
        self.include_other_arm = bool(include_other_arm)
        self.include_lift = bool(include_lift)
        self.joints_use = None
        self.ik_site_solver = None
        self.ik_targets = []
        self.solver_kwargs = {}
        self.solve_kwargs = dict(self.DEFAULT_SOLVE_CONFIG)
        self.last_info = None
        self.last_qpos_used_best = None

        if env is not None or ik_env is not None or rel_xml_path is not None:
            self.set_ik_env(env=env, ik_env=ik_env, rel_xml_path=rel_xml_path)

        if joints_use is None and self.ik_env is not None:
            joints_use = self.infer_joints_use(
                env=self.ik_env,
                use_arm=self.use_arm,
                include_other_arm=self.include_other_arm,
                include_lift=self.include_lift,
            )
        if joints_use is not None:
            self.joints_use = list(joints_use)

        if solve_kwargs is not None:
            self.set_solve_config(**dict(solve_kwargs))
        if solver_kwargs is not None:
            self.solver_kwargs.update(dict(solver_kwargs))
        if self.ik_env is not None and self.joints_use is not None:
            self.set_ik_config(remake_solver=True)

    def infer_joints_use(self, env=None, use_arm=None, include_other_arm=None, include_lift=None):
        """
        Infer the revolute/prismatic joints used by the SG2 IK problem.

        Parameters:
            env: MuJoCo parser/environment with ``rev_pri_joint_names``.
            use_arm (str | None): Main arm key. Defaults to ``self.use_arm``.
            include_other_arm (bool | None): Include the opposite arm joints.
            include_lift (bool | None): Include lift joints.

        Returns:
            list[str]: Joint names in the order they appear in
                ``env.rev_pri_joint_names``.
        """
        env = self.ik_env if env is None else env
        if env is None or not hasattr(env, "rev_pri_joint_names"):
            raise ValueError("env with rev_pri_joint_names is required.")

        use_arm = self.use_arm if use_arm is None else str(use_arm)
        include_other_arm = self.include_other_arm if include_other_arm is None else bool(include_other_arm)
        include_lift = self.include_lift if include_lift is None else bool(include_lift)
        joint_groups = group_by_prefix(
            items       = env.rev_pri_joint_names,
            prefix_dict = {
                "arm_l_joint_names": ["arm_l_joint"],
                "arm_r_joint_names": ["arm_r_joint"],
                "lift_joint_name": ["lift_joint"],
            },
        )
        arm_key = "arm_%s_joint_names" % use_arm
        other_key = "arm_l_joint_names" if use_arm == "r" else "arm_r_joint_names"

        joints = list(joint_groups.get(arm_key, []))
        if include_other_arm:
            joints = joints + list(joint_groups.get(other_key, []))
        if include_lift:
            joints = joints + list(joint_groups.get("lift_joint_name", []))
        return joints

    def set_ik_env(self, env=None, ik_env=None, rel_xml_path=None):
        """
        Set or replace the IK environment.

        Parameters:
            env: Optional live MuJoCo environment. Stored as ``self.env`` and
                used for solving and home posture extraction.
            ik_env: Optional pre-built IK MuJoCo parser.
            rel_xml_path (str | None): XML path used to build a new
                ``MuJoCoParser`` if ``ik_env`` is not provided.

        Returns:
            FFWSG2IKSolverAPI: ``self`` for chained calls.
        """
        if env is not None:
            self.env = env
        if ik_env is not None:
            self.ik_env = ik_env
        else:
            if rel_xml_path is None:
                env_for_path = self.env if env is None else env
                if env_for_path is None or not hasattr(env_for_path, "rel_xml_path"):
                    raise ValueError("One of ik_env, rel_xml_path, or env with rel_xml_path is required.")
                rel_xml_path = env_for_path.rel_xml_path
            self.ik_env = MuJoCoParser(rel_xml_path=rel_xml_path, verbose=False)

        if self.joints_use is None:
            self.joints_use = self.infer_joints_use(env=self.ik_env)
        if self.joints_use is not None:
            self.make_solver()
        return self

    def set_ik_config(self, joints_use=None, remake_solver=False, **solver_kwargs):
        """
        Update IK solver construction/configuration values.

        Parameters:
            joints_use (list[str] | None): New active joint list. Passing this
                remakes the solver.
            remake_solver (bool): Force rebuilding ``SitePositionIKSolverRevPriBase``.
            **solver_kwargs: Solver configuration kwargs such as
                ``max_ik_tick``, ``ik_stepsize_rev``, ``ik_stepsize_pri``,
                ``dls_damping``, or ``k_null``.

        Returns:
            FFWSG2IKSolverAPI: ``self`` for chained calls.
        """
        if joints_use is not None:
            self.joints_use = list(joints_use)
            remake_solver = True
        if len(solver_kwargs) > 0:
            self.solver_kwargs.update(dict(solver_kwargs))

        if self.ik_env is None:
            return self
        if self.joints_use is None:
            self.joints_use = self.infer_joints_use(env=self.ik_env)
        if remake_solver or self.ik_site_solver is None:
            self.make_solver()
        elif len(solver_kwargs) > 0:
            self.ik_site_solver.set_ik_config(**solver_kwargs)
        return self

    def set_solve_config(self, **solve_kwargs):
        """
        Update persistent solve-time kwargs.

        Parameters:
            **solve_kwargs: Kwargs passed to ``SitePositionIKSolverRevPriBase.solve_ik``,
                such as ``joint_limit_handle_flag``, ``nullspace_control_flag``,
                ``base_control_flag``, or base-motion thresholds.

        Returns:
            FFWSG2IKSolverAPI: ``self`` for chained calls.
        """
        self.solve_kwargs.update(dict(solve_kwargs))
        return self

    def make_solver(self):
        """
        Build a fresh low-level site-position IK solver.

        The solver uses ``self.ik_env``, ``self.joints_use``, default solver
        config, and any values stored in ``self.solver_kwargs``. The home
        posture is read from ``self.env`` when available, otherwise from
        ``self.ik_env``.

        Returns:
            SitePositionIKSolverRevPriBase: Newly constructed low-level solver.
        """
        if self.ik_env is None:
            raise ValueError("ik_env is required.")
        if self.joints_use is None:
            self.joints_use = self.infer_joints_use(env=self.ik_env)

        cfg = dict(self.DEFAULT_SOLVER_CONFIG)
        dls_damping_lift = cfg.pop("dls_damping_lift", None)
        if dls_damping_lift is not None and np.isscalar(cfg["dls_damping"]):
            cfg["dls_damping"] = float(cfg["dls_damping"]) * np.ones(len(self.joints_use))
            cfg["dls_damping"][-1] = float(dls_damping_lift) # lift
        if self.env is not None and hasattr(self.env, "rev_pri_joint_names"):
            cfg["q_home_rev_pri"] = self.env.get_qpos(joint_names=self.env.rev_pri_joint_names)
        elif hasattr(self.ik_env, "rev_pri_joint_names"):
            cfg["q_home_rev_pri"] = self.ik_env.get_qpos(joint_names=self.ik_env.rev_pri_joint_names)
        cfg.update(self.solver_kwargs)

        self.ik_site_solver = SitePositionIKSolverRevPriBase(
            ik_env     = self.ik_env,
            joints_use = self.joints_use,
            **cfg,
        )
        return self.ik_site_solver

    def reset(self, clear_targets=True, env=None):
        """
        Reset cached IK results and optionally clear buffered targets.

        Parameters:
            clear_targets (bool): If True, remove all buffered IK targets.
            env: Optional live environment to store as ``self.env``.

        Returns:
            FFWSG2IKSolverAPI: ``self`` for chained calls.
        """
        if env is not None:
            self.env = env
        if clear_targets:
            self.clear_targets()
        elif self.ik_site_solver is not None:
            self.ik_site_solver.reset_buffers(env=self.env)
        self.last_info = None
        self.last_qpos_used_best = None
        return self

    def clear_targets(self):
        """
        Remove all buffered IK targets and clear low-level solver buffers.

        Returns:
            FFWSG2IKSolverAPI: ``self`` for chained calls.
        """
        self.ik_targets = []
        if self.ik_site_solver is not None:
            self.ik_site_solver.reset_buffers()
        return self

    def get_site_name(self, name=None, type="site", use_arm=None, site_name=None):
        """
        Resolve a user target name/type into a MuJoCo site name.

        Parameters:
            name (str | None): User-facing target name. For ``type="site"``,
                this is treated as the actual site name.
            type (str): Target type. Supported values are "site", "tcp",
                "top", "side", and "front".
            use_arm (str | None): Arm key for gripper site aliases.
            site_name (str | None): Explicit MuJoCo site name. If provided, it
                takes precedence over ``name`` and ``type``.

        Returns:
            str: MuJoCo site name.
        """
        if site_name is not None:
            return str(site_name)
        if type == "site":
            if name is None:
                raise ValueError("name or site_name is required for type='site'.")
            return str(name)
        if type not in self.SITE_TYPE_TO_SUFFIX:
            raise ValueError("Unknown target type: %s" % str(type))
        use_arm = self.use_arm if use_arm is None else str(use_arm)
        return "gripper_%s_tcp%s" % (use_arm, self.SITE_TYPE_TO_SUFFIX[type])

    def compute_target_position(self, p_trgt=None, p_base=None, offset=None, R=None, rpy=None):
        """
        Compute a 3D target position.

        Parameters:
            p_trgt (array-like | None): Final target position. If provided,
                it is returned directly after shape conversion.
            p_base (array-like | None): Base position used when ``p_trgt`` is
                not provided.
            offset (array-like | None): Optional local/world offset added to
                ``p_base``.
            R (array-like | None): Optional 3x3 rotation applied to ``offset``.
            rpy (array-like | None): Optional roll-pitch-yaw rotation used when
                ``R`` is not provided.

        Returns:
            np.ndarray: Shape ``(3,)`` target position.
        """
        if p_trgt is not None:
            return np.asarray(p_trgt, dtype=np.float64).reshape(3)
        if p_base is None:
            raise ValueError("Either p_trgt or p_base must be provided.")

        p = np.asarray(p_base, dtype=np.float64).reshape(3)
        if offset is None:
            return p

        offset = np.asarray(offset, dtype=np.float64).reshape(3)
        if R is None and rpy is not None:
            R = rpy2r(rpy)
        if R is not None:
            offset = np.asarray(R, dtype=np.float64).reshape(3, 3) @ offset
        return p + offset

    def set_ik_target(
            self,
            name=None,
            type="site",
            site_name=None,
            p_trgt=None,
            p_base=None,
            offset=None,
            R=None,
            rpy=None,
            weight=1.0,
            clear=False,
            use_arm=None,
            **kwargs,
        ):
        """
        Add one site-position IK target.

        Parameters:
            name (str | None): User-facing target name. For ``type="site"``,
                this is also the MuJoCo site name unless ``site_name`` is given.
            type (str): Target type. Use "site" for an explicit site, or "tcp",
                "top", "side", "front" for gripper aliases.
            site_name (str | None): Explicit MuJoCo site name.
            p_trgt (array-like | None): Final 3D target position.
            p_base (array-like | None): Base position used to compute target
                when ``p_trgt`` is not provided.
            offset (array-like | None): Offset added to ``p_base``. If ``R`` or
                ``rpy`` is provided, the offset is interpreted as local.
            R (array-like | None): 3x3 rotation for ``offset``.
            rpy (array-like | None): Roll-pitch-yaw rotation for ``offset``.
            weight (float): IK task weight for this site target.
            clear (bool): If True, clear existing targets before adding this one.
            use_arm (str | None): Arm key for gripper site aliases.
            **kwargs: Convenience aliases. ``p``, ``pos``, ``position``, and
                ``target`` map to ``p_trgt``. ``site`` maps to ``site_name``.
                ``w`` maps to ``weight``.

        Returns:
            dict: Buffered target spec with ``name``, ``type``, ``site_name``,
                ``p_trgt``, and ``weight``.

        Example:
            ik_api.set_ik_target("tcp", type="tcp", p_trgt=p_tcp)
            ik_api.set_ik_target("gripper_r_tcp_top", type="site", p_trgt=p_top)
        """
        if "p" in kwargs and p_trgt is None:
            p_trgt = kwargs.pop("p")
        if "pos" in kwargs and p_trgt is None:
            p_trgt = kwargs.pop("pos")
        if "position" in kwargs and p_trgt is None:
            p_trgt = kwargs.pop("position")
        if "target" in kwargs and p_trgt is None:
            p_trgt = kwargs.pop("target")
        if "site" in kwargs and site_name is None:
            site_name = kwargs.pop("site")
        if "w" in kwargs:
            weight = kwargs.pop("w")
        if len(kwargs) > 0:
            raise TypeError("Unexpected IK target keys: %s" % sorted(kwargs.keys()))

        if clear:
            self.clear_targets()

        site_name = self.get_site_name(name=name, type=type, use_arm=use_arm, site_name=site_name)
        p_trgt = self.compute_target_position(
            p_trgt = p_trgt,
            p_base = p_base,
            offset = offset,
            R      = R,
            rpy    = rpy,
        )
        target = dict(
            name      = str(name) if name is not None else str(site_name),
            type      = str(type),
            site_name = str(site_name),
            p_trgt    = p_trgt,
            weight    = float(weight),
        )
        self.ik_targets.append(target)
        return target

    def set_ik_targets(self, targets, clear=True):
        """
        Add multiple IK targets.

        Parameters:
            targets: Target collection. Supported forms include:
                ``{"site_name": p_trgt}``, ``{"name": {"p_trgt": p}}``,
                ``[{"name": ..., "type": ..., "p_trgt": ...}]``, and
                ``[(name, p_trgt), (name, p_trgt, weight)]``.
            clear (bool): If True, clear existing targets before adding the
                collection.

        Returns:
            list[dict]: Buffered target specs returned by ``set_ik_target``.
        """
        if clear:
            self.clear_targets()
        if targets is None:
            return []

        if isinstance(targets, dict):
            targets_iter = []
            for name, value in targets.items():
                if isinstance(value, dict):
                    target = dict(value)
                    target.setdefault("name", name)
                else:
                    target = dict(name=name, p_trgt=value)
                targets_iter.append(target)
        else:
            targets_iter = list(targets)

        added = []
        for target in targets_iter:
            if isinstance(target, tuple):
                if len(target) == 2:
                    name, p_trgt = target
                    target = dict(name=name, p_trgt=p_trgt)
                elif len(target) == 3:
                    name, p_trgt, weight = target
                    target = dict(name=name, p_trgt=p_trgt, weight=weight)
                else:
                    raise ValueError("Tuple target must be (name, p_trgt[, weight]).")
            added.append(self.set_ik_target(**dict(target)))
        return added

    def add_targets_to_solver(self):
        """
        Copy buffered API targets into the low-level IK solver.

        Returns:
            FFWSG2IKSolverAPI: ``self`` for chained calls.
        """
        if self.ik_site_solver is None:
            self.make_solver()
        self.ik_site_solver.reset_buffers()
        for target in self.ik_targets:
            self.ik_site_solver.add_ik_target(
                site_name = target["site_name"],
                p_trgt    = target["p_trgt"],
                weight    = target.get("weight", 1.0),
            )
        return self

    def solve_ik(self, env=None, targets=None, clear_targets=True, joints_use=None, solve_kwargs=None):
        """
        Solve IK for the currently buffered targets.

        Parameters:
            env: Optional live MuJoCo environment. If provided, it replaces
                ``self.env`` before solving.
            targets: Optional target collection passed to ``set_ik_targets``
                before solving.
            clear_targets (bool): Used only when ``targets`` is provided.
            joints_use (list[str] | None): Optional one-call active joint list.
            solve_kwargs (dict | None): One-call overrides for persistent
                ``self.solve_kwargs``.

        Returns:
            tuple[np.ndarray, dict]: ``(qpos_used_best, ik_info)`` from the
                low-level solver.
        """
        if env is not None:
            self.env = env
        if self.env is None:
            raise ValueError("env is required for solve_ik.")

        if targets is not None:
            self.set_ik_targets(targets, clear=clear_targets)
        if len(self.ik_targets) == 0:
            raise ValueError("At least one IK target is required.")

        self.add_targets_to_solver()
        kwargs = dict(self.solve_kwargs)
        if solve_kwargs is not None:
            kwargs.update(dict(solve_kwargs))

        _, ik_info = self.ik_site_solver.solve_ik(
            env        = self.env,
            joints_use = self.joints_use if joints_use is None else list(joints_use),
            **kwargs,
        )
        self.last_info = ik_info
        self.last_qpos_used_best = ik_info["qpos_used_best"]
        return self.last_qpos_used_best, ik_info

    def get_ctrl(self, gripper_ctrl=None, qpos_used=None):
        """
        Build a control vector from an IK solution.

        Parameters:
            gripper_ctrl (array-like | None): Optional extra gripper control
                values appended after the IK qpos.
            qpos_used (array-like | None): Optional IK qpos. If None, uses the
                last solution from ``solve_ik``.

        Returns:
            np.ndarray: IK qpos, optionally concatenated with gripper controls.
        """
        qpos_used = self.last_qpos_used_best if qpos_used is None else qpos_used
        if qpos_used is None:
            raise ValueError("No IK solution is available. Call solve_ik first or pass qpos_used.")
        qpos_used = np.asarray(qpos_used, dtype=np.float64).reshape(-1)
        if gripper_ctrl is None:
            return qpos_used
        return np.concatenate([qpos_used, np.asarray(gripper_ctrl, dtype=np.float64).reshape(-1)])

    def get_step_joint_names(self, extra_joint_names=None):
        """
        Return joint names aligned with ``get_ctrl`` output.

        Parameters:
            extra_joint_names (list[str] | None): Optional names appended after
                ``self.joints_use``, usually gripper joints/actuators.

        Returns:
            list[str]: Joint names for ``env.step``.
        """
        if extra_joint_names is None:
            return list(self.joints_use)
        return list(self.joints_use) + list(extra_joint_names)

    def render(self, env=None, **kwargs):
        """
        Render low-level IK targets on the given environment.

        Parameters:
            env: Optional MuJoCo environment. Defaults to ``self.env``.
            **kwargs: Additional kwargs forwarded to the low-level solver
                render method.

        Returns:
            Any: Return value of ``self.ik_site_solver.render``.
        """
        env = self.env if env is None else env
        if env is None:
            raise ValueError("env is required for render.")
        return self.ik_site_solver.render(env, **kwargs)


class PickAndPlaceObservationAPI(object):
    """
    Observation collector for interactive pick-and-place notebooks.

    The class gathers the pieces that a notebook usually needs at each loop
    iteration: RGB-D images, point clouds, robot joint state, object body
    poses, selected-object metadata, and optional OWLv2+SAM2 segmentation.

    Perception is opt-in. Calling ``get_observation(segmentation=False)`` only
    captures state and images. Calling it with ``segmentation=True`` runs the
    configured text-prompt segmenter for ``target_object["text_query"]`` and
    can use the resulting mask for point-cloud extraction.

    Heavy perception backends are created lazily. If segmentation is requested
    and no segmenter exists yet, ``set_segmenter`` builds one from the local
    HuggingFace model directories in ``DEFAULT_SEGMENTER_KWARGS``.

    Example:
        obs_api = PickAndPlaceObservationAPI(
            env         = env,
            head_lookat = np.array([0.8, 0.0, 0.7]),
            joint_names = joint_groups["use_upper_body_joint_names"],
        )
        obs_api.set_target_object("body_obj_can_17", text_query="soda can")
        obs = obs_api.get_observation(image=True, pcd=True, robot_state=True)
        rgb = obs["image"]["rgb"]
        pcd = obs["pcd"]["full"]

    Attributes:
        env: MuJoCo parser/environment used for observation capture.
        cam_name (str): Body name of the egocentric camera.
        head_lookat (np.ndarray | None): World-frame point the camera looks at.
        joint_names (list[str] | None): Joints queried for robot state.
        target_object (dict): Target object metadata used by segmentation and
            object-state collection.
        segmenter: Optional OWLv2+SAM2 backend.
        segmenter_kwargs (dict): Persistent kwargs used when lazily building
            the segmenter.
        segmentation_kwargs (dict): Persistent kwargs passed to the segmenter.
        last_observation (dict | None): Most recent observation dictionary.
    """

    DEFAULT_SEGMENTER_KWARGS = dict(
        owl_local_dir  = "../hf_models/owlv2",
        sam2_local_dir = "../hf_models/sam2",
        device         = None,
        verbose        = True,
    )

    DEFAULT_SEGMENTATION_KWARGS = dict(
        det_threshold          = 0.1,
        fallback_det_thresholds = (0.05, 0.02),
        topk_per_query         = 1,
        neg_margin             = 10,
        use_bbox_neg_points    = False,
        fill_holes             = True,
        keep_largest           = True,
        prefer_pos_mask        = False,
        clip_mask_to_box       = True,
        box_clip_margin        = 4,
        bbox_mask_fallback     = True,
        overlay_alpha          = 0.5,
        use_text_aliases       = True,
    )

    def __init__(
            self,
            env=None,
            cam_name="head_cam",
            head_lookat=None,
            joint_names=None,
            target_body_name=None,
            target_text_query=None,
            segmenter=None,
            segmenter_kwargs=None,
            segmentation_kwargs=None,
            name="Pick-and-Place Observation API",
        ):
        """
        Initialize the observation API.

        Parameters:
            env: Optional MuJoCo parser/environment.
            cam_name (str): Body name used as the egocentric camera position.
            head_lookat (array-like | None): World-frame camera look-at point.
            joint_names (list[str] | None): Joints returned under
                ``obs["robot_state"]["q"]``.
            target_body_name (str | None): Body name of the selected object.
            target_text_query (str | None): Text query used by OWLv2+SAM2.
            segmenter: Optional pre-built text-prompt segmenter.
            segmenter_kwargs (dict | None): Persistent kwargs for lazy
                segmenter construction.
            segmentation_kwargs (dict | None): Persistent kwargs for each
                segmentation call.
            name (str): Human-readable API name.
        """
        self.name = str(name)
        self.env = env
        self.cam_name = str(cam_name)
        self.head_lookat = None if head_lookat is None else np.asarray(head_lookat, dtype=np.float64).reshape(3)
        self.joint_names = None if joint_names is None else list(joint_names)
        self.target_object = dict(body_name=None, text_query=None, metadata={})
        self.segmenter = segmenter
        self.segmenter_kwargs = dict(self.DEFAULT_SEGMENTER_KWARGS)
        self.segmentation_kwargs = dict(self.DEFAULT_SEGMENTATION_KWARGS)
        self.last_observation = None

        if segmenter_kwargs is not None:
            self.segmenter_kwargs.update(dict(segmenter_kwargs))
        if segmentation_kwargs is not None:
            self.segmentation_kwargs.update(dict(segmentation_kwargs))
        if target_body_name is not None or target_text_query is not None:
            self.set_target_object(body_name=target_body_name, text_query=target_text_query)

    def set_env(self, env):
        """
        Set the MuJoCo environment used for observation capture.

        Parameters:
            env: MuJoCo parser/environment.

        Returns:
            PickAndPlaceObservationAPI: ``self`` for chained calls.
        """
        self.env = env
        return self

    def set_camera(self, cam_name=None, head_lookat=None):
        """
        Set camera capture parameters.

        Parameters:
            cam_name (str | None): Body name of the egocentric camera.
            head_lookat (array-like | None): World-frame look-at point.

        Returns:
            PickAndPlaceObservationAPI: ``self`` for chained calls.
        """
        if cam_name is not None:
            self.cam_name = str(cam_name)
        if head_lookat is not None:
            self.head_lookat = np.asarray(head_lookat, dtype=np.float64).reshape(3)
        return self

    def set_joint_names(self, joint_names):
        """
        Set joints queried by ``get_observation(robot_state=True)``.

        Parameters:
            joint_names (list[str] | None): Joint names to query. Passing None
                keeps only full qpos/qvel in the robot-state output.

        Returns:
            PickAndPlaceObservationAPI: ``self`` for chained calls.
        """
        self.joint_names = None if joint_names is None else list(joint_names)
        return self

    def set_target_object(self, body_name=None, text_query=None, **metadata):
        """
        Set the selected object metadata.

        Parameters:
            body_name (str | None): MuJoCo body name of the selected object.
            text_query (str | None): Text query for OWLv2+SAM2. If omitted,
                ``body_name`` is also used as a coarse fallback query.
            **metadata: Extra user metadata stored under ``target_object``.

        Returns:
            PickAndPlaceObservationAPI: ``self`` for chained calls.
        """
        if body_name is not None:
            self.target_object["body_name"] = str(body_name)
        if text_query is not None:
            self.target_object["text_query"] = str(text_query)
        elif self.target_object.get("text_query", None) is None and body_name is not None:
            self.target_object["text_query"] = str(body_name).replace("body_obj_", "").replace("_", " ")
        if len(metadata) > 0:
            self.target_object.setdefault("metadata", {}).update(dict(metadata))
        return self

    def set_segmenter(self, segmenter=None, **segmenter_kwargs):
        """
        Set or lazily construct the OWLv2+SAM2 segmenter.

        Parameters:
            segmenter: Optional pre-built segmenter. If None, a segmenter is
                constructed with ``make_owlv2_sam2_segmenter``.
            **segmenter_kwargs: Persistent kwargs for construction.

        Returns:
            PickAndPlaceObservationAPI: ``self`` for chained calls.
        """
        if len(segmenter_kwargs) > 0:
            self.segmenter_kwargs.update(dict(segmenter_kwargs))
        self.segmenter = segmenter
        if self.segmenter is None:
            self.segmenter = make_owlv2_sam2_segmenter(**self.segmenter_kwargs)
        return self

    def set_segmentation_config(self, **segmentation_kwargs):
        """
        Update persistent kwargs passed to the segmentation backend.

        Parameters:
            **segmentation_kwargs: Backend-specific kwargs such as
                ``det_threshold``, ``topk_per_query``, ``neg_margin``,
                ``fill_holes``, and ``keep_largest``.

        Returns:
            PickAndPlaceObservationAPI: ``self`` for chained calls.
        """
        self.segmentation_kwargs.update(dict(segmentation_kwargs))
        return self

    def get_observation(
            self,
            env=None,
            image=True,
            pcd=True,
            robot_state=True,
            object_state=True,
            segmentation=False,
            target_object=True,
            extra_body_names=None,
            joint_names=None,
            cam_name=None,
            head_lookat=None,
            rsz_rate_for_img=1 / 4,
            rsz_rate_for_pcd=1 / 2,
            fovy=45,
            pcd_mask=None,
            to_world=True,
            restore_view=True,
            segmentation_kwargs=None,
        ):
        """
        Capture an observation dictionary.

        Parameters:
            env: Optional MuJoCo environment. If provided, it replaces
                ``self.env`` for this and future calls.
            image (bool): Include RGB-D image output.
            pcd (bool): Include point-cloud output. Requires depth, so image is
                captured internally even when ``image=False``.
            robot_state (bool): Include qpos/qvel and selected joint q.
            object_state (bool): Include target and extra body poses.
            segmentation (bool): Run OWLv2+SAM2 for ``target_object``.
            target_object (bool): Include selected-object metadata.
            extra_body_names (list[str] | None): Additional body poses.
            joint_names (list[str] | None): One-call joint-state override.
            cam_name (str | None): One-call camera-name override.
            head_lookat (array-like | None): One-call look-at override.
            rsz_rate_for_img (float | None): RGB-D resize factor.
            rsz_rate_for_pcd (float | None): Point-cloud downsampling factor.
            fovy (float | None): Camera field of view in degrees. The default
                45 matches the older notebook/10 point-cloud path.
            pcd_mask (np.ndarray | None): Optional mask for pcd extraction.
            to_world (bool): Return pcd points in world frame when possible.
            restore_view (bool): Restore viewer camera after RGB-D capture.
            segmentation_kwargs (dict | None): One-call segmentation overrides.

        Returns:
            dict: Observation dictionary. Common keys are ``image``, ``pcd``,
                ``robot_state``, ``object_state``, ``target_object``, and
                ``segmentation``.
        """
        if env is not None:
            self.env = env
        if self.env is None:
            raise ValueError("env is required for get_observation.")

        cam_name = self.cam_name if cam_name is None else str(cam_name)
        p_trgt = self.head_lookat if head_lookat is None else np.asarray(head_lookat, dtype=np.float64).reshape(3)
        obs = dict()

        need_image = bool(image) or bool(pcd) or bool(segmentation)
        image_obs = None
        if need_image:
            image_obs = capture_rgbd_observation(
                env              = self.env,
                cam_name         = cam_name,
                p_trgt           = p_trgt,
                rsz_rate_for_img = rsz_rate_for_img,
                restore_view     = restore_view,
            )
            if image:
                obs["image"] = image_obs
                obs["robot0_robotview"] = dict(
                    images     = dict(rgb=image_obs["rgb"], depth=image_obs["depth"]),
                    intrinsics = None,
                    pose_mat   = image_obs["T_cam"],
                )

        seg_results = None
        seg_result = None
        if segmentation:
            if self.segmenter is None:
                self.set_segmenter()
            text_query = self.target_object.get("text_query", None)
            if text_query is None:
                raise ValueError("target_object text_query is required for segmentation.")
            seg_kwargs = dict(self.segmentation_kwargs)
            if segmentation_kwargs is not None:
                seg_kwargs.update(dict(segmentation_kwargs))
            seg_result = run_prompt_to_segmentation(
                segmenter    = self.segmenter,
                rgb          = image_obs["rgb"],
                prompt       = text_query,
                **seg_kwargs,
            )
            seg_results = [] if seg_result is None else list(seg_result.get("candidates", [seg_result]))
            obs["segmentation"] = dict(
                text_query = text_query,
                results    = seg_results,
                best       = seg_result,
            )

        if pcd:
            mask = pcd_mask
            if mask is None and seg_result is not None:
                mask = seg_result.get("mask", None)
            obs["pcd"] = build_point_cloud_observation(
                env              = self.env,
                depth            = image_obs["depth"],
                mask             = mask,
                p_ego            = image_obs["p_ego"],
                p_trgt           = image_obs["p_trgt"],
                fovy             = fovy,
                rsz_rate_for_pcd = rsz_rate_for_pcd,
                to_world         = to_world,
            )

        if robot_state:
            state_joint_names = self.joint_names if joint_names is None else list(joint_names)
            obs["robot_state"] = collect_joint_state(
                env          = self.env,
                joint_names  = state_joint_names,
                include_full = True,
            )
            obs["robot_state"]["sim_time"] = self.env.get_sim_time() if hasattr(self.env, "get_sim_time") else self.env.data.time

        if object_state:
            body_names = []
            if self.target_object.get("body_name", None) is not None:
                body_names.append(self.target_object["body_name"])
            if extra_body_names is not None:
                body_names += list(extra_body_names)
            obs["object_state"] = collect_body_state(env=self.env, body_names=body_names)

        if target_object:
            obs["target_object"] = dict(self.target_object)

        self.last_observation = obs
        return obs


class InteractivePickAndPlaceLoopController(object):
    """
    Readable state-machine controller for interactive pick-and-place.

    The controller owns only the mode/phase logic and the command generation.
    It does not own the viewer loop, the Qt widgets, perception calls, or
    rendering. This separation lets the notebook keep the high-level rhythm
    explicit:

        while env.is_viewer_alive():
            obs = obs_api.get_observation(...)
            controller.set_target_object(...)
            controller.set_pick_place_targets(...)
            rollout = controller.update(pick_requested=button_is_on)
            if rollout["step"]["do_step"]:
                env.step(...)
                controller.after_step()
            render(...)

    Modes:
        ``idle``:
            Initial camera/physics settling period. The controller returns a
            small hold command so the robot remains near its current posture.
        ``choose_to_pick``:
            Waits until the outside notebook provides a selected object and a
            pick request, usually from radio buttons.
        ``execute``:
            Runs the pick-and-place phase machine. Pose phases solve IK; grasp
            and release phases command the gripper.
        ``done``:
            Rollout finished.

    Execution phases:
        ``approach`` -> ``pregrasp`` -> ``grasp`` -> ``lift`` ->
        ``place_approach`` -> ``place`` -> ``release`` -> ``retreat`` ->
        ``done``.

    Phase meanings:
        ``approach``:
            Move near the object before going to the final grasp pose.
        ``pregrasp``:
            Move to the grasp pose while keeping the gripper open.
        ``grasp``:
            Close the active gripper and record the object-to-TCP transform.
        ``lift``:
            Lift the grasped object away from the table.
        ``place_approach``:
            Move above the place target before descending.
        ``place``:
            Move to the release pose near the bowl/place region.
        ``release``:
            Open the gripper and detach the object.
        ``retreat``:
            Move the gripper away from the released object.
        ``done``:
            Finish the rollout and stop sending execution commands.

    Attributes:
        env: MuJoCo parser/environment.
        ik_api (FFWSG2IKSolverAPI): Site-position IK API used for phase targets.
        mode (str): Current outer mode.
        phase (str): Current execution phase.
        target_object (dict): Selected object metadata.
        targets (dict | None): Phase targets generated from pick/place points.
    """

    # Full high-level phase order for the pick-and-place rollout.
    PHASES = ["approach", "pregrasp", "grasp", "lift", "place_approach", "place", "release", "retreat", "done"]

    # Phases that are solved as IK pose-tracking problems. The other execution
    # phases are gripper-specific phases handled by ``update_grasp_phase`` and
    # ``update_release_phase``.
    POSE_PHASES = ["approach", "pregrasp", "lift", "place_approach", "place", "retreat"]

    def __init__(
            self,
            env=None,
            ik_api=None,
            use_arm=None,
            upper_body_joint_names=None,
            gripper_joint_names=None,
            idle_tick_goal=30,
            execute_nstep=10,
            idle_nstep=10,
            ik_err_thr=5e-1,
            hold_tick_goal=10,
            release_tick_goal=10,
            max_done_tick=30,
            gripper_dq_thr=1e-2,
            grasp_pitch_deg=0.0,
            grasp_axis_len=0.1,
            name="Interactive Pick-and-Place Loop Controller",
        ):
        """
        Initialize the loop controller.

        Parameters:
            env: Optional MuJoCo parser/environment.
            ik_api (FFWSG2IKSolverAPI | None): Optional IK API. If None, one is
                created when ``env`` is available.
            use_arm (str | None): Forced arm key. If None, the controller
                selects the closer arm when pick targets are set.
            upper_body_joint_names (list[str] | None): IK-controlled joints.
            gripper_joint_names (list[str] | None): Gripper joints appended to
                step commands.
            idle_tick_goal (int): Number of idle update ticks before entering
                ``choose_to_pick``.
            execute_nstep (int): MuJoCo substeps for execution commands.
            idle_nstep (int): MuJoCo substeps for idle settling commands.
            ik_err_thr (float): IK error threshold used for phase completion.
            hold_tick_goal (int): Stable ticks required to advance pose/grasp.
            release_tick_goal (int): Ticks to keep release command active.
            max_done_tick (int): Ticks before ``done`` stops returning steps.
            gripper_dq_thr (float): Gripper velocity threshold for stable grasp.
            grasp_pitch_deg (float): Pitch used for default guide offsets.
            grasp_axis_len (float): Guide-point distance from TCP.
            name (str): Human-readable controller name.
        """
        self.name = str(name)
        self.env = env
        self.use_arm = None if use_arm is None else str(use_arm)
        self.upper_body_joint_names = None if upper_body_joint_names is None else list(upper_body_joint_names)
        self.gripper_joint_names = ["gripper_l_joint1", "gripper_r_joint1"] if gripper_joint_names is None else list(gripper_joint_names)
        self.ik_api = ik_api
        if self.ik_api is None and env is not None:
            self.ik_api = FFWSG2IKSolverAPI(env=env, use_arm="r" if self.use_arm is None else self.use_arm, joints_use=self.upper_body_joint_names)

        self.idle_tick_goal = int(idle_tick_goal)
        self.execute_nstep = int(execute_nstep)
        self.idle_nstep = int(idle_nstep)
        self.ik_err_thr = float(ik_err_thr)
        self.hold_tick_goal = int(hold_tick_goal)
        self.release_tick_goal = int(release_tick_goal)
        self.max_done_tick = int(max_done_tick)
        self.gripper_dq_thr = float(gripper_dq_thr)
        self.grasp_offsets = get_grasp_offsets(pitch_deg=grasp_pitch_deg, axis_len=grasp_axis_len)

        self.mode = "idle"
        self.phase = "approach"
        self.idle_tick = 0
        self.phase_tick = 0
        self.done_tick = 0
        self.ik_err = np.inf
        self.target_object = dict(body_name=None, metadata={})
        self.pick_position = None
        self.place_position = None
        self.targets = None
        self.ctrl_upper_body = None
        self.ctrl_gripper = None
        self.use_gripper_prev_q = None
        self.attach_active = False
        self.T_body_obj = None
        self.pending_attach_after_step = False
        self.last_rollout = None

    def set_env(self, env):
        """
        Set the MuJoCo environment.

        Parameters:
            env: MuJoCo parser/environment.

        Returns:
            InteractivePickAndPlaceLoopController: ``self``.
        """
        self.env = env
        if self.ik_api is None:
            self.ik_api = FFWSG2IKSolverAPI(env=env, use_arm="r" if self.use_arm is None else self.use_arm, joints_use=self.upper_body_joint_names)
        else:
            self.ik_api.set_ik_env(env=env)
        return self

    def set_target_object(self, body_name=None, **metadata):
        """
        Set the object selected by the outside radio-button UI.

        Parameters:
            body_name (str | None): MuJoCo body name of the object to pick.
            **metadata: Extra metadata copied into rollout info.

        Returns:
            InteractivePickAndPlaceLoopController: ``self``.
        """
        if body_name is not None:
            self.target_object["body_name"] = str(body_name)
        if len(metadata) > 0:
            self.target_object.setdefault("metadata", {}).update(dict(metadata))
        return self

    def set_pick_place_targets(self, pick_position=None, place_position=None, targets=None, use_arm=None):
        """
        Set externally computed pick/place targets.

        Parameters:
            pick_position (array-like | None): Pick TCP target in world frame.
            place_position (array-like | None): Place region/object center.
            targets (dict | None): Optional full phase target dictionary. If
                omitted, targets are generated from pick/place positions.
            use_arm (str | None): Optional one-call arm override.

        Returns:
            InteractivePickAndPlaceLoopController: ``self``.
        """
        if pick_position is not None:
            self.pick_position = np.asarray(pick_position, dtype=np.float64).reshape(3)
        if place_position is not None:
            self.place_position = np.asarray(place_position, dtype=np.float64).reshape(3)
        if use_arm is not None:
            self.use_arm = str(use_arm)
        elif self.use_arm is None and self.env is not None and self.pick_position is not None:
            self.use_arm = select_arm_by_position(env=self.env, p_trgt=self.pick_position)

        if targets is not None:
            self.targets = targets
        elif self.pick_position is not None and self.place_position is not None:
            self.targets = make_pick_place_phase_targets(
                pick_position  = self.pick_position,
                place_position = self.place_position,
                grasp_offsets  = self.grasp_offsets,
            )
        return self

    def reset(self, mode="idle", clear_targets=True):
        """
        Reset controller state.

        Parameters:
            mode (str): Mode to enter after reset.
            clear_targets (bool): Clear selected object and pick/place targets.

        Returns:
            InteractivePickAndPlaceLoopController: ``self``.
        """
        self.mode = str(mode)
        self.phase = "approach"
        self.idle_tick = 0
        self.phase_tick = 0
        self.done_tick = 0
        self.ik_err = np.inf
        self.ctrl_upper_body = None
        self.ctrl_gripper = None
        self.use_gripper_prev_q = None
        self.attach_active = False
        self.T_body_obj = None
        self.pending_attach_after_step = False
        if clear_targets:
            self.target_object = dict(body_name=None, metadata={})
            self.pick_position = None
            self.place_position = None
            self.targets = None
        return self

    def update(
            self,
            observation=None,
            selected_body_name=None,
            pick_requested=False,
            pick_position=None,
            place_position=None,
            targets=None,
            use_arm=None,
        ):
        """
        Update the state machine and return notebook-facing rollout info.

        Parameters:
            observation (dict | None): Optional external observation payload.
                Stored in rollout info for convenience, but not interpreted.
            selected_body_name (str | None): Current radio-button selection.
            pick_requested (bool): True when the outside UI says to start.
            pick_position, place_position (array-like | None): Externally
                computed 3D targets.
            targets (dict | None): Optional full phase target dictionary.
            use_arm (str | None): Optional arm override.

        Returns:
            dict: Rollout info with ``mode``, ``phase``, ``step``, ``render``,
                target metadata, and current IK error.
        """
        if selected_body_name is not None:
            self.set_target_object(body_name=selected_body_name)
        if pick_position is not None or place_position is not None or targets is not None or use_arm is not None:
            self.set_pick_place_targets(
                pick_position  = pick_position,
                place_position = place_position,
                targets        = targets,
                use_arm        = use_arm,
            )

        if self.mode == "idle":
            rollout = self.update_idle()
        elif self.mode == "choose_to_pick":
            rollout = self.update_choose_to_pick(pick_requested=pick_requested)
        elif self.mode == "execute":
            rollout = self.update_execute()
        elif self.mode == "done":
            rollout = self.make_rollout_info(status="done", reason="pick-and-place rollout is done.")
        else:
            raise ValueError("Unknown mode: %s" % self.mode)

        rollout["observation"] = observation
        self.last_rollout = rollout
        return rollout

    def update_idle(self):
        """
        Run the initial settle period, then switch to ``choose_to_pick``.

        Returns:
            dict: Rollout info.
        """
        if self.idle_tick >= self.idle_tick_goal:
            self.mode = "choose_to_pick"
            return self.make_rollout_info(reason="waiting for object selection.")

        self.idle_tick += 1
        return self.make_rollout_info(
            reason = "settling camera and scene.",
            step   = self.make_idle_step_info(),
        )

    def update_choose_to_pick(self, pick_requested=False):
        """
        Wait for selected object and externally computed pick/place targets.

        Parameters:
            pick_requested (bool): True when the outside UI triggers picking.

        Returns:
            dict: Rollout info.
        """
        ready = (
            bool(pick_requested)
            and self.target_object.get("body_name", None) is not None
            and self.targets is not None
            and self.use_arm is not None
        )
        if ready:
            self.start_execution()
            return self.make_rollout_info(reason="starting pick-and-place execution.")
        return self.make_rollout_info(reason="waiting for pick request and 3D targets.")

    def start_execution(self):
        """
        Initialize execution-phase state.

        Returns:
            InteractivePickAndPlaceLoopController: ``self``.
        """
        if self.env is None:
            raise ValueError("env is required before execution.")
        if self.ik_api is None:
            self.ik_api = FFWSG2IKSolverAPI(env=self.env, use_arm=self.use_arm, joints_use=self.upper_body_joint_names)
        self.ik_api.use_arm = self.use_arm
        if self.upper_body_joint_names is not None:
            self.ik_api.set_ik_config(joints_use=self.upper_body_joint_names, remake_solver=True)

        self.phase = "approach"
        self.phase_tick = 0
        self.done_tick = 0
        self.ik_err = np.inf
        self.ctrl_upper_body = self.env.get_qpos(joint_names=self.ik_api.joints_use)
        self.ctrl_gripper = self.env.get_qpos(joint_names=self.gripper_joint_names)
        self.use_gripper_prev_q = self.env.get_qpos(joint_names=["gripper_%s_joint1" % self.use_arm])
        self.attach_active = False
        self.T_body_obj = None
        self.pending_attach_after_step = False
        self.mode = "execute"
        return self

    def update_execute(self):
        """
        Advance one execution tick.

        This method is the phase dispatcher. Pose phases are routed to
        ``update_pose_phase`` where students implement IK target tracking.
        The ``grasp`` and ``release`` phases are provided as scaffold because
        they mainly handle gripper closure/opening and object attachment.

        Returns:
            dict: Rollout info containing a step command.
        """
        if self.phase in self.POSE_PHASES:
            self.update_pose_phase()
        elif self.phase == "grasp":
            self.update_grasp_phase()
        elif self.phase == "release":
            self.update_release_phase()
        elif self.phase == "done":
            self.ctrl_gripper = get_gripper_ctrl(self.use_arm, "open")
            self.done_tick += 1
            if self.done_tick >= self.max_done_tick:
                self.mode = "done"
                return self.make_rollout_info(status="done", reason="pick-and-place rollout is done.")

        self.pending_attach_after_step = bool(self.attach_active and self.phase != "release")
        return self.make_rollout_info(
            reason = "executing %s." % self.phase,
            step   = self.make_execute_step_info(),
        )

    def update_pose_phase(self):
        """
        Solve IK for a pose-tracking phase and advance when stable.

        FILL IN:
            Implement the pose-phase update used by ``approach``, ``pregrasp``,
            ``lift``, ``place_approach``, ``place``, and ``retreat``.

        Suggested steps:
            1. Read the current phase targets from ``self.targets[self.phase]``.
               Each target dictionary can contain guide points such as
               ``tcp``, ``top``, ``side``, and ``front``.
            2. Clear old IK targets and add the current phase targets to
               ``self.ik_api``.
            3. Call ``self.ik_api.solve_ik(env=self.env)`` and store the
               resulting upper-body control in ``self.ctrl_upper_body``.
            4. Store the IK error in ``self.ik_err`` so the phase can decide
               whether the pose is stable.
            5. Choose the gripper command for the phase. For example, approach
               and pregrasp usually keep the gripper open, while lift/place
               usually keep it closed.
            6. If an object is attached, keep updating its pose relative to the
               active gripper.
            7. Advance to the next phase after the IK error has stayed below
               ``self.ik_err_thr`` for ``self.hold_tick_goal`` ticks.

        Things to think about:
            - Why do several phases use IK targets while grasp/release do not?
            - What happens if only the TCP target is used and the guide targets
              are ignored?
            - How should the gripper state change across the phase sequence?

        Returns:
            None.
        """
        raise NotImplementedError(
            "FILL IN: implement IK target tracking and phase advancement in "
            "update_pose_phase."
        )

    def update_grasp_phase(self):
        """
        Close the selected gripper and attach after stable contact.

        Returns:
            None.
        """
        use_gripper_curr_q = self.env.get_qpos(joint_names=["gripper_%s_joint1" % self.use_arm])
        use_gripper_dq = use_gripper_curr_q - self.use_gripper_prev_q
        self.ctrl_gripper = get_gripper_ctrl(self.use_arm, "close")

        is_stable = np.all(np.abs(use_gripper_dq) < self.gripper_dq_thr) and np.all(use_gripper_curr_q > 0.2)
        self.phase_tick = self.phase_tick + 1 if bool(is_stable) else 0
        if self.phase_tick >= self.hold_tick_goal:
            T_w_obj = self.env.get_T(self.target_object["body_name"], type="body")
            T_w_tcp = self.env.get_T("gripper_%s_tcp" % self.use_arm, type="body")
            self.T_body_obj = view_in_local(T=T_w_obj, T_wl=T_w_tcp)
            self.attach_active = True
            self.phase = "lift"
            self.phase_tick = 0
            self.ik_err = np.inf
        self.use_gripper_prev_q = use_gripper_curr_q

    def update_release_phase(self):
        """
        Open the gripper, then release and switch to retreat.

        Returns:
            None.
        """
        if self.attach_active:
            self.attach_object()
        self.ctrl_gripper = get_gripper_ctrl(self.use_arm, "open")
        self.attach_active = False
        self.phase_tick += 1
        if self.phase_tick >= self.release_tick_goal:
            self.phase = "retreat"
            self.phase_tick = 0
            self.ik_err = np.inf

    def after_step(self):
        """
        Apply pending object attachment after the notebook steps MuJoCo.

        Returns:
            None.
        """
        if self.pending_attach_after_step:
            self.attach_object()
            self.pending_attach_after_step = False

    def attach_object(self):
        """
        Attach the selected object to the current gripper TCP pose.

        Returns:
            None.
        """
        if self.T_body_obj is None or self.target_object.get("body_name", None) is None:
            return
        attach_body_to_body(
            env           = self.env,
            obj_body_name = self.target_object["body_name"],
            body_name     = "gripper_%s_tcp" % self.use_arm,
            T_body_obj    = self.T_body_obj,
            obj_type      = "base_body",
            body_type     = "body",
        )

    def make_idle_step_info(self):
        """
        Build the idle settling step command.

        Returns:
            dict: Step command dictionary.
        """
        if self.env is None:
            return self.empty_step_info()
        if self.upper_body_joint_names is None and self.ik_api is not None:
            self.upper_body_joint_names = list(self.ik_api.joints_use)
        joint_names = [] if self.upper_body_joint_names is None else list(self.upper_body_joint_names)
        if len(joint_names) == 0:
            return self.empty_step_info()
        return dict(
            do_step = True,
            ctrl = self.env.get_qpos(joint_names=joint_names),
            joint_names = joint_names,
            nstep = self.idle_nstep,
            compensate_root_body_names = ["base_link"],
        )

    def make_execute_step_info(self):
        """
        Build the execution step command.

        FILL IN:
            Pack the controls computed by the controller into a dictionary that
            the notebook can pass to ``env.step(...)``.

        Expected dictionary fields:
            ``do_step``:
                True when a MuJoCo step should be executed.
            ``ctrl``:
                One control vector aligned with ``joint_names``. It should
                combine the IK upper-body control and the gripper command.
            ``joint_names``:
                Joint names corresponding to the entries of ``ctrl``.
            ``nstep``:
                Number of MuJoCo substeps for this control command.
            ``compensate_root_body_names``:
                Root bodies used by the notebook for gravity compensation.

        Things to think about:
            - The order of ``ctrl`` must match the order of ``joint_names``.
            - The IK API only controls upper-body joints; the gripper command
              must be appended separately.
            - If controls are not ready yet, returning ``empty_step_info`` is a
              reasonable no-op behavior.

        Returns:
            dict: Step command dictionary.
        """
        raise NotImplementedError(
            "FILL IN: package execution controls for env.step in "
            "make_execute_step_info."
        )

    def empty_step_info(self):
        """
        Build a no-op step command.

        Returns:
            dict: No-op step command dictionary.
        """
        return dict(
            do_step = False,
            ctrl = None,
            joint_names = [],
            nstep = 0,
            compensate_root_body_names = ["base_link"],
        )

    def make_render_info(self):
        """
        Build a lightweight render payload for the outside notebook.

        Returns:
            dict: Render information.
        """
        return dict(
            mode = self.mode,
            phase = self.phase if self.mode == "execute" else None,
            target_object = dict(self.target_object),
            pick_position = None if self.pick_position is None else self.pick_position.copy(),
            place_position = None if self.place_position is None else self.place_position.copy(),
            targets = self.targets,
            use_arm = self.use_arm,
            ik_api = self.ik_api,
            ik_err = self.ik_err,
            attach_active = self.attach_active,
        )

    def make_rollout_info(self, status="running", reason="", step=None):
        """
        Build the dictionary returned by ``update``.

        Parameters:
            status (str): Rollout status.
            reason (str): Human-readable status text.
            step (dict | None): Step command. If None, use no-op.

        Returns:
            dict: Rollout information.
        """
        return dict(
            status = status,
            reason = str(reason),
            mode = self.mode,
            phase = self.phase if self.mode == "execute" else None,
            target_object = dict(self.target_object),
            pick_position = self.pick_position,
            place_position = self.place_position,
            use_arm = self.use_arm,
            ik_err = self.ik_err,
            step = self.empty_step_info() if step is None else step,
            render = self.make_render_info(),
        )
