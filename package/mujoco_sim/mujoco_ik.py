"""
mujoco_ik.py
"""
import mujoco
import warnings
import numpy as np

from package.utility.utils import (
    get_idxs,
    TicToc,
    trim_scale,
)
from package.utility.viz import (
    get_colors,
    print_red,
)
from package.kinematics.transforms import (
    t2x,
    t2y,
    t2z,
)

def _normalize_lam_vec(lam,n,name="_normalize_lam_vec"):
    """
    Normalize damping to a vector of length n.

    Parameters:
        lam (float or array-like): Scalar damping or per-column damping vector.
        n   (int): Number of columns / joints.
        name (str): Name used in assertion messages.

    Returns:
        np.ndarray: Damping vector of shape (n,).
    """
    if np.isscalar(lam):
        lam_vec = np.full((n,),float(lam),dtype=np.float64)
    else:
        lam_vec = np.asarray(lam,dtype=np.float64).reshape(-1)
        assert len(lam_vec)==n,"[%s] len(lam)=[%d] != n=[%d]"%(name,len(lam_vec),n)

    if not np.isfinite(lam_vec).all():
        raise ValueError("[%s] lam contains non-finite values."%name)
    if np.any(lam_vec<0.0):
        raise ValueError("[%s] lam contains negative values."%name)

    lam_vec = np.maximum(lam_vec,1e-12)
    return lam_vec

def _solve_dls(J,e,lam):
    """
    Solve damped least squares in primal form:

        dq = argmin ||J dq - e||^2 + dq^T Lambda dq

    where
        - lam is scalar  -> Lambda = lam * I
        - lam is vector  -> Lambda = diag(lam)

    Parameters:
        J   (np.ndarray): Jacobian matrix (m x n)
        e   (np.ndarray): Error vector (m,)
        lam (float or array-like): Damping factor(s)
    """
    # Input type casting
    J = np.asarray(J,dtype=np.float64)
    e = np.asarray(e,dtype=np.float64).reshape(-1)
    m = int(J.shape[0])
    n = int(J.shape[1])

    # guard: empty inputs
    if (m==0) or (n==0):
        print("[_solve_dls] warning: empty inputs. Returning zeros.")
        return np.zeros((n,),dtype=np.float64)

    # guard: non-finite inputs
    if (not np.isfinite(J).all()) or (not np.isfinite(e).all()):
        print("[_solve_dls] warning: non-finite inputs. Returning zeros.")
        return np.zeros((n,),dtype=np.float64)

    # normalize damping to per-column vector
    try:
        lam_vec = _normalize_lam_vec(lam,n,name="_solve_dls")
    except Exception as ex:
        print("[_solve_dls] warning:",str(ex))
        return np.zeros((n,),dtype=np.float64)

    # scale J for numerical stability
    s = float(np.max(np.abs(J)))
    if (not np.isfinite(s)) or (s<=0.0):
        print("[_solve_dls] warning: non-finite or non-positive scale. Returning zeros.")
        print("[_solve_dls] max|J|=",s)
        return np.zeros((n,),dtype=np.float64)

    # guard: too small scale
    if s<1e-12:
        print("[_solve_dls] warning: too small scale. Returning zeros.")
        return np.zeros((n,),dtype=np.float64)

    # Scaled primal solve:
    #   min ||(J/s)dq - (e/s)||^2 + dq^T diag(lam/s^2) dq
    Js    = J/s
    es    = e/s
    lam_s = np.maximum(lam_vec/(s*s),1e-12)

    # guard: non-finite Js
    if not np.isfinite(Js).all():
        i,j = np.argwhere(~np.isfinite(Js))[0]
        print("[_solve_dls] Js non-finite at",i,j,"Js=",Js[i,j],"s=",s,"J=",J[i,j])
        return np.zeros((n,),dtype=np.float64)

    # guard: huge Js
    if np.max(np.abs(Js))>1e200:
        print("[_solve_dls] Js huge max=",np.max(np.abs(Js)),"s=",s,"max|J|=",np.max(np.abs(J)))
        return np.zeros((n,),dtype=np.float64)

    try:
        with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
            H   = Js.T@Js+np.diag(lam_s)
            rhs = Js.T@es
            dq  = np.linalg.solve(H,rhs)
    except Exception:
        dq = np.zeros((n,),dtype=np.float64)

    # guard: non-finite output
    if not np.isfinite(dq).all():
        print("[_solve_dls] warning: non-finite output. Returning zeros.")
        dq = np.zeros_like(dq)

    return dq

def _project_dq_null_damped(J_use,dq_null,lam):
    """
    Project dq_null with the same damped metric used in the primal DLS solve.

    We compute:
        corr   = argmin ||J x - J dq_null||^2 + x^T Lambda x
        dq_add = dq_null - corr

    Parameters:
        J_use   (np.ndarray): Jacobian matrix (m x k)
        dq_null (np.ndarray): Desired home-directed motion (k,)
        lam     (float or array-like): Damping factor(s) for the k joints

    Returns:
        np.ndarray: Projected nullspace-like motion (k,)
    """
    J_use   = np.asarray(J_use,dtype=np.float64)
    dq_null = np.asarray(dq_null,dtype=np.float64).reshape(-1)

    m = int(J_use.shape[0])
    k = int(J_use.shape[1])

    if (m==0) or (k==0):
        return np.zeros((k,),dtype=np.float64)

    if len(dq_null)!=k:
        return np.zeros((k,),dtype=np.float64)

    if (not np.isfinite(J_use).all()) or (not np.isfinite(dq_null).all()):
        return np.zeros((k,),dtype=np.float64)

    try:
        lam_vec = _normalize_lam_vec(lam,k,name="_project_dq_null_damped")
    except Exception:
        return np.zeros((k,),dtype=np.float64)

    s = float(np.max(np.abs(J_use)))
    if (not np.isfinite(s)) or (s<=0.0):
        return dq_null.copy()

    if s<1e-12:
        return dq_null.copy()

    Js    = J_use/s
    lam_s = np.maximum(lam_vec/(s*s),1e-12)

    try:
        with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
            H    = Js.T@Js+np.diag(lam_s)
            rhs  = Js.T@(Js@dq_null)
            corr = np.linalg.solve(H,rhs)
            dq_add = dq_null-corr
    except Exception:
        dq_add = np.zeros((k,),dtype=np.float64)

    if not np.isfinite(dq_add).all():
        dq_add = np.zeros((k,),dtype=np.float64)

    return dq_add




class SitePositionIKSolverRevPriBase(object):
    """
    Site-position IK solver using DLS (weighted least squares).
    """
    def __init__(
            self,
            name              = "Site Position IK Solver",
            ik_env            = None,
            joints_use        = None,
            max_ik_tick       = 100,
            ik_stepsize_rev   = np.deg2rad(3.0),
            ik_stepsize_pri   = 0.02,
            ik_update_th_rev  = np.deg2rad(5.0),
            ik_update_th_pri  = 0.05,
            dls_damping       = 1e-6,
            max_probe_rev     = np.deg2rad(3.0),
            max_probe_pri     = 0.02,
            k_null            = 0.01,
            q_home_rev_pri    = None,
        ):
        """
        This IK solver uses the revolute+prismatic joint ordering defined in ik_env.rev_pri_joint_names.
        It supports per-joint-type step sizes and probing limits, and base (free joint) control.

        Parameters:
            name (str): Name of the IK solver.
            ik_env: The IK environment object. Must have rev_pri_joint_names attribute.
            joints_use (list): List of joint names to use for IK. If None, use all rev_pri joints.
            max_ik_tick (int): Maximum number of IK iterations.
            ik_stepsize_rev (float): Step size for revolute joints. If None, use ik_stepsize.
            ik_stepsize_pri (float): Step size for prismatic joints. If None, use ik_stepsize.
            ik_update_th_rev (float): Threshold for IK update trimming for revolute joints.
            ik_update_th_pri (float): Threshold for IK update trimming for prismatic joints.
            dls_damping (float or array-like): Damping factor(s) for DLS.
                - scalar: same damping for all used joints
                - vector: per-joint damping, must match len(joints_use)
            max_probe_rev (float): Maximum probing angle for revolute joints. If None, use ik_stepsize_rev.
            max_probe_pri (float): Maximum probing distance for prismatic joints. If None, use ik_stepsize_pri.
            k_null (float): Gain for nullspace control towards home configuration.
            q_home_rev_pri (np.ndarray): Home configuration for rev_pri joints.
        """
        self.name   = name
        self.ik_env = ik_env
        assert self.ik_env is not None,"[%s] ik_env is None."%self.name
        assert hasattr(self.ik_env,"rev_pri_joint_names"),"[%s] ik_env has no rev_pri_joint_names."%self.name

        # Canonical rev_pri joint list
        self.rev_pri_joint_names = list(self.ik_env.rev_pri_joint_names)
        self._rev_pri_set        = set(self.rev_pri_joint_names)

        # Per-joint type mask in rev_pri order
        rev_set = set(self.ik_env.rev_joint_names)
        self._is_rev_in_rev_pri = np.array([j in rev_set for j in self.rev_pri_joint_names],dtype=bool)

        # Joint ids aligned with rev_pri order
        self._rev_pri_jidxs = np.array([self.ik_env.model.joint(j).id for j in self.rev_pri_joint_names],dtype=np.int32)

        # Limited + mins/maxs (unlimited -> +/-inf)
        limited = self.ik_env.model.jnt_limited[self._rev_pri_jidxs].astype(bool)
        mins    = self.ik_env.model.jnt_range[self._rev_pri_jidxs,0].astype(np.float64,copy=False).copy()
        maxs    = self.ik_env.model.jnt_range[self._rev_pri_jidxs,1].astype(np.float64,copy=False).copy()
        mins[~limited] = -np.inf
        maxs[~limited] =  np.inf
        self._rev_pri_limited   = limited
        self.rev_pri_joint_mins = mins
        self.rev_pri_joint_maxs = maxs

        # Active/passive over rev_pri.
        passive_rev_pri = []
        for eq_idx in range(int(self.ik_env.model.eq_data.shape[0])):
            if self.ik_env.model.eq_type[eq_idx]!=mujoco.mjtEq.mjEQ_JOINT:
                continue
            pj = self.ik_env.model.joint(int(self.ik_env.model.eq_obj1id[eq_idx])).name
            if pj in self._rev_pri_set:
                passive_rev_pri.append(pj)

        # unique keep order
        seen = set()
        passive_rev_pri = [j for j in passive_rev_pri if (j not in seen and not seen.add(j))]

        self.passive_joint_names_rev_pri = passive_rev_pri
        passive_set = set(self.passive_joint_names_rev_pri)
        self.active_joint_names_rev_pri  = [j for j in self.rev_pri_joint_names if j not in passive_set]
        self.idxs_rev_pri_active         = get_idxs(self.rev_pri_joint_names,self.active_joint_names_rev_pri)

        # joints_use normalization
        self.joints_use       = self._normalize_joints_use(joints_use)
        self.idxs_rev_pri_use = get_idxs(self.rev_pri_joint_names,self.joints_use)

        # Jacobian nv column indices for rev_pri joints (hinge+slide)
        self.idxs_jac_rev_pri = self.ik_env.get_idxs_jac(self.rev_pri_joint_names)

        self.set_ik_config(
            max_ik_tick      = max_ik_tick,
            ik_stepsize_rev  = ik_stepsize_rev,
            ik_stepsize_pri  = ik_stepsize_pri,
            ik_update_th_rev = ik_update_th_rev,
            ik_update_th_pri = ik_update_th_pri,
            dls_damping      = dls_damping,
            max_probe_rev    = max_probe_rev,
            max_probe_pri    = max_probe_pri,
            k_null           = k_null,
            q_home_rev_pri   = q_home_rev_pri,
        )

        self.reset_buffers()
        self.tt = TicToc(name='%s'%(self.name))

    # -----------------------------
    # helpers
    # -----------------------------
    def _normalize_joints_use(self,joints_use=None):
        """
        Normalize joints_use list.

        Parameters:
            joints_use (list): List of joint names to use for IK.
        """
        if joints_use is None:
            return list(self.rev_pri_joint_names)
        joints_use = list(joints_use)
        bad = [j for j in joints_use if j not in self._rev_pri_set]
        if len(bad)>0:
            raise ValueError("[%s] joints_use has unknown joints: %s"%(self.name,bad))
        return joints_use

    def _get_dls_damping_use(self):
        """
        Return damping for current joints_use.

        Returns:
            float or np.ndarray:
                - scalar damping, or
                - vector damping with len == len(self.joints_use)
        """
        dls = self.dls_damping
        if np.isscalar(dls):
            return float(dls)

        dls = np.asarray(dls,dtype=np.float64).reshape(-1)
        assert len(dls)==len(self.joints_use),(
            "[%s] len(dls_damping)=[%d] != len(joints_use)=[%d]"%
            (self.name,len(dls),len(self.joints_use))
        )
        return dls.copy()

    def _get_dls_damping_base(self):
        """
        Return scalar damping for base solve.

        Base DOFs are not part of joints_use, so when vector damping is used,
        we fall back to a scalar. We use the minimum damping value so that
        base motion is not over-damped relative to the most-permissive joint.
        """
        dls = self.dls_damping
        if np.isscalar(dls):
            return max(float(dls),1e-12)

        dls = np.asarray(dls,dtype=np.float64).reshape(-1)
        if len(dls)==0:
            return 1e-6
        return max(float(np.min(dls)),1e-12)

    def _get_rev_pri_qpos_idxs(self,model):
        """
        Get indices of rev_pri joint positions in qpos.

        Parameters:
            model (mujoco.MjModel): Mujoco model
        """
        rev_pri_qpos_idxs = []
        for jn in self.rev_pri_joint_names:
            jid  = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,jn)
            qadr = int(model.jnt_qposadr[jid])
            rev_pri_qpos_idxs.append(qadr)
        return np.asarray(rev_pri_qpos_idxs,dtype=int)

    def _get_base_dof_idxs(self,model):
        """
        Get indices of base (free joint) DOFs in velocity vector.

        Parameters:
            model (mujoco.MjModel): Mujoco model
        """
        free_jid = None
        for j_id in range(model.njnt):
            if model.jnt_type[j_id]==mujoco.mjtJoint.mjJNT_FREE:
                free_jid = j_id
                break
        if free_jid is None:
            return None
        dofadr = int(model.jnt_dofadr[free_jid])
        return np.arange(dofadr,dofadr+6,dtype=int)

    def _get_jacp_site_full(self,model,data,site_name):
        """
        Get full Jacobian position part for a site.

        Parameters:
            model (mujoco.MjModel): Mujoco model
            data (mujoco.MjData): Mujoco data
            site_name (str): Name of the site
        """
        sid  = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,site_name)
        jacp = np.zeros((3,model.nv),dtype=np.float64)
        jacr = np.zeros((3,model.nv),dtype=np.float64)
        mujoco.mj_jacSite(model,data,jacp,jacr,sid)
        return jacp

    def _build_weighted_Je_full(self,model,data):
        """
        Build weighted full Jacobian and error vectors.

        Parameters:
            model (mujoco.MjModel): Mujoco model
            data (mujoco.MjData): Mujoco data
        """
        J_blocks = []
        e_blocks = []
        for i in range(self.n_site):
            site_name            = self.site_names[i]
            p_curr               = self.ik_env.get_p(site_name,'site')
            self.p_curr_list[i]  = p_curr
            self.p_err_list[i]   = self.p_trgt_list[i]-p_curr
            self.J_p_list[i]     = self._get_jacp_site_full(model,data,site_name)

            w  = float(self.weight_list[i])
            ws = np.sqrt(w)
            J_blocks.append(ws*self.J_p_list[i])
            e_blocks.append(ws*self.p_err_list[i])

        J_p_array   = np.vstack(J_blocks)
        p_err_array = np.hstack(e_blocks)
        return J_p_array,p_err_array

    # -----------------------------
    # config
    # -----------------------------
    def set_ik_config(
            self,
            max_ik_tick      = None,
            ik_stepsize_rev  = None,
            ik_stepsize_pri  = None,
            ik_update_th_rev = None,
            ik_update_th_pri = None,
            dls_damping      = None,
            max_probe_rev    = None,
            max_probe_pri    = None,
            k_null           = None,
            q_home_rev_pri   = None,
        ):
        """
        Set IK solver configuration.

        Parameters:
            max_ik_tick (int): Maximum number of IK iterations.
            ik_stepsize_rev (float): Step size for revolute joints.
            ik_stepsize_pri (float): Step size for prismatic joints.
            ik_update_th_rev (float): Threshold for IK update trimming for revolute joints.
            ik_update_th_pri (float): Threshold for IK update trimming for prismatic joints.
            dls_damping (float or array-like): Damping factor(s) for DLS.
            max_probe_rev (float): Maximum probing angle for revolute joints.
            max_probe_pri (float): Maximum probing distance for prismatic joints.
            k_null (float): Gain for nullspace control towards home configuration.
            q_home_rev_pri (np.ndarray): Home configuration for rev_pri joints.
        """
        if max_ik_tick is not None:
            self.max_ik_tick = int(max_ik_tick)

        if ik_stepsize_rev is not None:
            self.ik_stepsize_rev = float(ik_stepsize_rev)
        else:
            self.ik_stepsize_rev = float(getattr(self,"ik_stepsize_rev",0.1))

        if ik_stepsize_pri is not None:
            self.ik_stepsize_pri = float(ik_stepsize_pri)
        else:
            self.ik_stepsize_pri = float(getattr(self,"ik_stepsize_pri",0.1))

        if ik_update_th_pri is not None:
            self.ik_update_th_pri = float(ik_update_th_pri)

        if ik_update_th_rev is not None:
            self.ik_update_th_rev = float(ik_update_th_rev)

        if dls_damping is not None:
            if np.isscalar(dls_damping):
                self.dls_damping = float(dls_damping)
            else:
                self.dls_damping = np.asarray(dls_damping,dtype=np.float64).reshape(-1).copy()

        if max_probe_rev is not None:
            self.max_probe_rev = float(max_probe_rev)

        if max_probe_pri is not None:
            self.max_probe_pri = float(max_probe_pri)

        if k_null is not None:
            self.k_null = float(k_null)

        if q_home_rev_pri is not None:
            self.q_home_rev_pri = np.asarray(q_home_rev_pri,dtype=float).copy()
            if len(self.q_home_rev_pri)!=len(self.rev_pri_joint_names):
                raise ValueError(
                    "[%s] len(q_home_rev_pri)=[%d] but len(rev_pri_joint_names)=[%d]"%
                    (self.name,len(self.q_home_rev_pri),len(self.rev_pri_joint_names))
                )
        else:
            if not hasattr(self,"q_home_rev_pri"):
                self.q_home_rev_pri = None

    # -----------------------------
    # buffers / targets
    # -----------------------------
    def reset_buffers(self,env=None):
        """
        Reset IK target buffers.
        """
        self.site_names   = []
        self.n_site       = 0
        self.p_curr_list  = []
        self.p_trgt_list  = []
        self.p_err_list   = []
        self.J_p_list     = []
        self.weight_list  = []

        if env is not None:
            self.sync_env(env)

    def add_ik_target(
            self,
            site_name = None,
            p_trgt    = None,
            weight    = 1.0,
        ):
        """
        Add an IK target.

        Parameters:
            site_name (str): Name of the target site.
            p_trgt (array-like): Target position (3,).
            weight (float): Weight of the target.
        """
        if weight is None:
            weight = 1.0
        weight = float(weight)
        if weight<=0.0 or (not np.isfinite(weight)):
            raise ValueError("weight must be positive and finite (got [%g])."%weight)

        self.site_names.append(site_name)
        self.n_site = len(self.site_names)

        p_trgt = np.asarray(p_trgt).reshape(3,)
        p_curr = self.ik_env.get_p(site_name,'site')
        perr   = p_trgt-p_curr
        J_p    = self.ik_env.get_J_site(site_name,return_type='pos')

        self.p_trgt_list.append(p_trgt)
        self.p_curr_list.append(p_curr)
        self.p_err_list.append(perr)
        self.J_p_list.append(J_p)
        self.weight_list.append(weight)

    def update_ik_state(self):
        """
        Update current positions, errors, and Jacobians for all IK targets.
        """
        for site_idx,site_name in enumerate(self.site_names):
            p_curr = self.ik_env.get_p(site_name,'site')
            self.p_curr_list[site_idx] = p_curr
            p_trgt = self.p_trgt_list[site_idx]
            self.p_err_list[site_idx]  = p_trgt-p_curr
            J_p = self.ik_env.get_J_site(site_name,return_type='pos')
            self.J_p_list[site_idx]    = J_p

    # -----------------------------
    # dq computations (NO state change)
    # -----------------------------
    def compute_dq_base(
            self,
            J_p_array,
            p_err_array,
            base_dof_idxs,
            lam,
            base_stepsize,
            base_pos_th,
            base_rot_th,
        ):
        """
        Compute base (free joint) delta.

        Parameters:
            J_p_array (np.ndarray): Full Jacobian position part.
            p_err_array (np.ndarray): Full position error vector.
            base_dof_idxs (np.ndarray): Indices of base DOFs.
            lam (float): Scalar damping factor for base DLS.
            base_stepsize (float): Step size for base control.
            base_pos_th (float): Position threshold for base trimming.
            base_rot_th (float): Rotation threshold for base trimming.
        """
        J_base = J_p_array[:,base_dof_idxs]
        dq6    = _solve_dls(J_base,p_err_array,lam)
        dq6    = float(base_stepsize)*dq6
        if not np.isfinite(dq6).all():
            dq6 = np.zeros_like(dq6)

        dp  = trim_scale(x=dq6[:3],th=base_pos_th)
        dr  = trim_scale(x=dq6[3:],th=base_rot_th)
        dq6 = np.hstack([dp,dr])
        if not np.isfinite(dq6).all():
            dq6 = np.zeros_like(dq6)
        return dq6

    def compute_dq_rev_pri(
            self,
            J_p_array,
            p_err_array,
            joint_limit_handle_flag,
            nullspace_control_flag,
        ):
        """
        Compute revolute+prismatic joint delta.

        This function computes a joint-space update dq for revolute and
        prismatic joints only. No state is modified inside this function.

        The procedure includes:
        - task-space DLS solve in joint coordinates
        - optional coupling constraints (mjEQ_JOINT)
        - type-wise step size handling (rev / prismatic)
        - optional joint-limit handling (two-pass probing)
        - optional nullspace pull-to-home

        Damping behavior:
        - scalar dls_damping:
            same damping for all used joints
        - vector dls_damping:
            per-joint damping for joints_use only
            len(dls_damping) must equal len(joints_use)

        Parameters:
            J_p_array (np.ndarray): Full Jacobian position part.
            p_err_array (np.ndarray): Full position error vector.
            joint_limit_handle_flag (bool): Whether to handle joint limits.
            nullspace_control_flag (bool): Whether to apply nullspace control towards home configuration.
        """
        # ------------------------------------------------------------
        # 1. Extract joint Jacobian (rev_pri DOFs)
        #    Slice the full task Jacobian to revolute+prismatic columns.
        # ------------------------------------------------------------
        J_rev_pri = J_p_array[:,self.idxs_jac_rev_pri]

        # ------------------------------------------------------------
        # 2. Coupling handling (mjEQ_JOINT, quartic constraints)
        #    If a joint is defined as passive = f(active),
        #    propagate its effect into the active joint Jacobian
        #    and suppress the passive column.
        # ------------------------------------------------------------
        J_rev_pri_coupled = J_rev_pri.copy()
        for eq_idx in range(self.ik_env.model.eq_data.shape[0]):
            if self.ik_env.model.eq_type[eq_idx]!=mujoco.mjtEq.mjEQ_JOINT:
                continue

            passive_joint = self.ik_env.model.joint(self.ik_env.model.eq_obj1id[eq_idx]).name
            active_joint  = self.ik_env.model.joint(self.ik_env.model.eq_obj2id[eq_idx]).name
            if (passive_joint not in self._rev_pri_set) or (active_joint not in self._rev_pri_set):
                continue

            coef        = self.ik_env.model.eq_data[eq_idx,:5]
            x0          = self.ik_env.model.joint(active_joint).qpos0[0]
            x           = self.ik_env.data.joint(active_joint).qpos[0]
            idx_passive = self.rev_pri_joint_names.index(passive_joint)
            idx_active  = self.rev_pri_joint_names.index(active_joint)
            dfdx        = self.ik_env.eval_quartic_derivative(x,coef,x0)

            J_rev_pri_coupled[:,idx_active] += J_rev_pri_coupled[:,idx_passive]*dfdx
            J_rev_pri_coupled[:,idx_passive] = 0.0

        # ------------------------------------------------------------
        # 3. Reduced solve on joints_use only
        #    This is where per-joint damping is applied.
        # ------------------------------------------------------------
        idxs_use = np.asarray(self.idxs_rev_pri_use,dtype=int)
        J_use    = J_rev_pri_coupled[:,idxs_use]
        lam_use  = self._get_dls_damping_use()

        dq_use = _solve_dls(J_use,p_err_array,lam_use)

        # ------------------------------------------------------------
        # 4. Type-wise step size application
        #    Revolute and prismatic joints can have different step sizes.
        # ------------------------------------------------------------
        is_rev_use = self._is_rev_in_rev_pri[idxs_use]
        steps_use  = np.where(
            is_rev_use,
            float(self.ik_stepsize_rev),
            float(self.ik_stepsize_pri),
        )
        dq_use = dq_use*steps_use
        if not np.isfinite(dq_use).all():
            dq_use = np.zeros_like(dq_use)

        # Current joint configuration (rev_pri order)
        q_rev_pri_curr = self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names)

        violated_idxs = np.array([],dtype=int)
        valid_local   = np.arange(len(idxs_use),dtype=int)

        # ------------------------------------------------------------
        # 5. [optional] Joint-limit handling (two-pass probing)
        #    Predict limit violations and re-solve with masked joints.
        # ------------------------------------------------------------
        if joint_limit_handle_flag:
            q_use = q_rev_pri_curr[idxs_use]

            # Type-wise probing distance/angle
            probe_use = np.where(
                is_rev_use,
                float(self.max_probe_rev),
                float(self.max_probe_pri),
            )

            q_use_next_probe = q_use+probe_use*np.sign(dq_use)

            q_mins_use = self.rev_pri_joint_mins[idxs_use]
            q_maxs_use = self.rev_pri_joint_maxs[idxs_use]

            hit_max       = (q_use_next_probe>q_maxs_use)&(dq_use>0)
            hit_min       = (q_use_next_probe<q_mins_use)&(dq_use<0)
            violated_idxs = np.where(hit_max|hit_min)[0]

            # Re-solve with violating joints removed from the reduced system
            if len(violated_idxs)>0:
                valid_mask  = np.ones(len(idxs_use),dtype=bool)
                valid_mask[violated_idxs] = False
                valid_local = np.where(valid_mask)[0]

                if len(valid_local)>0:
                    J_use2 = J_use[:,valid_local]

                    if np.isscalar(lam_use):
                        lam_use2 = float(lam_use)
                    else:
                        lam_use2 = np.asarray(lam_use,dtype=np.float64).reshape(-1)[valid_local]

                    dq_use2 = _solve_dls(J_use2,p_err_array,lam_use2)
                    dq_use2 = dq_use2*steps_use[valid_local]

                    dq_use = np.zeros_like(dq_use)
                    dq_use[valid_local] = dq_use2
                else:
                    dq_use = np.zeros_like(dq_use)

        if not np.isfinite(dq_use).all():
            dq_use = np.zeros_like(dq_use)

        # ------------------------------------------------------------
        # 6. [optional] Nullspace control (pull-to-home)
        #    Project a home-directed update into the damped task nullspace.
        # ------------------------------------------------------------
        if nullspace_control_flag and (self.q_home_rev_pri is not None):
            if len(valid_local)>0:
                idxs_use_valid = idxs_use[valid_local]
                J_use_valid    = J_use[:,valid_local]

                q_err_use = self.q_home_rev_pri[idxs_use_valid]-q_rev_pri_curr[idxs_use_valid]
                dq_null   = self.k_null*q_err_use

                if np.isscalar(lam_use):
                    lam_valid = float(lam_use)
                else:
                    lam_valid = np.asarray(lam_use,dtype=np.float64).reshape(-1)[valid_local]

                dq_add = _project_dq_null_damped(
                    J_use   = J_use_valid,
                    dq_null = dq_null,
                    lam     = lam_valid,
                )

                # Apply type-wise step size to nullspace update
                dq_add = dq_add*steps_use[valid_local]

                # Accumulate nullspace contribution
                dq_use[valid_local] += dq_add

        # ------------------------------------------------------------
        # 7. Scatter reduced solution back to full rev_pri vector
        # ------------------------------------------------------------
        dq_rev_pri = np.zeros((len(self.rev_pri_joint_names),),dtype=np.float64)
        dq_rev_pri[idxs_use] = dq_use

        # ------------------------------------------------------------
        # 8. Numerical safety and return
        # ------------------------------------------------------------
        if not np.isfinite(dq_rev_pri).all():
            dq_rev_pri = np.zeros_like(dq_rev_pri)

        return dq_rev_pri

    # -----------------------------
    # applying updates (state change)
    # -----------------------------
    def apply_base_delta(
            self,
            model,
            data,
            base_dof_idxs,
            dq6,
        ):
        """
        Apply base DOF delta.
        This function is used internally by solve_ik.

        Parameters:
            model (mujoco.MjModel): Mujoco model
            data (mujoco.MjData): Mujoco data
            base_dof_idxs (np.ndarray): Indices of base DOFs.
            dq6 (np.ndarray): Delta for base DOFs (6,).
        """
        dq_full = np.zeros(model.nv,dtype=np.float64)
        dq_full[base_dof_idxs] = dq6
        mujoco.mj_integratePos(model,data.qpos,dq_full,1.0)
        mujoco.mj_forward(model,data)

    def _dq_rev_pri_to_q_rev_pri_next(self,dq_rev_pri):
        """
        Convert dq_rev_pri to next q_rev_pri, enforcing coupling and limits.

        - Apply per-type update trimming:
        * revolute: ik_update_th_rev (rad)
        * prismatic: ik_update_th_pri (m)

        Parameters:
            dq_rev_pri (np.ndarray): Delta for rev_pri joints.

        Returns:
            q_rev_pri_next (np.ndarray): Next rev_pri joint positions.
        """
        q_rev_pri_curr = self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names)

        # active-only delta
        dq_active = dq_rev_pri[self.idxs_rev_pri_active]

        # split active deltas by joint type (rev/prismatic) and trim separately
        is_rev_active = self._is_rev_in_rev_pri[self.idxs_rev_pri_active]

        if np.any(is_rev_active):
            dq_active_rev            = dq_active[is_rev_active]
            dq_active_rev            = trim_scale(x=dq_active_rev,th=self.ik_update_th_rev)
            dq_active[is_rev_active] = dq_active_rev

        if np.any(~is_rev_active):
            dq_active_pri             = dq_active[~is_rev_active]
            dq_active_pri             = trim_scale(x=dq_active_pri,th=self.ik_update_th_pri)
            dq_active[~is_rev_active] = dq_active_pri

        # apply to active joints
        q_active = self.ik_env.get_qpos(joint_names=self.active_joint_names_rev_pri)+dq_active

        q_rev_pri_next = q_rev_pri_curr.copy()
        q_rev_pri_next[self.idxs_rev_pri_active] = q_active

        # enforce coupling constraints (quartic)
        for eq_idx in range(self.ik_env.model.eq_data.shape[0]):
            if self.ik_env.model.eq_type[eq_idx]!=mujoco.mjtEq.mjEQ_JOINT:
                continue

            passive_joint = self.ik_env.model.joint(self.ik_env.model.eq_obj1id[eq_idx]).name
            active_joint  = self.ik_env.model.joint(self.ik_env.model.eq_obj2id[eq_idx]).name
            if (passive_joint not in self._rev_pri_set) or (active_joint not in self._rev_pri_set):
                continue

            coef = self.ik_env.model.eq_data[eq_idx,:5]
            x0   = self.ik_env.model.joint(active_joint).qpos0[0]
            x    = q_rev_pri_next[self.rev_pri_joint_names.index(active_joint)]
            y    = self.ik_env.eval_quartic(x=x,coef=coef,x0=x0)
            q_rev_pri_next[self.rev_pri_joint_names.index(passive_joint)] = y

        # clip only limited joints (unlimited are +/-inf)
        q_rev_pri_next = np.clip(q_rev_pri_next,self.rev_pri_joint_mins,self.rev_pri_joint_maxs)
        return q_rev_pri_next

    def apply_rev_pri_qpos(
            self,
            model,
            data,
            rev_pri_qpos_idxs,
            q_rev_pri_next,
        ):
        """
        Apply revolute+prismatic joint positions.

        Parameters:
            model (mujoco.MjModel): Mujoco model
            data (mujoco.MjData): Mujoco data
            rev_pri_qpos_idxs (np.ndarray): Indices of rev_pri joint positions in qpos.
            q_rev_pri_next (np.ndarray): Next rev_pri joint positions
        """
        data.qpos[rev_pri_qpos_idxs] = q_rev_pri_next
        mujoco.mj_forward(model,data)

    # -----------------------------
    # public entry
    # -----------------------------
    def sync_env(self,env):
        """
        Sync the IK environment state with the given environment.

        Parameters:
            env: The environment to sync from. Must have compatible state structure.
        """
        self.ik_env.data.qpos[:] = env.data.qpos.copy()
        for body_name in env.body_names:
            self.ik_env.model.body(body_name).pos  = env.model.body(body_name).pos.copy()
            self.ik_env.model.body(body_name).quat = env.model.body(body_name).quat.copy()
        mujoco.mj_forward(self.ik_env.model,self.ik_env.data)

    def solve_ik(
            self,
            env                     = None,
            joints_use              = None,
            joint_limit_handle_flag = True,
            nullspace_control_flag  = True,
            init_from_median_flag   = False,
            base_control_flag       = False,
            base_stepsize           = 0.5, # meters
            base_pos_th             = 0.1, # meters
            base_rot_th             = np.deg2rad(5.0), # radians
        ):
        """
        Solve inverse kinematics for the current IK targets.
        It returns the best revolute+prismatic joint configuration found.

        solve_ik() procedure:
        -----------------
        ├─ State synchronization
        │   └─ Sync full MuJoCo state from external env
        │       (qpos + body pose) to ik_env
        │
        ├─ Problem setup
        │   ├─ Normalize joints_use
        │   │   └─ Supports: revolute only / prismatic only / mixed rev+pri
        │   ├─ [optional] Median initialization
        │   └─ Initial update_ik_state
        │
        ├─ Iterative IK loop
        │
        │   ├─ Task construction (task-space)
        │   │   ├─ Build stacked position Jacobian for all sites
        │   │   ├─ Apply per-site weights
        │   │   └─ Form a single least-squares task
        │   │
        │   ├─ [optional] Base motion (free joint)
        │   │   ├─ Solve DLS for base DOFs
        │   │   ├─ Apply step size and safety thresholds
        │   │   └─ Integrate base motion and forward kinematics
        │   │
        │   ├─ Joint-space update (rev + prismatic)
        │   │   ├─ Extract joint Jacobian in rev_pri order
        │   │   │
        │   │   ├─ Coupling handling
        │   │   │   ├─ Detect mjEQ_JOINT constraints
        │   │   │   ├─ Propagate passive joint effects to active joints
        │   │   │   └─ Suppress passive Jacobian columns
        │   │   │
        │   │   ├─ Joint selection
        │   │   │   └─ Reduced solve on joints_use
        │   │   │       with scalar/vector damping
        │   │   │
        │   │   ├─ Primary task solve
        │   │   │   ├─ Solve DLS in joint space
        │   │   │   └─ Apply type-wise step size
        │   │   │       (revolute vs prismatic)
        │   │   │
        │   │   ├─ [optional] Joint-limit handling
        │   │   │   ├─ Probe next configuration (type-wise probe)
        │   │   │   ├─ Detect limit-violating joints
        │   │   │   └─ Re-solve with violated joints removed
        │   │   │
        │   │   └─ [optional] Nullspace control
        │   │       ├─ Project toward home configuration
        │   │       ├─ Respect joint-limit masking
        │   │       └─ Use the same damping metric
        │   │
        │   ├─ dq → q conversion
        │   │   ├─ Update active joints only
        │   │   ├─ Trim per-joint update magnitude
        │   │   ├─ Enforce coupling constraints (quartic)
        │   │   └─ Clip to joint limits
        │   │
        │   ├─ State commit
        │   │   ├─ Write joint positions to qpos
        │   │   └─ Forward kinematics
        │   │
        │   └─ Task state refresh
        │       └─ update_ik_state and log error
        │
        └─ Solution selection
            └─ Choose iteration with minimum task error

        Parameters:
            env: The environment object to sync state from.
            joints_use (list): List of joint names to use for IK. If None, use all rev_pri joints.
            joint_limit_handle_flag (bool): Whether to handle joint limits.
            nullspace_control_flag (bool): Whether to apply nullspace control towards home configuration.
            init_from_median_flag (bool): Whether to initialize from median joint configuration.
            base_control_flag (bool): Whether to control base (free joint) DOFs.
            base_stepsize (float): Step size for base control.
            base_pos_th (float): Position threshold for base trimming.
            base_rot_th (float): Rotation threshold for base trimming.
        """
        # ------------------------------------------------------------
        # 1. State synchronization (external env -> ik_env)
        #    Bring ik_env to exactly the same kinematic state as env.
        #    This allows IK to be solved on a shadow environment.
        # ------------------------------------------------------------
        if env is not None:
            self.ik_env.data.qpos[:] = env.data.qpos.copy()
            for body_name in env.body_names:
                self.ik_env.model.body(body_name).pos  = env.model.body(body_name).pos.copy()
                self.ik_env.model.body(body_name).quat = env.model.body(body_name).quat.copy()
            mujoco.mj_forward(self.ik_env.model,self.ik_env.data)

        # ------------------------------------------------------------
        # 2. Guard: no IK targets
        #    If no site targets are specified, return current state.
        # ------------------------------------------------------------
        if len(self.site_names)==0:
            q_rev_pri_curr = self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names)
            info = {
                'ik_err_array':np.array([0.0]),
                'idx_best':0,
                'ik_err_best':0.0,
                'elapsed_time':0.0,
                'qpos_full_best':self.ik_env.data.qpos.copy(),
            }
            return q_rev_pri_curr,info

        # ------------------------------------------------------------
        # 3. Joint selection
        #    Normalize joints_use and build index mapping.
        #    Supports revolute-only / prismatic-only / mixed cases.
        # ------------------------------------------------------------
        if joints_use is not None:
            self.joints_use = self._normalize_joints_use(joints_use)
        self.idxs_rev_pri_use = get_idxs(self.rev_pri_joint_names,self.joints_use)

        # ------------------------------------------------------------
        # 4. Optional initialization from median joint configuration
        #    Useful for escaping poor initial states.
        # ------------------------------------------------------------
        if init_from_median_flag:
            q_rev_pri_median = 0.5*(self.rev_pri_joint_mins+self.rev_pri_joint_maxs)
            q_curr = self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names)
            bad    = ~np.isfinite(q_rev_pri_median)
            if np.any(bad):
                q_rev_pri_median[bad] = q_curr[bad]
            self.ik_env.forward(q=q_rev_pri_median,joint_names=self.rev_pri_joint_names)

        # ------------------------------------------------------------
        # 5. Cache model/data handles and index mappings
        # ------------------------------------------------------------
        model = self.ik_env.model
        data  = self.ik_env.data

        # qpos indices corresponding to rev_pri joints
        rev_pri_qpos_idxs = self._get_rev_pri_qpos_idxs(model)

        # base (free joint) DOFs, if enabled
        base_dof_idxs = None
        if base_control_flag:
            base_dof_idxs = self._get_base_dof_idxs(model)

        # ------------------------------------------------------------
        # 6. Initial task state update
        #    Initial update_ik_state to sync positions, errors, Jacobians
        #    for all IK targets.
        # ------------------------------------------------------------
        self.update_ik_state()

        # ------------------------------------------------------------
        # 7. Logging buffers for all IK iterations
        # ------------------------------------------------------------
        q_rev_pri_list = []
        qpos_list      = []
        ik_err_array   = np.zeros(self.max_ik_tick+1)

        q_rev_pri_list.append(self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names))
        qpos_list.append(self.ik_env.data.qpos.copy())
        ik_err_array[0] = float(np.linalg.norm(np.hstack(self.p_err_list)))

        # Scalar fallback damping for base solve
        lam_base = self._get_dls_damping_base()

        # ------------------------------------------------------------
        # 8. Main IK iteration loop
        # ------------------------------------------------------------
        self.tt.tic()
        for ik_tick in range(self.max_ik_tick):

            # --------------------------------------------------------
            # 8.1 Build weighted task Jacobian and error
            # --------------------------------------------------------
            J_p_array,p_err_array = self._build_weighted_Je_full(model,data)

            # --------------------------------------------------------
            # 8.2 Optional base (free joint) update
            #     Performed before joint updates to reduce large errors.
            # --------------------------------------------------------
            if base_control_flag and (base_dof_idxs is not None):
                dq6 = self.compute_dq_base(
                    J_p_array,p_err_array,
                    base_dof_idxs,
                    lam_base,
                    base_stepsize,base_pos_th,base_rot_th
                ) # (6,)
                self.apply_base_delta(model,data,base_dof_idxs,dq6)

                # Base motion changes kinematics -> refresh task state
                self.update_ik_state()
                J_p_array,p_err_array = self._build_weighted_Je_full(model,data)

            # --------------------------------------------------------
            # 8.3 Joint-space update (revolute + prismatic)
            #     Includes coupling, joint limits, and nullspace control.
            # --------------------------------------------------------
            dq_rev_pri = self.compute_dq_rev_pri(
                J_p_array,
                p_err_array,
                joint_limit_handle_flag,
                nullspace_control_flag,
            ) # (n_rev_pri,)

            # Convert dq to a valid next joint configuration
            q_rev_pri_next = self._dq_rev_pri_to_q_rev_pri_next(dq_rev_pri)

            # Commit joint update to MuJoCo state
            self.apply_rev_pri_qpos(model,data,rev_pri_qpos_idxs,q_rev_pri_next)

            # Refresh task state after joint update
            self.update_ik_state()

            # --------------------------------------------------------
            # 8.4 Log iteration result
            # --------------------------------------------------------
            q_rev_pri_list.append(self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names))
            qpos_list.append(self.ik_env.data.qpos.copy())
            ik_err_array[ik_tick+1] = float(np.linalg.norm(np.hstack(self.p_err_list)))

        # ------------------------------------------------------------
        # 9. Select best solution over all iterations
        # ------------------------------------------------------------
        idx_best       = int(np.argmin(ik_err_array))
        q_rev_pri_best = q_rev_pri_list[idx_best]
        qpos_full_best = qpos_list[idx_best].copy()
        ik_err_best    = float(ik_err_array[idx_best])
        elapsed_time   = float(self.tt.toc())

        idxs_use       = np.asarray(self.idxs_rev_pri_use,dtype=int)
        qpos_used_best = qpos_full_best[rev_pri_qpos_idxs[idxs_use]].copy()

        # Return best solution and info
        info = {
            'ik_err_array':ik_err_array,
            'idx_best':idx_best,
            'ik_err_best':ik_err_best,
            'elapsed_time':elapsed_time,
            'qpos_full_best':qpos_full_best,
            'qpos_used_best':qpos_used_best,
            "joints_use":self.joints_use,
        }
        return q_rev_pri_best,info

    def render(
            self,
            env,
            r                      = 0.02,
            alpha                  = 1.0,
            cmap_name              = 'rainbow',
            show_weights           = False,
            scale_radius_by_weight = False,
            plot_arrows            = False,
            rgba                   = None,
        ):
        """
        Render IK targets and current positions in the given environment.

        Parameters:
            env: The environment object to render in.
            r (float): Radius of the spheres to render.
            alpha (float): Alpha value for the colors.
            cmap_name (str): Colormap name for coloring the targets.
            show_weights (bool): Whether to show weights as labels.
            scale_radius_by_weight (bool): Whether to scale sphere radius by weight.
            plot_arrows (bool): Whether to plot arrows from current to target positions.
            rgba (array-like): Optional RGBA color to use for all targets (overrides colormap).
        """
        n_trgt = len(self.site_names) # number of targets
        if n_trgt<=0:
            return # nothing to render
        colors = get_colors(cmap_name=cmap_name,n_color=n_trgt,alpha=alpha)

        for i,site_name in enumerate(self.site_names): # plot each target
            p_trgt = self.p_trgt_list[i]
            p_curr = env.get_p(site_name,'site')
            rgba_i = colors[i] if rgba is None else rgba
            weight = self.weight_list[i]
            if scale_radius_by_weight:
                r_use = max(r*weight,0.005)
            else:
                r_use = r

            # Plot current position
            env.plot_sphere(p_curr,r_use,rgba_i)

            if show_weights:
                label = "%.2f"%(weight)
            else:
                label = ""

            # Plot target position
            env.plot_sphere(p_trgt,r_use,rgba_i,label=label)

            # Plot arrow from current to target
            if plot_arrows:
                env.plot_arrow_fr2to(p_fr=p_curr,p_to=p_trgt,rgba=rgba_i,r=r_use*0.5)
            else:
                env.plot_line_fr2to(p_fr=p_curr,p_to=p_trgt,rgba=rgba_i)


































class BodyPositionIKSolverRevPriBase(object):
    """
    Body-point position IK solver using DLS (weighted least squares).
    - Joint space: env.rev_pri_joint_names (hinge+slide)
    - Targets are arbitrary points attached to bodies (via body-local point inferred from p_curr)
    - Jacobian computed by mujoco.mj_jac(model,data,jacp,jacr,p_world,body_id)
    - Supports coupling constraints (mjEQ_JOINT quartic), joint limits, nullspace pull-to-home
    - Optional base (free joint) control

    Simple usage:
    # Create a shadow env for IK
    ik_env    = MuJoCoParser(rel_xml_path=env.rel_xml_path)
    ik_solver = BodyPositionIKSolverRevPriBase(
        ik_env            = ik_env,
        max_ik_tick       = 1,
        ik_stepsize_rev   = 3*D2R,
        ik_stepsize_pri   = 0.02,
        ik_update_th_rev  = 5*D2R,
        ik_update_th_pri  = 0.05,
        dls_damping       = 1e-8,
        max_probe_rev     = 3*D2R,
        max_probe_pri     = 0.02,
        k_null            = 0.5,
        q_home_rev_pri    = env.get_qpos(joint_names=env.rev_pri_joint_names),
    )

    # (A) Single body-point depenetration target
    # body_idx: MuJoCo body id (int)
    # p_curr  : current point position in world frame (3,)
    # p_trgt  : desired point position in world frame (3,)
    ik_solver.reset_buffers(env=env)
    ik_solver.add_ik_target(body_idx=body_idx,p_curr=p_contact,p_trgt=p_contact+0.02*push_dir,weight=1.0)

    # (B) Multiple targets (e.g., from contacts)
    # ik_solver.reset_buffers(env=env)
    # for each contact:
    #     if body1_idx in body_idxs_robot:
    #         ik_solver.add_ik_target(body_idx=body1_idx,p_curr=p_contact,p_trgt=p_contact-len_dir*norm_dir,weight=1.0)
    #     if body2_idx in body_idxs_robot:
    #         ik_solver.add_ik_target(body_idx=body2_idx,p_curr=p_contact,p_trgt=p_contact+len_dir*norm_dir,weight=1.0)

    # Solve IK on the shadow env (syncs from env internally)
    q_rev_pri_best,info = ik_solver.solve_ik(
        env                     = env,
        joints_use              = env.rev_pri_joint_names,  # or a subset
        joint_limit_handle_flag = True,
        nullspace_control_flag  = False,
        init_from_median_flag   = False,
        base_control_flag       = False,
        base_stepsize           = 0.5,
        base_pos_th             = 0.1,
        base_rot_th             = 5*D2R,
    )

    # Apply solution to the real env if desired
    env.forward(q=q_rev_pri_best,joint_names=env.rev_pri_joint_names)

    # Optional visualization of target points vs current
    ik_solver.render(env,r=0.005,alpha=0.2,cmap_name='winter',plot_arrows=True)
    """
    def __init__(
            self,
            name              = "Body Position IK Solver",
            ik_env            = None,
            joints_use        = None,
            max_ik_tick       = 10,
            ik_stepsize_rev   = np.deg2rad(3.0),
            ik_stepsize_pri   = 0.05,
            ik_update_th_rev  = np.deg2rad(5.0),
            ik_update_th_pri  = 0.05,
            dls_damping       = 1e-6,
            max_probe_rev     = np.deg2rad(3.0),
            max_probe_pri     = 0.01,
            k_null            = 0.2,
            q_home_rev_pri    = None,
        ):
        self.name   = name
        self.ik_env = ik_env
        assert self.ik_env is not None,"[%s] ik_env is None."%self.name
        assert hasattr(self.ik_env,"rev_pri_joint_names"),"[%s] ik_env has no rev_pri_joint_names."%self.name

        self.rev_pri_joint_names = list(self.ik_env.rev_pri_joint_names)
        self._rev_pri_set        = set(self.rev_pri_joint_names)

        rev_set = set(self.ik_env.rev_joint_names)
        self._is_rev_in_rev_pri = np.array([j in rev_set for j in self.rev_pri_joint_names],dtype=bool)

        self._rev_pri_jidxs = np.array([self.ik_env.model.joint(j).id for j in self.rev_pri_joint_names],dtype=np.int32)

        limited = self.ik_env.model.jnt_limited[self._rev_pri_jidxs].astype(bool)
        mins    = self.ik_env.model.jnt_range[self._rev_pri_jidxs,0].astype(np.float64,copy=False).copy()
        maxs    = self.ik_env.model.jnt_range[self._rev_pri_jidxs,1].astype(np.float64,copy=False).copy()
        mins[~limited] = -np.inf
        maxs[~limited] =  np.inf
        self._rev_pri_limited   = limited
        self.rev_pri_joint_mins = mins
        self.rev_pri_joint_maxs = maxs

        passive_rev_pri = []
        for eq_idx in range(int(self.ik_env.model.eq_data.shape[0])):
            if self.ik_env.model.eq_type[eq_idx]!=mujoco.mjtEq.mjEQ_JOINT:
                continue
            pj = self.ik_env.model.joint(int(self.ik_env.model.eq_obj1id[eq_idx])).name
            if pj in self._rev_pri_set:
                passive_rev_pri.append(pj)

        seen = set()
        passive_rev_pri = [j for j in passive_rev_pri if (j not in seen and not seen.add(j))]

        self.passive_joint_names_rev_pri = passive_rev_pri
        passive_set = set(self.passive_joint_names_rev_pri)
        self.active_joint_names_rev_pri  = [j for j in self.rev_pri_joint_names if j not in passive_set]
        self.idxs_rev_pri_active         = get_idxs(self.rev_pri_joint_names,self.active_joint_names_rev_pri)

        self.joints_use       = self._normalize_joints_use(joints_use)
        self.idxs_rev_pri_use = get_idxs(self.rev_pri_joint_names,self.joints_use)

        self.idxs_jac_rev_pri = self.ik_env.get_idxs_jac(self.rev_pri_joint_names)

        self.set_ik_config(
            max_ik_tick      = max_ik_tick,
            ik_stepsize_rev  = ik_stepsize_rev,
            ik_stepsize_pri  = ik_stepsize_pri,
            ik_update_th_rev = ik_update_th_rev,
            ik_update_th_pri = ik_update_th_pri,
            dls_damping      = dls_damping,
            max_probe_rev    = max_probe_rev,
            max_probe_pri    = max_probe_pri,
            k_null           = k_null,
            q_home_rev_pri   = q_home_rev_pri,
        )

        self.reset_buffers()
        self.tt = TicToc(name='%s'%(self.name))

    # -----------------------------
    # helpers
    # -----------------------------
    def _normalize_joints_use(self,joints_use=None):
        """
        Normalize joints_use to ensure it's a valid subset of rev_pri_joint_names.

        Parameters:
            joints_use (list): List of joint names to use for IK. If None, use all rev_pri joints.
        """
        if joints_use is None:
            return list(self.rev_pri_joint_names)
        joints_use = list(joints_use)
        bad = [j for j in joints_use if j not in self._rev_pri_set]
        if len(bad)>0:
            raise ValueError("[%s] joints_use has unknown joints: %s"%(self.name,bad))
        return joints_use

    def _get_dls_damping_use(self):
        """
        Return damping for current joints_use.

        Returns:
            float or np.ndarray:
                - scalar damping, or
                - vector damping with len == len(self.joints_use)
        """
        dls = self.dls_damping
        if np.isscalar(dls):
            return float(dls)

        dls = np.asarray(dls,dtype=np.float64).reshape(-1)
        assert len(dls)==len(self.joints_use),(
            "[%s] len(dls_damping)=[%d] != len(joints_use)=[%d]"%
            (self.name,len(dls),len(self.joints_use))
        )
        return dls.copy()

    def _get_dls_damping_base(self):
        """
        Return scalar damping for base solve.

        Base DOFs are not part of joints_use, so when vector damping is used,
        we fall back to a scalar. We use the minimum damping value so that
        base motion is not over-damped relative to the most-permissive joint.
        """
        dls = self.dls_damping
        if np.isscalar(dls):
            return max(float(dls),1e-12)

        dls = np.asarray(dls,dtype=np.float64).reshape(-1)
        if len(dls)==0:
            return 1e-6
        return max(float(np.min(dls)),1e-12)

    def _get_rev_pri_qpos_idxs(self,model):
        """
        Get the indices of the qpos elements corresponding to the rev_pri joints.

        Parameters:
            model: The MuJoCo model.

        Returns:
            np.ndarray: Array of qpos indices for the rev_pri joints.
        """
        rev_pri_qpos_idxs = []
        for jn in self.rev_pri_joint_names:
            jid  = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,jn)
            qadr = int(model.jnt_qposadr[jid])
            rev_pri_qpos_idxs.append(qadr)
        return np.asarray(rev_pri_qpos_idxs,dtype=int)

    def _get_base_dof_idxs(self,model):
        """
        Get the indices of the qpos elements corresponding to the base degrees of freedom (DOFs).

        Parameters:
            model: The MuJoCo model.

        Returns:
            np.ndarray or None: Array of qpos indices for the base DOFs, or None if no free joint is found.
        """
        free_jid = None
        for j_id in range(model.njnt):
            if model.jnt_type[j_id]==mujoco.mjtJoint.mjJNT_FREE:
                free_jid = j_id
                break
        if free_jid is None:
            return None
        dofadr = int(model.jnt_dofadr[free_jid])
        return np.arange(dofadr,dofadr+6,dtype=int)

    def _body_xmat(self,data,body_id):
        """
        Get the rotation matrix of a body.

        Parameters:
            data: The MuJoCo data.
            body_id: The ID of the body.

        Returns:
            np.ndarray: 3x3 rotation matrix of the body.
        """
        return data.xmat[body_id].reshape(3,3)

    def _body_xpos(self,data,body_id):
        """
        Get the position of a body.

        Parameters:
            data: The MuJoCo data.
            body_id: The ID of the body.

        Returns:
            np.ndarray: 3D position of the body.
        """
        return data.xpos[body_id].copy()

    def _p_world_from_local(self,data,body_id,p_local):
        """
        Convert a point from local body coordinates to world coordinates.

        Parameters:
            data: The MuJoCo data.
            body_id: The ID of the body.
            p_local: The point in local body coordinates.

        Returns:
            np.ndarray: The point in world coordinates.
        """
        R = self._body_xmat(data,body_id)
        p = self._body_xpos(data,body_id)
        return p + R@p_local

    def _p_local_from_world(self,data,body_id,p_world):
        """
        Convert a point from world coordinates to local body coordinates.

        Parameters:
            data: The MuJoCo data.
            body_id: The ID of the body.
            p_world: The point in world coordinates.

        Returns:
            np.ndarray: The point in local body coordinates.
        """
        R = self._body_xmat(data,body_id)
        p = self._body_xpos(data,body_id)
        return R.T@(p_world-p)

    def _get_jacp_point_full(self,model,data,body_id,p_world):
        """
        Get the positional Jacobian of a point on a body.

        Parameters:
            model: The MuJoCo model.
            data: The MuJoCo data.
            body_id: The ID of the body.
            p_world: The point in world coordinates.

        Returns:
            np.ndarray: The 3xN positional Jacobian of the point, where N is the number of degrees of freedom.
        """
        jacp = np.zeros((3,model.nv),dtype=np.float64)
        jacr = np.zeros((3,model.nv),dtype=np.float64)
        mujoco.mj_jac(model,data,jacp,jacr,p_world,int(body_id))
        return jacp

    def _build_weighted_Je_full(self,model,data):
        """
        Build the weighted Jacobian and error vectors for all target points.

        Parameters:
            model: The MuJoCo model.
            data: The MuJoCo data.

        Returns:
            tuple: A tuple containing the weighted Jacobian matrix and the weighted error vector.
        """
        J_blocks = []
        e_blocks = []
        for i in range(self.n_target):
            body_id             = int(self.body_idx_list[i])
            p_local             = self.p_local_list[i]
            p_curr              = self._p_world_from_local(data,body_id,p_local)
            self.p_curr_list[i] = p_curr
            self.p_err_list[i]  = self.p_trgt_list[i]-p_curr
            self.J_p_list[i]    = self._get_jacp_point_full(model,data,body_id,p_curr)

            w  = float(self.weight_list[i])
            ws = np.sqrt(w)
            J_blocks.append(ws*self.J_p_list[i])
            e_blocks.append(ws*self.p_err_list[i])

        J_p_array   = np.vstack(J_blocks)
        p_err_array = np.hstack(e_blocks)
        return J_p_array,p_err_array

    # -----------------------------
    # config
    # -----------------------------
    def set_ik_config(
            self,
            max_ik_tick      = None,
            ik_stepsize_rev  = None,
            ik_stepsize_pri  = None,
            ik_update_th_rev = None,
            ik_update_th_pri = None,
            dls_damping      = None,
            max_probe_rev    = None,
            max_probe_pri    = None,
            k_null           = None,
            q_home_rev_pri   = None,
        ):
        """
        Set the inverse kinematics (IK) configuration parameters.

        Parameters:
            max_ik_tick: Maximum number of IK iterations.
            ik_stepsize_rev: Step size for the reverse IK solver.
            ik_stepsize_pri: Step size for the primary IK solver.
            ik_update_th_rev: Update threshold for the reverse IK solver.
            ik_update_th_pri: Update threshold for the primary IK solver.
            dls_damping: Damped least squares damping factor(s).
            max_probe_rev: Maximum probe distance for the reverse IK solver.
            max_probe_pri: Maximum probe distance for the primary IK solver.
            k_null: Null space gain.
            q_home_rev_pri: Home configuration for the reverse and primary IK solvers.
        """
        if max_ik_tick is not None:
            self.max_ik_tick = int(max_ik_tick)

        if ik_stepsize_rev is not None:
            self.ik_stepsize_rev = float(ik_stepsize_rev)
        else:
            self.ik_stepsize_rev = float(getattr(self,"ik_stepsize_rev",0.1))

        if ik_stepsize_pri is not None:
            self.ik_stepsize_pri = float(ik_stepsize_pri)
        else:
            self.ik_stepsize_pri = float(getattr(self,"ik_stepsize_pri",0.1))

        if ik_update_th_pri is not None:
            self.ik_update_th_pri = float(ik_update_th_pri)

        if ik_update_th_rev is not None:
            self.ik_update_th_rev = float(ik_update_th_rev)

        if dls_damping is not None:
            if np.isscalar(dls_damping):
                self.dls_damping = float(dls_damping)
            else:
                self.dls_damping = np.asarray(dls_damping,dtype=np.float64).reshape(-1).copy()

        if max_probe_rev is not None:
            self.max_probe_rev = float(max_probe_rev)

        if max_probe_pri is not None:
            self.max_probe_pri = float(max_probe_pri)

        if k_null is not None:
            self.k_null = float(k_null)

        # home configuration for nullspace control (in rev_pri joint order)
        if q_home_rev_pri is not None: #
            self.q_home_rev_pri = np.asarray(q_home_rev_pri,dtype=float).copy()
            if len(self.q_home_rev_pri)!=len(self.rev_pri_joint_names):
                raise ValueError(
                    "[%s] len(q_home_rev_pri)=[%d] but len(rev_pri_joint_names)=[%d]"%
                    (self.name,len(self.q_home_rev_pri),len(self.rev_pri_joint_names))
                )
        else:
            if not hasattr(self,"q_home_rev_pri"):
                self.q_home_rev_pri = None

    # -----------------------------
    # buffers / targets
    # -----------------------------
    def reset_buffers(self,env=None):
        """
        Reset the IK buffers and optionally sync with the given environment.

        Parameters:
            env: The environment to sync from. Must have compatible state structure.
        """
        self.body_idx_list  = []
        self.p_local_list   = []
        self.p_curr_list    = []
        self.p_trgt_list    = []
        self.p_err_list     = []
        self.J_p_list       = []
        self.weight_list    = []
        self.n_target       = 0

        if env is not None:
            self.sync_env(env)

    def add_ik_target(
            self,
            body_idx  = None,
            body_name = None,
            p_curr    = None,
            p_trgt    = None,
            weight    = 1.0,
        ):
        """
        Add an inverse kinematics (IK) target.

        Parameters:
            body_idx: The index of the body to target.
            body_name: The name of the body to target.
            p_curr: The current position of the target in world coordinates.
            p_trgt: The desired target position in world coordinates.
            weight: The weight of the target in the IK solution.
        """
        if body_idx is None and body_name is None:
            raise ValueError("[%s] both body_idx and body_name are None."%(self.name))

        if weight is None:
            weight = 1.0
        weight = float(weight)
        if weight<=0.0 or (not np.isfinite(weight)):
            raise ValueError("weight must be positive and finite (got [%g])."%weight)

        p_curr = np.asarray(p_curr,dtype=float).reshape(3,)
        p_trgt = np.asarray(p_trgt,dtype=float).reshape(3,)

        if body_idx is None and body_name is not None:
            body_idx = self.ik_env.model.body(body_name).id
        body_id  = int(body_idx)
        p_local  = self._p_local_from_world(self.ik_env.data,body_id,p_curr)

        self.body_idx_list.append(body_id)
        self.p_local_list.append(p_local)
        self.p_trgt_list.append(p_trgt)

        p_curr_now = self._p_world_from_local(self.ik_env.data,body_id,p_local)
        perr       = p_trgt-p_curr_now
        J_p        = self._get_jacp_point_full(self.ik_env.model,self.ik_env.data,body_id,p_curr_now)

        self.p_curr_list.append(p_curr_now)
        self.p_err_list.append(perr)
        self.J_p_list.append(J_p)
        self.weight_list.append(weight)

        self.n_target = len(self.body_idx_list)

    def add_ik_coordinate_target(self,body_name,T_curr,T_trgt,weight=1.0,aw=0.1):
        """
        Add IK targets for the x, y, and z coordinates of a point on a body.

        Parameters:
            body_name (str): The name of the body to target.
            T_curr (np.ndarray): The current transformation matrix of the target point (4x4).
            T_trgt (np.ndarray): The desired transformation matrix of the target point (4x4).
            weight (float): The weight of the target in the IK solution.
            aw (float): The axis width for converting coordinate targets to point targets. This is used to determine how far apart the coordinate targets are placed along each axis.
        """
        self.add_ik_target(
            body_name = body_name,
            p_curr    = t2x(T_curr,aw),
            p_trgt    = t2x(T_trgt,aw),
            weight    = weight,
        )
        self.add_ik_target(
            body_name = body_name,
            p_curr    = t2y(T_curr,aw),
            p_trgt    = t2y(T_trgt,aw),
            weight    = weight,
        )
        self.add_ik_target(
            body_name = body_name,
            p_curr    = t2z(T_curr,aw),
            p_trgt    = t2z(T_trgt,aw),
            weight    = weight,
        )

    def update_ik_state(self):
        """
        Update the inverse kinematics (IK) state for all targets.
        """
        for i in range(self.n_target):
            body_id             = int(self.body_idx_list[i])
            p_local             = self.p_local_list[i]
            p_curr              = self._p_world_from_local(self.ik_env.data,body_id,p_local)
            self.p_curr_list[i] = p_curr
            p_trgt              = self.p_trgt_list[i]
            self.p_err_list[i]  = p_trgt-p_curr
            self.J_p_list[i]    = self._get_jacp_point_full(self.ik_env.model,self.ik_env.data,body_id,p_curr)

    # -----------------------------
    # dq computations (NO state change)
    # -----------------------------
    def compute_dq_base(
            self,
            J_p_array,
            p_err_array,
            base_dof_idxs,
            lam,
            base_stepsize,
            base_pos_th,
            base_rot_th,
        ):
        """
        Compute the delta for the base (free joint) DOFs using DLS.

        Parameters:
            J_p_array: The positional Jacobian array for all targets.
            p_err_array: The positional error array for all targets.
            base_dof_idxs: The indices of the base DOFs in the qpos.
            lam: The scalar damping factor for DLS computation.
            base_stepsize: The step size to apply to the computed delta.
            base_pos_th: The position threshold for trimming the delta.
            base_rot_th: The rotation threshold for trimming the delta.
        """
        J_base = J_p_array[:,base_dof_idxs]
        dq6    = _solve_dls(J_base,p_err_array,lam)
        dq6    = float(base_stepsize)*dq6
        if not np.isfinite(dq6).all():
            dq6 = np.zeros_like(dq6)

        dp  = trim_scale(x=dq6[:3],th=base_pos_th)
        dr  = trim_scale(x=dq6[3:],th=base_rot_th)
        dq6 = np.hstack([dp,dr])
        if not np.isfinite(dq6).all():
            dq6 = np.zeros_like(dq6)
        return dq6

    def compute_dq_rev_pri(
            self,
            J_p_array,
            p_err_array,
            joint_limit_handle_flag,
            nullspace_control_flag,
        ):
        """
        Compute the delta for the revolute and prismatic joints using DLS,
        with coupling, joint limit handling, and nullspace control.

        Damping behavior:
        - scalar dls_damping:
            same damping for all used joints
        - vector dls_damping:
            per-joint damping for joints_use only
            len(dls_damping) must equal len(joints_use)

        Parameters:
            J_p_array: The positional Jacobian array for all targets.
            p_err_array: The positional error array for all targets.
            joint_limit_handle_flag: Whether to perform joint limit handling.
            nullspace_control_flag: Whether to perform nullspace control towards the home configuration.
        """
        # ------------------------------------------------------------
        # 1. Extract joint Jacobian (rev_pri DOFs)
        # ------------------------------------------------------------
        J_rev_pri = J_p_array[:,self.idxs_jac_rev_pri]

        # ------------------------------------------------------------
        # 2. Coupling handling
        # ------------------------------------------------------------
        J_rev_pri_coupled = J_rev_pri.copy()
        for eq_idx in range(self.ik_env.model.eq_data.shape[0]):
            if self.ik_env.model.eq_type[eq_idx]!=mujoco.mjtEq.mjEQ_JOINT:
                continue

            passive_joint = self.ik_env.model.joint(self.ik_env.model.eq_obj1id[eq_idx]).name
            active_joint  = self.ik_env.model.joint(self.ik_env.model.eq_obj2id[eq_idx]).name
            if (passive_joint not in self._rev_pri_set) or (active_joint not in self._rev_pri_set):
                continue

            coef        = self.ik_env.model.eq_data[eq_idx,:5]
            x0          = self.ik_env.model.joint(active_joint).qpos0[0]
            x           = self.ik_env.data.joint(active_joint).qpos[0]
            idx_passive = self.rev_pri_joint_names.index(passive_joint)
            idx_active  = self.rev_pri_joint_names.index(active_joint)
            dfdx        = self.ik_env.eval_quartic_derivative(x,coef,x0)

            J_rev_pri_coupled[:,idx_active] += J_rev_pri_coupled[:,idx_passive]*dfdx
            J_rev_pri_coupled[:,idx_passive] = 0.0

        # ------------------------------------------------------------
        # 3. Reduced solve on joints_use only
        # ------------------------------------------------------------
        idxs_use = np.asarray(self.idxs_rev_pri_use,dtype=int)
        J_use    = J_rev_pri_coupled[:,idxs_use]
        lam_use  = self._get_dls_damping_use()

        dq_use = _solve_dls(J_use,p_err_array,lam_use)

        # ------------------------------------------------------------
        # 4. Type-wise step size
        # ------------------------------------------------------------
        is_rev_use = self._is_rev_in_rev_pri[idxs_use]
        steps_use  = np.where(
            is_rev_use,
            float(self.ik_stepsize_rev),
            float(self.ik_stepsize_pri),
        )
        dq_use = dq_use*steps_use
        if not np.isfinite(dq_use).all():
            dq_use = np.zeros_like(dq_use)

        q_rev_pri_curr = self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names)

        violated_idxs = np.array([],dtype=int)
        valid_local   = np.arange(len(idxs_use),dtype=int)

        # ------------------------------------------------------------
        # 5. Optional joint-limit handling
        # ------------------------------------------------------------
        if joint_limit_handle_flag:
            q_use = q_rev_pri_curr[idxs_use]

            is_rev_use = self._is_rev_in_rev_pri[idxs_use]
            probe_use  = np.where(
                is_rev_use,
                float(self.max_probe_rev),
                float(self.max_probe_pri),
            )

            q_use_next_probe = q_use+probe_use*np.sign(dq_use)

            q_mins_use = self.rev_pri_joint_mins[idxs_use]
            q_maxs_use = self.rev_pri_joint_maxs[idxs_use]

            hit_max       = (q_use_next_probe>q_maxs_use)&(dq_use>0)
            hit_min       = (q_use_next_probe<q_mins_use)&(dq_use<0)
            violated_idxs = np.where(hit_max|hit_min)[0]

            if len(violated_idxs)>0:
                valid_mask  = np.ones(len(idxs_use),dtype=bool)
                valid_mask[violated_idxs] = False
                valid_local = np.where(valid_mask)[0]

                if len(valid_local)>0:
                    J_use2 = J_use[:,valid_local]

                    if np.isscalar(lam_use):
                        lam_use2 = float(lam_use)
                    else:
                        lam_use2 = np.asarray(lam_use,dtype=np.float64).reshape(-1)[valid_local]

                    dq_use2 = _solve_dls(J_use2,p_err_array,lam_use2)
                    dq_use2 = dq_use2*steps_use[valid_local]

                    dq_use = np.zeros_like(dq_use)
                    dq_use[valid_local] = dq_use2
                else:
                    dq_use = np.zeros_like(dq_use)

        if not np.isfinite(dq_use).all():
            dq_use = np.zeros_like(dq_use)

        # ------------------------------------------------------------
        # 6. Optional nullspace control
        # ------------------------------------------------------------
        if nullspace_control_flag and (self.q_home_rev_pri is not None):
            if len(valid_local)>0:
                idxs_use_valid = idxs_use[valid_local]
                J_use_valid    = J_use[:,valid_local]

                q_err_use = self.q_home_rev_pri[idxs_use_valid]-q_rev_pri_curr[idxs_use_valid]
                dq_null   = self.k_null*q_err_use

                if np.isscalar(lam_use):
                    lam_valid = float(lam_use)
                else:
                    lam_valid = np.asarray(lam_use,dtype=np.float64).reshape(-1)[valid_local]

                dq_add = _project_dq_null_damped(
                    J_use   = J_use_valid,
                    dq_null = dq_null,
                    lam     = lam_valid,
                )

                dq_add = dq_add*steps_use[valid_local]
                dq_use[valid_local] += dq_add

        if not np.isfinite(dq_use).all():
            dq_use = np.zeros_like(dq_use)

        # ------------------------------------------------------------
        # 7. Scatter reduced result back to full rev_pri vector
        # ------------------------------------------------------------
        dq_rev_pri = np.zeros((len(self.rev_pri_joint_names),),dtype=np.float64)
        dq_rev_pri[idxs_use] = dq_use

        if not np.isfinite(dq_rev_pri).all():
            dq_rev_pri = np.zeros_like(dq_rev_pri)

        return dq_rev_pri

    # -----------------------------
    # applying updates (state change)
    # -----------------------------
    def apply_base_delta(self,model,data,base_dof_idxs,dq6):
        """
        Apply the delta to the base degrees of freedom.

        Parameters:
            model: The Mujoco model.
            data: The Mujoco data.
            base_dof_idxs: The indices of the base degrees of freedom.
            dq6: The delta to apply to the base degrees of freedom.
        """
        dq_full = np.zeros(model.nv,dtype=np.float64)
        dq_full[base_dof_idxs] = dq6
        mujoco.mj_integratePos(model,data.qpos,dq_full,1.0)
        mujoco.mj_forward(model,data)

    def _dq_rev_pri_to_q_rev_pri_next(self,dq_rev_pri):
        """
        Convert the delta for the revolute and prismatic joints to the next joint positions.

        Parameters:
            dq_rev_pri: The delta for the revolute and prismatic joints.

        Returns:
            q_rev_pri_next: The next joint positions for the revolute and prismatic joints.
        """
        q_rev_pri_curr = self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names)

        dq_active = dq_rev_pri[self.idxs_rev_pri_active]
        is_rev_active = self._is_rev_in_rev_pri[self.idxs_rev_pri_active]

        if np.any(is_rev_active):
            dq_active_rev            = dq_active[is_rev_active]
            dq_active_rev            = trim_scale(x=dq_active_rev,th=self.ik_update_th_rev)
            dq_active[is_rev_active] = dq_active_rev

        if np.any(~is_rev_active):
            dq_active_pri             = dq_active[~is_rev_active]
            dq_active_pri             = trim_scale(x=dq_active_pri,th=self.ik_update_th_pri)
            dq_active[~is_rev_active] = dq_active_pri

        q_active = self.ik_env.get_qpos(joint_names=self.active_joint_names_rev_pri)+dq_active

        q_rev_pri_next = q_rev_pri_curr.copy()
        q_rev_pri_next[self.idxs_rev_pri_active] = q_active

        for eq_idx in range(self.ik_env.model.eq_data.shape[0]):
            if self.ik_env.model.eq_type[eq_idx]!=mujoco.mjtEq.mjEQ_JOINT:
                continue

            passive_joint = self.ik_env.model.joint(self.ik_env.model.eq_obj1id[eq_idx]).name
            active_joint  = self.ik_env.model.joint(self.ik_env.model.eq_obj2id[eq_idx]).name
            if (passive_joint not in self._rev_pri_set) or (active_joint not in self._rev_pri_set):
                continue

            coef = self.ik_env.model.eq_data[eq_idx,:5]
            x0   = self.ik_env.model.joint(active_joint).qpos0[0]
            x    = q_rev_pri_next[self.rev_pri_joint_names.index(active_joint)]
            y    = self.ik_env.eval_quartic(x=x,coef=coef,x0=x0)
            q_rev_pri_next[self.rev_pri_joint_names.index(passive_joint)] = y

        q_rev_pri_next = np.clip(q_rev_pri_next,self.rev_pri_joint_mins,self.rev_pri_joint_maxs)
        return q_rev_pri_next

    def apply_rev_pri_qpos(self,model,data,rev_pri_qpos_idxs,q_rev_pri_next):
        """
        Apply the next joint positions for the revolute and prismatic joints.

        Parameters:
            model: The Mujoco model.
            data: The Mujoco data.
            rev_pri_qpos_idxs: The indices of the qpos corresponding to the revolute and prismatic joints.
            q_rev_pri_next: The next joint positions for the revolute and prismatic joints.
        """
        data.qpos[rev_pri_qpos_idxs] = q_rev_pri_next
        mujoco.mj_forward(model,data)

    # -----------------------------
    # public entry
    # -----------------------------
    def sync_env(self,env):
        """
        Sync the IK environment state with the given environment.

        Parameters:
            env: The environment to sync from. Must have compatible state structure.
        """
        self.ik_env.data.qpos[:] = env.data.qpos.copy()
        for body_name in env.body_names:
            self.ik_env.model.body(body_name).pos  = env.model.body(body_name).pos.copy()
            self.ik_env.model.body(body_name).quat = env.model.body(body_name).quat.copy()
        mujoco.mj_forward(self.ik_env.model,self.ik_env.data)

    def solve_ik(
            self,
            env                     = None,
            joints_use              = None,
            joint_limit_handle_flag = True,
            nullspace_control_flag  = True,
            init_from_median_flag   = False,
            base_control_flag       = False,
            base_stepsize           = 0.5,
            base_pos_th             = 0.1,
            base_rot_th             = np.deg2rad(5.0),
        ):
        """
        Solve the inverse kinematics for the given environment.

        Parameters:
            env: The environment to sync from. Must have compatible state structure.
            joints_use: The joints to use for IK.
            joint_limit_handle_flag: Whether to handle joint limits.
            nullspace_control_flag: Whether to use nullspace control.
            init_from_median_flag: Whether to initialize from the median joint positions.
            base_control_flag: Whether to control the base.
            base_stepsize: The step size for the base control.
            base_pos_th: The position threshold for the base control.
            base_rot_th: The rotation threshold for the base control.

        Returns:
            q_rev_pri_curr: The current joint positions for the revolute and prismatic joints.
            info: A dictionary containing IK information.
        """
        if env is not None:
            self.sync_env(env)

        if self.n_target==0:
            q_rev_pri_curr = self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names)
            info = {
                'ik_err_array':np.array([0.0]),
                'idx_best':0,
                'ik_err_best':0.0,
                'elapsed_time':0.0,
                'qpos_full_best':self.ik_env.data.qpos.copy(),
            }
            return q_rev_pri_curr,info

        if joints_use is not None:
            self.joints_use = self._normalize_joints_use(joints_use)
        self.idxs_rev_pri_use = get_idxs(self.rev_pri_joint_names,self.joints_use)

        if init_from_median_flag:
            q_rev_pri_median = 0.5*(self.rev_pri_joint_mins+self.rev_pri_joint_maxs)
            q_curr = self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names)
            bad    = ~np.isfinite(q_rev_pri_median)
            if np.any(bad):
                q_rev_pri_median[bad] = q_curr[bad]
            self.ik_env.forward(q=q_rev_pri_median,joint_names=self.rev_pri_joint_names)

        model = self.ik_env.model
        data  = self.ik_env.data

        rev_pri_qpos_idxs = self._get_rev_pri_qpos_idxs(model)

        base_dof_idxs = None
        if base_control_flag:
            base_dof_idxs = self._get_base_dof_idxs(model)

        self.update_ik_state()

        q_rev_pri_list = []
        qpos_list      = []
        ik_err_array   = np.zeros(self.max_ik_tick+1)

        q_rev_pri_list.append(self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names))
        qpos_list.append(self.ik_env.data.qpos.copy())
        ik_err_array[0] = float(np.linalg.norm(np.hstack(self.p_err_list)))

        # Scalar fallback damping for base solve
        lam_base = self._get_dls_damping_base()

        self.tt.tic()
        for ik_tick in range(self.max_ik_tick):
            J_p_array,p_err_array = self._build_weighted_Je_full(model,data)

            if base_control_flag and (base_dof_idxs is not None):
                dq6 = self.compute_dq_base(
                    J_p_array,p_err_array,
                    base_dof_idxs,
                    lam_base,
                    base_stepsize,base_pos_th,base_rot_th
                )
                self.apply_base_delta(model,data,base_dof_idxs,dq6)
                self.update_ik_state()
                J_p_array,p_err_array = self._build_weighted_Je_full(model,data)

            dq_rev_pri = self.compute_dq_rev_pri(
                J_p_array,
                p_err_array,
                joint_limit_handle_flag,
                nullspace_control_flag,
            )
            q_rev_pri_next = self._dq_rev_pri_to_q_rev_pri_next(dq_rev_pri)
            self.apply_rev_pri_qpos(model,data,rev_pri_qpos_idxs,q_rev_pri_next)
            self.update_ik_state()

            q_rev_pri_list.append(self.ik_env.get_qpos(joint_names=self.rev_pri_joint_names))
            qpos_list.append(self.ik_env.data.qpos.copy())
            ik_err_array[ik_tick+1] = float(np.linalg.norm(np.hstack(self.p_err_list)))

        idx_best       = int(np.argmin(ik_err_array))
        q_rev_pri_best = q_rev_pri_list[idx_best]
        qpos_full_best = qpos_list[idx_best].copy()
        ik_err_best    = float(ik_err_array[idx_best])
        elapsed_time   = float(self.tt.toc())

        idxs_use       = np.asarray(self.idxs_rev_pri_use,dtype=int)
        qpos_used_best = qpos_full_best[rev_pri_qpos_idxs[idxs_use]].copy()

        info = {
            'ik_err_array':ik_err_array,
            'idx_best':idx_best,
            'ik_err_best':ik_err_best,
            'elapsed_time':elapsed_time,
            'qpos_full_best':qpos_full_best,
            'qpos_used_best':qpos_used_best,
            "joints_use":self.joints_use,
        }
        return q_rev_pri_best,info

    # -----------------------------
    # render
    # -----------------------------
    def render(
            self,
            env,
            r                      = 0.02,
            alpha                  = 1.0,
            cmap_name              = 'rainbow',
            show_weights           = False,
            scale_radius_by_weight = False,
            plot_arrows            = False,
            rgba                   = None,
        ):
        """
        Render the IK targets and current positions.

        Parameters:
            env: The environment to render in.
            r: The radius of the spheres.
            alpha: The alpha value for the colors.
            cmap_name: The name of the colormap.
            show_weights: Whether to show the weights.
            scale_radius_by_weight: Whether to scale the radius by the weight.
            plot_arrows: Whether to plot arrows from current to target positions.
            rgba: The RGBA color to use for rendering. If None, colors will be generated from the colormap.
        """
        n_trgt = int(self.n_target)
        if n_trgt<=0:
            return
        colors = get_colors(cmap_name=cmap_name,n_color=n_trgt,alpha=alpha)

        for i in range(n_trgt):
            body_id = int(self.body_idx_list[i])
            p_local = self.p_local_list[i]
            p_curr  = self._p_world_from_local(env.data,body_id,p_local)
            p_trgt  = self.p_trgt_list[i]
            rgba_i  = colors[i] if rgba is None else rgba
            w       = float(self.weight_list[i])

            if scale_radius_by_weight:
                r_use = max(r*w,0.005)
            else:
                r_use = r

            label = ""
            if show_weights:
                label = "%.2f"%(w)

            env.plot_sphere(p_curr,r_use,rgba_i)
            env.plot_sphere(p_trgt,r_use,rgba_i,label=label)
            if plot_arrows:
                env.plot_arrow_fr2to(p_fr=p_curr,p_to=p_trgt,rgba=rgba_i,r=r_use*0.5)
            else:
                env.plot_line_fr2to(p_fr=p_curr,p_to=p_trgt,rgba=rgba_i)
