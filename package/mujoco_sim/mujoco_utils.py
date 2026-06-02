"""
mujoco_utils.py
"""
import os
import numpy as np
import xml.etree.ElementTree as ET

# Import utility functions
# """
# MuJoCo XML utility helpers.
# """
# from utils import (
#     indent_xml,
# )
from package.utility.utils import (
    indent_xml
)
from package.kinematics.transforms import (
    pr2t
)

def normalize_mujoco_joint_types(joint_types):
    """
    Normalize user joint-type input into MuJoCo joint type integers.
    """
    if joint_types is None:
        return None

    if not isinstance(joint_types,(list,tuple,set,np.ndarray)):
        joint_types = [joint_types]

    joint_type_set = set()

    for jt in joint_types:
        if isinstance(jt,str):
            jt_lower = jt.lower()

            if jt_lower in ["rev","revolute","hinge"]:
                joint_type_set.add(int(mujoco.mjtJoint.mjJNT_HINGE))
            elif jt_lower in ["pri","prismatic","slide"]:
                joint_type_set.add(int(mujoco.mjtJoint.mjJNT_SLIDE))
            elif jt_lower in ["ball"]:
                joint_type_set.add(int(mujoco.mjtJoint.mjJNT_BALL))
            elif jt_lower in ["free"]:
                joint_type_set.add(int(mujoco.mjtJoint.mjJNT_FREE))
            else:
                raise ValueError("Unsupported joint type string:[%s]"%(jt))
        else:
            joint_type_set.add(int(jt))

    return joint_type_set

def merge_mjcfs(
        mjcf_model_name     = 'scene', 
        included_mjcf_files = ['a.xml', 'b.xml'], 
        timestep            = 0.002,
        integrator          = 'implicitfast',
        noslip              = None,  # e.g., 1, 5, 10
        memory              = None,  # "500M", "1G", etc.
        armature            = None,  # e.g., 1.0
        damping             = None,  # e.g., 2.0
        cone                = None,  # e.g., 'elliptic'
        exclude_body_list   = None,   # NEW ARG: [['palm','ff_base'], ...]
        output_xml_path     = 'merged_scene.xml',
        verbose             = True,
    ):
    """
    Merges multiple MJCF files into a single MuJoCo model XML, adds optional simulation settings,
    pretty-prints it, saves it to a file, and returns the file path.

    Parameters:
    ------------
        mjcf_model_name (str): Name of the MuJoCo model.
        included_mjcf_files (list): List of MJCF file paths to include.
        timestep (float): Simulation timestep.
        integrator (str): Integrator type (e.g., 'implicitfast', 'RK4').
        noslip (int or None): MuJoCo noslip iteration count. If None, do not set it.
        memory (str or None): Memory allocation for the simulation (e.g., "500M", "1G").
        armature (float or None): Default armature value for joints.
        damping (float or None): Default damping value for joints.
        cone (str or None): Cone type for contact handling (e.g., 'elliptic').
        exclude_body_list (list or None): List of body pairs to exclude from collision detection.
        output_xml_path (str): Path to save the merged XML file.
        verbose (bool): If True, prints progress information.

    Returns:
    -------- 
        str: The path to the saved XML file.
    """
    # Create root element
    mujoco_elem = ET.Element("mujoco", attrib={"model": mjcf_model_name})

    output_dir = os.path.dirname(os.path.abspath(output_xml_path))

    # Add include elements
    for filepath in included_mjcf_files:
        abs_path = os.path.abspath(filepath)
        rel_path = os.path.relpath(abs_path, output_dir)
        ET.SubElement(mujoco_elem, "include", attrib={"file": rel_path})

    # Add <default> tag if needed
    if armature is not None or damping is not None:
        default_elem = ET.SubElement(mujoco_elem, "default")
        joint_attrs = {}
        if armature is not None:
            joint_attrs["armature"] = str(armature)
        if damping is not None:
            joint_attrs["damping"] = str(damping)
        ET.SubElement(default_elem, "joint", attrib=joint_attrs)

    # Add <option> tag with extra attributes if given
    option_attrs = {
        "timestep": str(timestep),
        "integrator": integrator,
    }
    if noslip is not None:
        option_attrs["noslip_iterations"] = str(int(noslip))
    if cone is not None:
        option_attrs["cone"] = cone
    ET.SubElement(mujoco_elem, "option", attrib=option_attrs)

    # Add <size> tag if memory is given
    if memory is not None:
        mem_value = memory if isinstance(memory, str) else "1G"
        ET.SubElement(mujoco_elem, "size", attrib={"memory": mem_value})

    # Add <contact> tag with <exclude> children if exclude_body_list is given
    if exclude_body_list is not None and len(exclude_body_list) > 0:
        contact_elem = ET.SubElement(mujoco_elem, "contact")
        for pair in exclude_body_list:
            if len(pair) != 2:
                raise ValueError(f"Each exclude pair must have length 2, got: {pair}")
            body1, body2 = pair
            ET.SubElement(contact_elem, "exclude", attrib={
                "body1": str(body1),
                "body2": str(body2),
            })

    # Convert to pretty string
    pretty_xml = indent_xml(mujoco_elem)

    # Ensure the directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write to file
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    # Verbose output
    if verbose:
        print("[merge_mjcfs] Merging [%d] MJCF files:"%(len(included_mjcf_files)))
        for f_idx,f in enumerate(included_mjcf_files):
            print("  - [%d] [%s]"%(f_idx,f))
        print(f"[merge_mjcfs] Saved merged XML to:[%s]"%(output_xml_path))

    return output_xml_path

def get_label_colors_from_joint_axis(env,joints_names=None,bright_colors=True):
    """
    Get label colors based on the primary axis of the specified joints.

    Parameters:
        joints_names (list or None): List of joint names to consider. If None, uses all revolute joints.
        bright_colors (bool): If True, uses brighter colors for the labels.
    """
    # Default to all revolute joints if none specified
    if joints_names is None:
        joints_names = env.rev_joint_names

    if bright_colors:
        rgb_list = ['#FF6666', '#66FF66', '#66CCFF']
    else:
        rgb_list = ['#AA0000', '#00AA00', '#0066AA']

    # Determine label colors based on joint axes
    label_colors = []
    for joint_name in joints_names:
        joint_idx   = env.joint_names.index(joint_name)
        axis_joint  = env.model.jnt_axis[joint_idx]
        if np.allclose(axis_joint, 0):
            label_color = '#888888'  # Default gray color for zero axis
        else:
            label_color = rgb_list[int(np.argmax(np.abs(axis_joint)))]
        label_colors.append(label_color)

    # Return the list of label colors
    return label_colors

def get_T_hand_from_sites(
        env,
        site_name_common = None,
        hand_type        = "right",
        site_type        = "site",
        postfix_dict     = None,
    ):
    """
    Build hand coordinate frame from four sites:
        <common>
        <common>_top
        <common>_palmar
        <common>_front

    Parameters:
        env: MuJoCo environment / parser
        site_name_common (str): common site name such as
            "doosan_rpalm"
            "franka_lpalm"
        hand_type (str): "right" or "left"
        site_type (str): usually "site"
        postfix_dict (dict or None): optional postfix override
            default:
                {
                    "palm"   :"",
                    "top"    :"_top",
                    "palmar" :"_palmar",
                    "front"  :"_front",
                }

    Returns:
        T_palm (np.ndarray): shape (4,4)
    """
    def _normalize(v,eps=1e-12):
        v = np.asarray(v,dtype=np.float64).reshape(3,)
        n = np.linalg.norm(v)
        if n < eps:
            raise ValueError("Zero-length vector encountered.")
        return v/n

    if site_name_common is None:
        raise ValueError("site_name_common is required.")

    if postfix_dict is None:
        postfix_dict = {
            "palm"   :"",
            "top"    :"_top",
            "palmar" :"_palmar",
            "front"  :"_front",
        }

    hand_type = str(hand_type).lower()
    if hand_type in ["right","r","rh"]:
        is_right = True
    elif hand_type in ["left","l","lh"]:
        is_right = False
    else:
        raise ValueError("Unsupported hand_type:[%s]"%(hand_type))

    # Resolve site names
    site_name_palm   = site_name_common + postfix_dict["palm"]
    site_name_top    = site_name_common + postfix_dict["top"]
    site_name_palmar = site_name_common + postfix_dict["palmar"]
    site_name_front  = site_name_common + postfix_dict["front"]

    # Get points
    p_palm   = env.get_p(site_name_palm,site_type)
    p_top    = env.get_p(site_name_top,site_type)
    p_palmar = env.get_p(site_name_palmar,site_type)
    p_front  = env.get_p(site_name_front,site_type)

    # Axis hints
    ux      = _normalize(p_front-p_palm)          # +x: front
    uz_hint = _normalize(p_top-p_palm)            # +z: top

    if is_right:
        uy_hint = _normalize(p_palmar-p_palm)     # +y: palm -> palmar
    else:
        uy_hint = _normalize(p_palm-p_palmar)     # +y: reversed for left hand

    # Orthonormalize while preserving handedness
    uz = np.cross(ux,uy_hint)
    uz = _normalize(uz)

    # Keep uz aligned with top direction
    if np.dot(uz,uz_hint) < 0.0:
        uz = -uz

    uy = np.cross(uz,ux)
    uy = _normalize(uy)

    R_palm = np.column_stack((ux,uy,uz)).reshape(3,3)
    T_palm = pr2t(p=p_palm,R=R_palm)
    return T_palm
