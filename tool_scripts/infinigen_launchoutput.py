# Scipt to show the output of the Infinigen scene in Isaac Sim

import argparse
import os
import sys

# Parse command line arguments before starting SimulationApp
parser = argparse.ArgumentParser(description='Setup Infinigen environment in Isaac Sim')
parser.add_argument('usd_path', 
                    help='Path to USD/USDC file relative to /home/kaliber/infinigen/outputs/ (e.g., omniverse/APT2_fast_door_open_room_fast_1sofa/export_scene.blend/export_scene.usdc)')
parser.add_argument('--base-path', '-b',
                    default='/home/kaliber/infinigen/outputs',
                    help='Base directory path (default: /home/kaliber/infinigen/outputs)')
parser.add_argument('--set-default-prim', 
                    action='store_true',
                    default=False,
                    help='Set /Environment as the default prim (default: False)')
parser.add_argument('--add-rigid-body', 
                    action='store_true',
                    default=False,
                    help='Add rigid body dynamics in addition to colliders (default: False, colliders only)')
parser.add_argument('--approximation', '-a', 
                    choices=['none', 'convexHull', 'boundingCube', 'boundingSphere'],
                    default='none',
                    help='Collision approximation type (default: none)')
parser.add_argument('--hide-walls', 
                    action='store_true',
                    default=True,
                    help='Hide ceiling and exterior walls (default: True)')

args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp(launch_config={"headless": False})

import omni.usd
import infinigen_sdg_utils as infinigen_utils

def verify_positions_unchanged(stage, before_transforms: dict, after_transforms: dict):
    """
    Verify that object positions haven't changed
    
    Args:
        stage: USD stage
        before_transforms: Dictionary of prim paths to transform matrices before setup
        after_transforms: Dictionary of prim paths to transform matrices after setup
    """
    changes_detected = 0
    total_checked = 0
    
    for prim_path, before_transform in before_transforms.items():
        if prim_path in after_transforms:
            after_transform = after_transforms[prim_path]
            total_checked += 1
            
            # Compare transform matrices (with small tolerance for floating point precision)
            if not transforms_are_equal(before_transform, after_transform, tolerance=1e-6):
                print(f"⚠️ Position changed for: {prim_path}")
                changes_detected += 1
    
    if changes_detected == 0:
        print(f" Position verification: All {total_checked} object positions preserved")
    else:
        print(f" Position verification: {changes_detected}/{total_checked} objects moved")
    
    return changes_detected == 0

def transforms_are_equal(transform1, transform2, tolerance=1e-6):
    """Check if two transform matrices are equal within tolerance"""
    try:
        import numpy as np
        # Convert to numpy arrays for comparison
        if transform1 is None or transform2 is None:
            return transform1 == transform2
        
        # Simple comparison for basic cases
        return abs(transform1 - transform2) < tolerance if hasattr(transform1, '__sub__') else transform1 == transform2
    except:
        # Fallback to direct comparison
        return transform1 == transform2

def capture_transforms(stage, root_path="/Environment"):
    """
    Capture all transform data under the specified root path
    
    Args:
        stage: USD stage
        root_path: Root path to capture transforms from
        
    Returns:
        Dictionary mapping prim paths to their transform data
    """
    from pxr import Usd, UsdGeom
    
    transforms = {}
    root_prim = stage.GetPrimAtPath(root_path)
    
    if not root_prim.IsValid():
        return transforms
    
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAttribute("xformOp:translate"):
            transform_data = {
                'translate': prim.GetAttribute("xformOp:translate").Get(),
                'rotation': prim.GetAttribute("xformOp:rotateXYZ").Get() if prim.HasAttribute("xformOp:rotateXYZ") else None,
                'scale': prim.GetAttribute("xformOp:scale").Get() if prim.HasAttribute("xformOp:scale") else None,
                'orient': prim.GetAttribute("xformOp:orient").Get() if prim.HasAttribute("xformOp:orient") else None
            }
            transforms[str(prim.GetPath())] = transform_data
    
    return transforms

def setup_infinigen_scene(usd_file_path: str, approximation_type: str = "none", hide_top_walls: bool = True, add_rigid_body: bool = False, set_default_prim: bool = False):
    """
    Load Infinigen USD file and setup environment
    
    Args:
        usd_file_path: Full path to the USD file
        approximation_type: Collision approximation type
        hide_top_walls: Whether to hide ceiling and exterior walls
        add_rigid_body: Whether to add rigid body dynamics in addition to colliders
        set_default_prim: Whether to set /Environment as the default prim
    """
    print(f"=== Starting Infinigen Environment Setup ===")
    print(f"File: {usd_file_path}")
    print(f"Approximation: {approximation_type}")
    print(f"Hide walls: {hide_top_walls}")
    print(f"Add rigid body: {add_rigid_body}")
    print(f"Set default prim: {set_default_prim}")
    
    # 1. Create new stage
    print("Creating new stage...")
    omni.usd.get_context().new_stage()
    
    # 2. Load environment
    print(f"Loading environment: {usd_file_path}")
    try:
        env_prim = infinigen_utils.load_env(
            usd_path=usd_file_path, 
            prim_path="/Environment", 
            remove_existing=True
        )
        print(f"✓ Environment successfully loaded to: /Environment")
    except Exception as e:
        print(f"✗ Loading failed: {e}")
        return False
    
    # 3. Set default prim if requested
    if set_default_prim:
        print("Setting /Environment as default prim...")
        setup_default_prim("/Environment")
    
    # 4. Update app to ensure loading is complete
    simulation_app.update()
    
    # 5. Capture initial transforms for verification
    stage = omni.usd.get_context().get_stage()
    print("Capturing initial object positions...")
    initial_transforms = capture_transforms(stage, "/Environment")
    print(f"✓ Captured {len(initial_transforms)} object transforms")
    
    # 6. Setup environment with custom collider/rigid body options
    print("Setting up environment...")
    setup_environment_with_options(
        root_path="/Environment",
        approximation_type=approximation_type,
        hide_top_walls=hide_top_walls,
        add_rigid_body=add_rigid_body
    )
    
    # 7. Verify positions haven't changed
    print("Verifying object positions...")
    final_transforms = capture_transforms(stage, "/Environment")
    position_preserved = verify_positions_unchanged(stage, initial_transforms, final_transforms)
    
    collision_type = "colliders and rigid bodies" if add_rigid_body else "colliders only"
    print(f"✓ Environment setup complete ({collision_type} added, ceiling hidden)")
    
    # 8. Resolve scaling issues
    print("Resolving scaling issues...")
    infinigen_utils.resolve_scale_issues_with_metrics_assembler()
    print("✓ Scaling issues resolved")
    
    # 9. Setup complete
    print("✓ All setup completed")
    
    print(f"=== Environment Setup Complete! ===")
    return True

def setup_default_prim(prim_path: str = "/Environment"):
    """
    Set the specified prim as the default prim for the stage
    
    Args:
        prim_path: Path to the prim to set as default (default: "/Environment")
    """
    stage = omni.usd.get_context().get_stage()
    
    # Get the prim at the specified path
    prim = stage.GetPrimAtPath(prim_path)
    
    if not prim.IsValid():
        print(f"⚠️ Warning: Prim at path '{prim_path}' is not valid. Cannot set as default prim.")
        return False
    
    # Set the default prim
    stage.SetDefaultPrim(prim)
    
    print(f"✓ Default prim set to: {prim_path}")
    return True

def setup_environment_with_options(root_path: str = None, approximation_type: str = "none", hide_top_walls: bool = False, add_rigid_body: bool = False):
    """
    Custom setup environment function with options for colliders only or colliders + rigid body
    
    Args:
        root_path: Root path for the environment
        approximation_type: Collision approximation type
        hide_top_walls: Whether to hide ceiling and exterior walls
        add_rigid_body: Whether to add rigid body dynamics in addition to colliders
    """
    # Fix ceiling lights: meshes are blocking the light and need to be set to invisible
    print("Fixing ceiling lights...")
    ceiling_light_meshes = infinigen_utils.find_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")
    for light_mesh in ceiling_light_meshes:
        light_mesh.GetAttribute("visibility").Set("invisible")

    # Hide ceiling light meshes for lighting fix
    infinigen_utils.hide_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")

    # Hide top walls for better debug view, if specified
    if hide_top_walls:
        print("Hiding ceiling and exterior walls...")
        infinigen_utils.hide_matching_prims(["_exterior", "_ceiling"], root_path)

    # Add colliders or colliders + rigid body dynamics
    if add_rigid_body:
        print(f"Adding colliders and rigid body dynamics with approximation type: {approximation_type}")
        add_colliders_and_rigid_body_to_env(root_path, approximation_type)
    else:
        print(f"Adding colliders only with approximation type: {approximation_type}")
        infinigen_utils.add_colliders_to_env(root_path, approximation_type)

    # Fix dining table collision by setting it to a bounding cube approximation
    table_prim = infinigen_utils.find_matching_prims(
        match_strings=["TableDining"], root_path=root_path, prim_type="Xform", first_match_only=True
    )
    if table_prim is not None:
        print("Adding special collision handling for dining table...")
        if add_rigid_body:
            infinigen_utils.add_colliders_and_rigid_body_dynamics(table_prim, disable_gravity=False)
        else:
            infinigen_utils.add_colliders(table_prim, approximation_type="boundingCube")
    else:
        print("No dining table found in the environment")

def add_colliders_and_rigid_body_to_env(root_path: str = None, approximation_type: str = "none"):
    """Add colliders and rigid body dynamics to all mesh prims using infinigen_utils functions"""
    import omni.usd
    from pxr import Usd, UsdGeom
    
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)
    
    processed_count = 0
    if root_prim:
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                # Use infinigen_utils function to add both colliders and rigid body
                infinigen_utils.add_colliders_and_rigid_body_dynamics(prim, disable_gravity=False)
                processed_count += 1
    
    print(f"Added colliders and rigid body dynamics to {processed_count} mesh prims")

def construct_full_path(relative_path: str, base_path: str = "/home/kaliber/infinigen/outputs") -> str:
    """
    Construct full file path from relative path
    
    Args:
        relative_path: Path relative to base_path (e.g., omniverse/APT2_fast_door_open_room_fast_1sofa/export_scene.blend/export_scene.usdc)
        base_path: Base directory path
    
    Returns:
        Full file path
    """
    # Ensure relative_path is not None or empty
    if not relative_path:
        raise ValueError("relative_path cannot be empty or None")
    
    # Remove leading slash if present
    if relative_path.startswith('/'):
        relative_path = relative_path[1:]
    
    # Construct full path
    full_path = os.path.join(base_path, relative_path)
    
    return full_path

# Main program
if __name__ == "__main__":
    try:
        # Construct full file path
        usd_file_path = construct_full_path(args.usd_path, args.base_path)
        
        print(f"🔗 Using file path: {usd_file_path}")
        
        # Check if file exists
        if not os.path.exists(usd_file_path):
            print(f"File not found: {usd_file_path}")
            print("Please check the path and try again.")
            sys.exit(1)
        
        # Setup environment
        success = setup_infinigen_scene(
            usd_file_path=usd_file_path,
            approximation_type=args.approximation,
            hide_top_walls=args.hide_walls,
            add_rigid_body=args.add_rigid_body,
            set_default_prim=args.set_default_prim
        )
        
        if success:
            collision_info = "colliders and rigid bodies" if args.add_rigid_body else "colliders only"
            default_prim_info = " with /Environment as default prim" if args.set_default_prim else ""
            print(f"\n🎉 Success! You can view the setup environment in Isaac Sim")
            print(f"   - Ceiling has been hidden (if enabled)")
            print(f"   - All objects have {collision_info}")
            print(f"   - Ceiling lights have been fixed")
            if args.set_default_prim:
                print(f"   - Default prim set to /Environment")
                print(f"   - All object positions preserved")
    
    except Exception as e:
        print(f" Error occurred: {e}")
        print("Please check your arguments and try again.")
        sys.exit(1)
    
    # Keep application running
    print("\n Close window to exit program...")
    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()
