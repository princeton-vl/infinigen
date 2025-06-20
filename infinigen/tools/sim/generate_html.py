"""
Generates an html with all the assets using all exporters for testing purposes
"""

import logging
from pathlib import Path

import mujoco
import numpy as np
import pybullet
from PIL import Image

from infinigen.assets import sim_objects
from infinigen.core.sim import sim_factory as sf

HEIGHT = 320
WIDTH = 480
DURATION = 8.0

additional_sim_objects = ["door", "toaster"]

exclude_sim_objects = [
    "mapping",
    "singlefridge",
    "doublefridge",
]


def get_sim_names(root_dir):
    root = Path(root_dir)
    sim_objects = [file.stem for file in root.iterdir() if file.is_file()]
    sim_objects.extend(additional_sim_objects)
    sim_objects = [item for item in sim_objects if item not in exclude_sim_objects]
    return sim_objects


def render_mjcf(file):
    try:
        spec = mujoco.MjSpec.from_file(str(file))
        spec.worldbody.add_light(pos=[0, 0, 5])

        spec.worldbody.add_camera(
            name="frontview", pos=[0, -5, 5], xyaxes=[1, 0, 0, 0, 0.707, 0.707]
        )

        spec.worldbody.add_camera(
            name="sideview", pos=[5, 0, 5], xyaxes=[0, 1, 0, -0.707, 0, 0.707]
        )

        m = spec.compile()

    except Exception as e:
        print(f"Issue compiling MJCF for {file}")
        print(e)
        return None, None

    d = mujoco.MjData(m)
    with mujoco.Renderer(m, width=WIDTH, height=HEIGHT) as renderer:
        mujoco.mj_forward(m, d)

        # getting the initial image
        scene_option = mujoco.MjvOption()
        scene_option.geomgroup = [0, 1, 0, 0, 0, 0]  # visual geoms only
        renderer.update_scene(d, camera="sideview", scene_option=scene_option)
        image = Image.fromarray(renderer.render())
        image_path = Path(__file__).parent / f"images/{file.stem}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)

        # steps through the simulation and generates a gif
        frames = []
        mujoco.mj_resetData(m, d)

        scene_option.geomgroup = [1, 0, 0, 0, 0, 0]  # collision geoms only
        while d.time < DURATION:
            mujoco.mj_step(m, d)
            if len(frames) < d.time * 24:
                renderer.update_scene(d, camera="sideview", scene_option=scene_option)
                pixels = renderer.render()
                frames.append(Image.fromarray(pixels))

        gif_path = Path(__file__).parent / f"gifs/mjcf/sideview/{file.stem}.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=DURATION,
            loop=0,  # Loop infinitely
        )

        return image_path, gif_path


def render_urdf(file):
    pybullet.connect(pybullet.DIRECT)

    try:
        objid = pybullet.loadURDF(str(file))
    except Exception as e:
        print(f"Issue compiling URDF for {file}")
        print(e)
        return None

    camTargetPos = [0, 0, 0]
    camEyePos = [5, 0, 5]
    cameraUp = [0, 0, 1]
    pybullet.setGravity(0, 0, -9.81)

    mode = pybullet.VELOCITY_CONTROL
    pybullet.setJointMotorControl2(objid, 0, controlMode=mode, targetVelocity=-10)

    nearPlane = 0.01
    farPlane = 100

    fov = 60

    frames = []

    delta_time = 1 / 24
    pybullet.setTimeStep(delta_time)
    pybullet.setRealTimeSimulation(0)

    while delta_time * len(frames) < DURATION:
        pybullet.stepSimulation()

        viewMatrix = pybullet.computeViewMatrix(camEyePos, camTargetPos, cameraUp)
        aspect = WIDTH / HEIGHT
        projectionMatrix = pybullet.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )
        img_arr = pybullet.getCameraImage(
            WIDTH,
            HEIGHT,
            viewMatrix,
            projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = img_arr[2]

        # reshape is needed
        pixels = np.reshape(rgb, (HEIGHT, WIDTH, 4))[:, :, :3]
        frames.append(Image.fromarray(pixels))

    gif_path = Path(__file__).parent / f"gifs/urdf/sideview/{file.stem}.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=DURATION,
        loop=0,  # Loop infinitely
    )

    pybullet.resetSimulation()
    pybullet.disconnect()

    return gif_path


def render_usd(file):
    pass


def main():
    render_funcs = {"mjcf": render_mjcf, "urdf": render_urdf, "usd": render_usd}

    root_dir = sim_objects.__path__[0]
    names = get_sim_names(root_dir)

    logging.info(f"Found the following sim objects: {names}")

    names = ["door"]
    seeds = 1

    # first export all the sim ready assets
    for obj in names:
        for exporter in ["mjcf", "usd", "urdf"]:
            for seed in range(seeds):
                try:
                    logging.info(f"{exporter} export for {obj} seed {seed}")
                    export_dir = Path(__file__).parent / "sim_exports/"
                    export_path, _ = sf.spawn_simready(
                        name=obj,
                        seed=seed,
                        exporter=exporter,
                        export_dir=export_dir,
                        visual_only=False,
                    )
                    render_funcs[exporter](export_path)
                except Exception:
                    logging.error(
                        f"Error exporting {obj} with {exporter} exporter (seed = {seed})."
                    )


if __name__ == "__main__":
    main()
