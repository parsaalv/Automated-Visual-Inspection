"""Entry point for the Digital Twin demo.

Sets up console encoding and environment variables, builds the demo's
object list (either real photographs or generated mock textures),
spawns them onto the conveyor over time, and runs the simulation loop
until every object has fallen off the far end.
"""

import os
import sys

# --- Console encoding setup (avoids crashes when logging non-ASCII text on some terminals) ---
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')

os.environ["TRUST_REMOTE_CODE"] = "1"

from config import Config
from logging_config import logger
from objects import ObjectState, create_dummy_texture
from simulation import SimulationManager


def run_digital_twin_demo():
    """Run the full digital twin demo end to end.

    Builds the list of items to spawn (real photographs in "real" vision
    mode, or generated mock textures otherwise), starts the simulation,
    spawns objects onto the conveyor at fixed step intervals, and steps
    the simulation until either the configured step budget is reached or
    every spawned object has fallen off the conveyor. Cleans up any
    generated mock texture files and disconnects PyBullet on exit.
    """
    config = Config()
    # Important: previously this line always forced VISION_MODE to "scenario",
    # even if it had been set to "real" above in Config or here! That caused the
    # system to keep following the pre-defined scenario even when VISION_MODE = "real"
    # was configured. This value is no longer overridden; whatever is set in
    # Config (the class at the top of the file) is what applies.
    # config.VISION_MODE = "real"   # <-- uncomment to test with the real model
    # config.VISION_MODE = "scenario"   # <-- uncomment to test without a model (manual scenario)
    dummy_files_to_cleanup = []  # Only the mock images we generate ourselves get cleaned up
    if config.VISION_MODE == "real":
        # =====================================================================
        # REAL mode: you must provide the path to your own real photographs here.
        # These photos must belong to one of the classes that YOLO/Anomaly were
        # trained on: screw, metal_nut, transistor, cable, bottle, toothbrush
        # otherwise YOLO will detect nothing and, per the fail-safe logic, every
        # object will be treated as GOOD (so you'll see neither a red box nor
        # the arms activating).
        #
        # Note (fixed bug): these paths used to be absolute and specific to one
        # particular MacBook (/Users/macbookair/...) and would immediately crash
        # with a FileNotFoundError on any other machine. Replace these paths with
        # the real path on your own system, or place the images next to this
        # script and use a relative path.
        # =====================================================================
        image_paths = []
        for root, dirs, files in os.walk("data"):
            for file in sorted(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.join(root, file)
                    norm_path = full_path.replace('\\', '/')
                    if "ground_truth" not in norm_path and "_mask" not in file:
                        image_paths.append(full_path)

        textures = {}
        for i, path in enumerate(image_paths):
            textures[f"object_{i+1:03d}"] = path

        missing = [path for path in textures.values() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "The following image file(s) were not found; fix the 'textures' path "
                f"in run_digital_twin_demo:\n" + "\n".join(missing)
            )

    else:
        # Mock mode (scenario/random): the same colored placeholder images as before
        config.VISION_SCENARIO = {
            "object_001": 0,  # Good
            "object_002": 1,  # Repair
            "object_003": 2,  # Scrap
        }
        textures = {
            "object_001": create_dummy_texture("part_good.png",   "GOOD",   (180, 220, 180)),
            "object_002": create_dummy_texture("part_repair.png", "REPAIR", (220, 220, 150)),
            "object_003": create_dummy_texture("part_scrap.png",  "SCRAP",  (220, 160, 160)),
        }
        dummy_files_to_cleanup = list(textures.values())
    sim = SimulationManager(config)
    try:
        sim.start()
        items_to_spawn = []
        if config.VISION_MODE == "real":
            for i, (obj_label, tex_path) in enumerate(textures.items()):
                parts = tex_path.replace('\\', '/').split('/')
                obj_type = f"Tool_{i}"
                for p_name in parts:
                    if p_name in config.SELECTED_CLASSES:
                        obj_type = p_name
                        break
                items_to_spawn.append((obj_type, tex_path))
        else:
            items_to_spawn = [
                ("Tool_A", textures["object_001"]),
                ("Tool_B", textures["object_002"]),
                ("Tool_C", textures["object_003"]),
            ]

        # Spawning items dynamically on conveyor so no object falls off at step 0
        spawn_interval = config.SPAWN_INTERVAL
        total_steps = max(config.SIMULATION_STEPS, len(items_to_spawn) * spawn_interval + 2500)

        print(f"\n--- Digital Twin Running (VISION_MODE = '{config.VISION_MODE}', Objects = {len(items_to_spawn)}) ---")

        spawn_index = 0
        for step_i in range(total_steps):
            if spawn_index < len(items_to_spawn) and (step_i == 10 or (step_i > 10 and (step_i - 10) % spawn_interval == 0)):
                obj_type, tex_path = items_to_spawn[spawn_index]
                start_pos = [-3.5, 0, config.CONVEYOR_HEIGHT + 0.1]
                sim.object_manager.spawn_object(obj_type, tex_path, start_pos=start_pos)
                spawn_index += 1

            sim.step()

            # Check if all objects have fallen off (meaning they passed the arm)
            if spawn_index == len(items_to_spawn):
                all_done = True
                for obj_id, obj in sim.object_manager.objects.items():
                    if obj.state != ObjectState.FELL_OFF:
                        all_done = False
                        break
                if all_done and len(sim.object_manager.objects) > 0:
                    print(f"\n[SimManager] All objects have fallen off. Terminating simulation at step {step_i}.")
                    break
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        # Previously, any exception other than KeyboardInterrupt wasn't fully
        # caught cleanly, and the error message sometimes got lost among
        # PyBullet's log output. Now it is printed clearly and 'finally' still runs.
        logger.error(f"[SimManager] Unexpected error: {e}")
        raise
    finally:
        sim.disconnect()
        for tex_path in dummy_files_to_cleanup:
            if os.path.exists(tex_path):
                os.remove(tex_path)


# Run it - re-running this cell will not crash!
if __name__ == "__main__":
    run_digital_twin_demo()
