import argparse
import json
import os
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ASSETS / "results.json"))
    ap.add_argument("--out", default=str(ASSETS / "humanoid.gif"))
    ap.add_argument("--fps", type=int, default=67)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium as gym
    import mujoco
    from PIL import Image

    results = json.loads(Path(args.results).read_text())
    traj = results["final"]["trajectory"]

    env = gym.make(results["env"], render_mode="rgb_array",
                   width=args.size, height=args.size)
    env.reset(seed=0)
    model = env.unwrapped.model
    data = env.unwrapped.data
    nq = model.nq

    images = []
    for state in traj[::args.every][:args.max_frames]:
        data.qpos[:] = state[:nq]
        data.qvel[:] = state[nq:]
        mujoco.mj_forward(model, data)
        frame = Image.fromarray(env.render())
        images.append(frame.quantize(args.colors, method=Image.MEDIANCUT))
    env.close()

    images[0].save(args.out, save_all=True, append_images=images[1:],
                   duration=int(1000 * args.every / args.fps), loop=0,
                   optimize=True)
    print(f"wrote {args.out}: {len(images)} frames, "
          f"return {results['final']['max']:.0f}")


if __name__ == "__main__":
    main()
