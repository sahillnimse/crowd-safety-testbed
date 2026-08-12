"""
Head counting for crowd density.

A density-map counter trained from the point labels that
scripts/annotate_heads.py produces.  It exists to supply the rho in Helbing
crowd pressure (see models/crowd_flow/density.py): at the density where a
crush becomes possible, bodies are almost entirely occluded and a box
detector stops counting, while heads stay visible and a summed density map
stays correct.

    scripts/extract_patches.py      cut training patches out of your footage
    scripts/annotate_heads.py       click the heads
    scripts/train_head_count.py     train
    HeadCounter                     use it
"""

from models.head_count.infer import HeadCounter
from models.head_count.model import HeadCountNet, CountLoss, OUTPUT_STRIDE

__all__ = ["HeadCounter", "HeadCountNet", "CountLoss", "OUTPUT_STRIDE"]
