"""Portable resource policy; no historical deadline or remote process control."""
import os
import shutil
import time
from .gpu_guard_core import Guard as Base, HEADROOM, ResourcePause

class Guard(Base):
    def __init__(self,output):
        super().__init__(output)
        self.deadline=float(os.environ['GAPSET_RUN_DEADLINE'])
    def check(self,force=False):
        if time.time()+30>=self.deadline:raise ResourcePause('Declared run time limit')
        if shutil.disk_usage(self.output).free<80*1024**3:raise ResourcePause('80 GiB free disk floor')
        return super().check(force)
