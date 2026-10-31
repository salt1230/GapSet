"""Full speed only without competing compute processes; retain actual headroom."""

import json
import os
import signal
import subprocess
import time

HEADROOM = 3 * 1024**3
POLICY = 'actual_free_minus3GiB_fullspeed_idle_only'

class ResourcePause(RuntimeError):
    pass

class Guard:
    def __init__(self, output):
        self.output = output
        self.stopped = False
        self.last_check = 0.
        self.allocation_cap_bytes = None
        signal.signal(signal.SIGUSR1, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *_):
        self.stopped = True

    def event(self, **record):
        with (self.output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps({'time_unix':time.time(),'pid':os.getpid(),**record})+'\n')

    def snapshot(self):
        raw = subprocess.check_output(['nvidia-smi',
            '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=10)
        total, used, free, util, temperature = map(int, raw.splitlines()[0].split(','))
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],text=True,timeout=10)
        pids = [int(line.strip()) for line in processes.splitlines() if line.strip()]
        return {'total_bytes':total*1024**2,'used_bytes':used*1024**2,
                'free_bytes':free*1024**2,'utilization':util,'temperature_c':temperature,
                'foreign_compute_pids':[pid for pid in pids if pid!=os.getpid()]}

    def pressure(self, card, cuda_free):
        if card['foreign_compute_pids']:
            raise ResourcePause('Competing GPU compute detected; yield only our worker')
        if card['temperature_c'] >= 82:
            raise ResourcePause('GPU temperature >=82C; preserve last committed epoch')
        if min(card['free_bytes'],cuda_free) < HEADROOM:
            raise ResourcePause('Actual free memory below3GiB reserve; preserve committed epoch')

    def admit(self):
        import torch

        card = self.snapshot()
        cuda_free, cuda_total = torch.cuda.mem_get_info()
        self.pressure(card,cuda_free)
        cap = min(card['free_bytes'],cuda_free)-HEADROOM
        if cap < 4*1024**3:
            raise ResourcePause('Insufficient currently free GPU memory after3GiB reserve')
        self.allocation_cap_bytes = cap
        self.event(event='admission',card=card,cuda_free_bytes=cuda_free,cuda_total_bytes=cuda_total,
            allocation_cap_bytes=cap,headroom_bytes=HEADROOM,memory_policy=POLICY,duty_fraction=1.)
        torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory)
        self.check(True)

    def check(self, force=False):
        import torch

        if self.stopped:
            raise ResourcePause('Owned finite controller/user stop requested')
        if not force and time.monotonic()-self.last_check < 2:
            return
        card = self.snapshot()
        cuda_free, cuda_total = torch.cuda.mem_get_info()
        self.event(event='memory_check',card=card,cuda_free_bytes=cuda_free,cuda_total_bytes=cuda_total,
            reserved_bytes=torch.cuda.memory_reserved(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            headroom_bytes=HEADROOM,memory_policy=POLICY)
        self.last_check = time.monotonic()
        self.pressure(card,cuda_free)

    def throttle(self, started):
        import torch

        torch.cuda.synchronize()
        self.check()
