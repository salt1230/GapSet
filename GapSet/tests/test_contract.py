"""Synthetic invariants only: no dataset access, model forward or GPU."""
import unittest
import numpy as np
from gapset.data.contract import Observation,challenge
from gapset.data.dataset import encode,conditions
from gapset.data.preprocess import aggregate_packets,window_array,identity
from gapset.analysis.interventions import transform
from gapset.analysis.cue_features import mask_features
from gapset.analysis.paired_groups import paired_intervals
from gapset.metrics import metrics
from pathlib import Path

class ContractTests(unittest.TestCase):
    def observation(self):
        x=np.ones((256,3,30),np.float32);m=np.ones_like(x,dtype=bool)
        return Observation(x,m,np.arange(256)*.01,np.arange(30),np.arange(3),'synthetic:window0','derived_grid')
    def test_all_fixed_conditions_preserve_grid_and_are_deterministic(self):
        self.assertEqual(len(conditions()),46)
        self.assertEqual(len({c['name'] for c in conditions()}),46)
        obs=self.observation()
        for c in conditions():
            if c['kind']=='joint':continue
            a,log=challenge(obs,c['kind'],c.get('amount',0),seed=c['seed'],position=c.get('position',.5),links=tuple(c.get('links',[])))
            b,_=challenge(obs,c['kind'],c.get('amount',0),seed=c['seed'],position=c.get('position',.5),links=tuple(c.get('links',[])))
            np.testing.assert_array_equal(a.valid,b.valid);np.testing.assert_array_equal(a.time_seconds,obs.time_seconds)
            self.assertFalse(log['labels_accessed']);self.assertLessEqual(a.valid.sum(),obs.valid.sum())
    def test_hidden_amplitude_cannot_change_common8(self):
        obs=self.observation();obs.valid[::3]=False
        hidden=obs.amplitude.copy();hidden[~obs.valid]=np.nan
        a=encode(obs.amplitude,obs.valid,0.,1.);b=encode(hidden,obs.valid,0.,1.)
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(8,30,256))
    def test_terminal_burst_keeps_original_positions(self):
        a,_=challenge(self.observation(),'step_burst',.5,position=1.)
        self.assertTrue(a.valid[:128].all());self.assertFalse(a.valid[128:].any())
    def test_packet_time_gaps_and_tail_are_not_stretched(self):
        x=np.ones((3,3,30),np.float32);a,m,_=aggregate_packets(x,np.array([0,10000,30000]))
        self.assertEqual(len(a),4);self.assertFalse(m[2].any())
        windows=window_array(a,m);self.assertEqual(windows[0][1].shape,(256,3,30))
        self.assertFalse(windows[0][2][4:].any())
    def test_group_split_keeps_receivers_together(self):
        rows=[identity(Path('raw/20181204/user1-1-1-1-1-r'+str(i)+'.dat'),Path('raw')) for i in (1,2,6)]
        self.assertEqual(len({r['role'] for r in rows}),1);self.assertEqual(len({r['group_id'] for r in rows}),1)
    def test_exact_metric_units_and_absent_class_policy(self):
        self.assertEqual(metrics(np.arange(6),np.arange(6),6)['macro_f1_percent'],100.)
        self.assertAlmostEqual(metrics(np.array([0]),np.array([0]),6)['macro_f1_percent'],100/6)
    def test_content_intervention_retains_mask_coordinates(self):
        obs=self.observation();image=encode(obs.amplitude,obs.valid,0.,1.)
        result,_=transform(image,'fit_mean_content','synthetic')
        np.testing.assert_array_equal(result[3:],image[3:]);self.assertFalse(result[:3].any())
        length,f=mask_features(obs.valid);self.assertEqual(length,256);self.assertEqual(len(f),861)
    def test_paired_group_bootstrap_has_zero_interval_for_identical_models(self):
        counts=np.zeros((2,3,2,4,6,6),dtype=np.int64)
        for g in range(4):counts[:,:,:,g,g,g]=g+1
        r=paired_intervals(counts,np.array([1,1,2,2]),['ours_gapset_relative_mass_control','control'],['clean','challenge'],replicates=11)
        for row in r['comparisons']:
            self.assertEqual(row['mean_seed_difference_pp'],0.)
            self.assertEqual(row['conditional_group_percentile95'],[0.,0.])

if __name__=='__main__':unittest.main()
