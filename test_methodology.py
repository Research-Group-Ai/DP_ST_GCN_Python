import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from dpstgcn.graph import adjacency
from dpstgcn.preprocess import prepare_clip,representation
from dpstgcn.model import Recognizer
from dpstgcn.data import load_manifest

# CPU checks should not use a large BLAS thread pool.
torch.set_num_threads(1)
CROSS=[0,1,2,4,8,5,9,6,10]


class MethodTests(unittest.TestCase):
    def test_graph(self):
        a=adjacency('dual',CROSS)
        np.testing.assert_allclose(a,a.T)
        self.assertEqual(np.count_nonzero(a[:25,25:]),9)
        self.assertTrue((np.diag(a)>0).all())
        self.assertFalse(adjacency('dual_no_cross')[:25,25:].any())

    def clip(self):
        x=np.ones((8,2,25,3),dtype=np.float32)
        x[:,0,2,1]=2
        x[:,1,:,0]+=2
        return x

    def test_missing_bone_motion(self):
        raw=self.clip(); raw[:,1]=0
        x,mask=prepare_clip(raw,frames=20,training=True,rng=np.random.default_rng(4))
        for stream in ['joint','bone','joint_motion','bone_motion']:
            features=representation(x,mask,stream)
            self.assertEqual(features.shape,(3,20,50))
            self.assertTrue(np.isfinite(features).all())
            self.assertFalse(features[:,:,25:].any())
        bone=representation(x,mask,'bone')
        self.assertFalse(bone[:,:,0].any())

    def test_no_future_leakage(self):
        a=self.clip(); b=a.copy(); b[4:]*=20
        xa,ma=prepare_clip(a,frames=20,fraction=.5)
        xb,mb=prepare_clip(b,frames=20,fraction=.5)
        np.testing.assert_array_equal(xa,xb)
        np.testing.assert_array_equal(ma,mb)

    def test_preserve_person_distance(self):
        x,_=prepare_clip(self.clip(),frames=20)
        np.testing.assert_allclose(x[:,1,0,0]-x[:,0,0,0],2.)

    def test_models_and_gradients(self):
        c=dict(classes=51,cross_joints=CROSS,dropout=.5)
        for mode in ['dual','single','dual_no_cross']:
            m=Recognizer(c,mode)
            x=torch.randn(2,3,20,50)
            logits,alpha=m(x)
            self.assertEqual(tuple(logits.shape),(2,51))
            torch.testing.assert_close(alpha.sum(-1),torch.ones_like(alpha.sum(-1)))
            loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1])); loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_fusion_aligns_ids(self):
        from scripts.fuse import aligned
        j=dict(ids=np.array(['a','b']),labels=np.array([0,1]),
               probabilities=np.array([[.8,.2],[.1,.9]]),split=np.array('val'),
               fraction=np.array(1.),stream=np.array('joint'),mode=np.array('dual'))
        b=dict(ids=np.array(['b','a']),labels=np.array([1,0]),
               probabilities=np.array([[.2,.8],[.7,.3]]),split=np.array('val'),
               fraction=np.array(1.),stream=np.array('bone'),mode=np.array('dual'))
        _,pb,_=aligned(j,b)
        np.testing.assert_allclose(pb,[[.7,.3],[.2,.8]])
        b['labels']=np.array([0,1])
        with self.assertRaisesRegex(ValueError,'Label mismatch'): aligned(j,b)

    def test_leakage_rejected(self):
        rows=[dict(id='a',path='a.npy',label=0,split='train',subjects=['shared'],recording='one'),
              dict(id='b',path='b.npy',label=0,split='val',subjects=['shared'],recording='two')]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json'; p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'subjects leakage'): load_manifest(p)


if __name__=='__main__': unittest.main()
