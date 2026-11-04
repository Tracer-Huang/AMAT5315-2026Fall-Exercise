import unittest
import infer
import numpy as np

class Tests(unittest.TestCase):
    def test_evidence_mass_and_dependence(self):
        m = infer.read_uai('asia.uai')
        z, marg, joint = infer.enumerate_asia(m, {0:0,7:0,6:0})
        self.assertAlmostEqual(z, 0.00098822675, places=14)
        self.assertAlmostEqual(joint[0,0], 0.022214, places=5)
        self.assertAlmostEqual(joint.sum(), 1)
        self.assertGreater(abs(joint[0,0]-marg[1][0]*marg[3][0]), .1)

    def test_bp_tree_agrees_exact(self):
        m = ([2,2], [((0,), np.array([.2,.8])), ((0,1), np.array([[.9,.1],[.3,.7]]))])
        beliefs, history = infer.bp(m, {}, max_sweeps=200, tolerance=1e-12, damping=.5)
        self.assertLess(abs(beliefs[1][0]-.42), 1e-10)
        self.assertLess(history[-1]['residual'], 1e-12)

class ContractionTests(unittest.TestCase):
    def test_contraction_matches_enumeration(self):
        m = infer.read_uai('asia.uai')
        for ev in ({}, {0:0,7:0}, {0:0,7:0,6:0}):
            z, marg, joint = infer.enumerate_asia(m, ev)
            z2, marg2, joint2 = infer.contract_asia(m, ev)
            self.assertAlmostEqual(z,z2,places=14)
            self.assertLess(max(np.max(np.abs(a-b)) for a,b in zip(marg,marg2)), 1e-10)
            self.assertLess(np.max(np.abs(joint-joint2)), 1e-10)

class PlanTests(unittest.TestCase):
    def test_plan_replay_preserves_mass(self):
        m=infer.read_uai('asia.uai'); ev={0:0,7:0}
        target=infer.enumerate_asia(m,ev)[0]
        for method in ('greedy','random'):
            plan=infer.make_plan(m,ev,method,42)
            value,cost=infer.execute_plan(m,ev,plan)
            self.assertAlmostEqual(value,target,places=14)
            self.assertGreater(cost['work'],0)
