"""
Unit tests and analytical verification for Equations (8)-(18) from:
Toba, Baruah, and Azumi, "Enhancing Deadline Miss Early Detection for DAG Tasks
with Variable Probabilistic Thresholds", ACM TECS 2026 (DOI: 10.1145/3808694).
"""
import os
import sys
import unittest
import numpy as np
import networkx as nx

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "benchmarks"))

from run_research_benchmarks import eq15_18, compute_plaxity


class TestWorkloadAndThresholdRecursions(unittest.TestCase):
    """Verifies Equations (13)-(18) on chain, branching, and merging DAGs."""

    def compute_rem(self, G, topo, w):
        """Analytical implementation of Equation (17): rem_v = w_v + min_{succ} rem_succ."""
        rem = {}
        for v in reversed(topo):
            succs = list(G.successors(v))
            if not succs:
                rem[v] = float(w[v])
            else:
                rem[v] = float(w[v]) + min(rem[u] for u in succs)
        return rem

    def test_chain_dag(self):
        """
        Linear Chain: 0 -> 1 -> 2 -> 3 -> 4
        Uniform workload w = 1.0 (Eq. 13).
        Analytical rem:
            rem[4] = 1.0
            rem[3] = 1.0 + 1.0 = 2.0
            rem[2] = 1.0 + 2.0 = 3.0
            rem[1] = 1.0 + 3.0 = 4.0
            rem[0] = 1.0 + 4.0 = 5.0
        Thresholds (th_min=0.8, th_max=1.0):
            th[0] = 0.8 (Eq. 15, entry)
            th[1] = th[0] + (1 - th[0]) / rem[1] = 0.8 + 0.2 / 4.0 = 0.85
            th[2] = th[1] + (1 - th[1]) / rem[2] = 0.85 + 0.15 / 3.0 = 0.90
            th[3] = th[2] + (1 - th[2]) / rem[3] = 0.90 + 0.10 / 2.0 = 0.95
            th[4] = 1.0 (Eq. 16, exit)
        """
        G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])
        topo = [0, 1, 2, 3, 4]
        w = {i: 1.0 for i in topo}
        rem = self.compute_rem(G, topo, w)

        self.assertEqual(rem, {4: 1.0, 3: 2.0, 2: 3.0, 1: 4.0, 0: 5.0})
        th = eq15_18(G, topo, w, rem, th_min=0.8, th_max=1.0)
        np.testing.assert_allclose([th[i] for i in topo], [0.80, 0.85, 0.90, 0.95, 1.00])

    def test_diamond_merging_dag_uniform(self):
        """
        Diamond DAG:
             1
           /   \\
          0     3
           \\   /
             2
        Edges: (0, 1), (0, 2), (1, 3), (2, 3)
        Uniform workload w = 1.0.
        Analytical rem:
            rem[3] = 1.0
            rem[1] = 1.0 + 1.0 = 2.0
            rem[2] = 1.0 + 1.0 = 2.0
            rem[0] = 1.0 + min(rem[1], rem[2]) = 1.0 + 2.0 = 3.0
        Thresholds:
            th[0] = 0.80
            th[1] = 0.80 + 0.20 / 2.0 = 0.90
            th[2] = 0.80 + 0.20 / 2.0 = 0.90
            th[3] = 1.00 (exit node, Eq. 16)
        """
        G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
        topo = [0, 1, 2, 3]
        w = {i: 1.0 for i in topo}
        rem = self.compute_rem(G, topo, w)

        self.assertEqual(rem, {3: 1.0, 1: 2.0, 2: 2.0, 0: 3.0})
        th = eq15_18(G, topo, w, rem, th_min=0.8, th_max=1.0)
        np.testing.assert_allclose([th[i] for i in topo], [0.80, 0.90, 0.90, 1.00])

    def test_asymmetric_branching_and_merging_timeweighted(self):
        """
        Asymmetric Diamond:
             1 (w=10)
           /          \\
          0 (w=10)     3 (w=20)
           \\          /
             2 (w=50)
        Workloads: w[0]=10, w[1]=10, w[2]=50, w[3]=20.
        Equation (17):
            rem[3] = 20.0
            rem[1] = 10.0 + 20.0 = 30.0
            rem[2] = 50.0 + 20.0 = 70.0
            rem[0] = 10.0 + min(rem[1], rem[2]) = 10.0 + min(30.0, 70.0) = 40.0
            (Paper page 16: "The minimum is taken to prioritize paths with smaller remaining workload")
        Thresholds (Eq. 15, 18, 16):
            th[0] = 0.80 (entry)
            th[1] = 0.80 + (1.0 - 0.80) / rem[1] = 0.80 + 0.20 / 30.0 = 0.806666...
            th[2] = 0.80 + (1.0 - 0.80) / rem[2] = 0.80 + 0.20 / 70.0 = 0.802857...
            th[3] = 1.00 (exit)
        Note: If node 3 were an intermediate node, max_u over predecessors would be:
            via 1: th[1] + (1 - th[1]) / rem[3]
            via 2: th[2] + (1 - th[2]) / rem[3]
            Since th[1] > th[2], the max correctly selects the path via node 1.
        """
        G = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 3)])
        topo = [0, 1, 2, 3]
        w = {0: 10.0, 1: 10.0, 2: 50.0, 3: 20.0}
        rem = self.compute_rem(G, topo, w)

        self.assertEqual(rem[3], 20.0)
        self.assertEqual(rem[1], 30.0)
        self.assertEqual(rem[2], 70.0)
        self.assertEqual(rem[0], 40.0)

        th = eq15_18(G, topo, w, rem, th_min=0.8, th_max=1.0)
        self.assertAlmostEqual(th[0], 0.80)
        self.assertAlmostEqual(th[1], 0.80 + 0.20 / 30.0)
        self.assertAlmostEqual(th[2], 0.80 + 0.20 / 70.0)
        self.assertAlmostEqual(th[3], 1.00)

    def test_multi_predecessor_intermediate_selection(self):
        """
        Verify Eq. (18) max over predecessors on an intermediate node:
        0 -> 2, 1 -> 2, 2 -> 3
        0: th_min = 0.80, rem[0] = 50
        1: th_min = 0.80, rem[1] = 30
        2: intermediate node with rem[2] = 20
        """
        G = nx.DiGraph([(0, 2), (1, 2), (2, 3)])
        topo = [0, 1, 2, 3]
        w = {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0}
        rem = self.compute_rem(G, topo, w)
        # rem[3] = 10, rem[2] = 20, rem[0] = 30, rem[1] = 30
        th = eq15_18(G, topo, w, rem, th_min=0.8, th_max=1.0)
        # 0 and 1 are entry nodes -> 0.80
        # 2 has preds {0, 1}:
        # candidate from 0: 0.80 + (1.0 - 0.80) / 20 = 0.81
        # candidate from 1: 0.80 + (1.0 - 0.80) / 20 = 0.81
        self.assertAlmostEqual(th[0], 0.80)
        self.assertAlmostEqual(th[1], 0.80)
        self.assertAlmostEqual(th[2], 0.81)
        self.assertAlmostEqual(th[3], 1.00)


class TestPlaxityPaperExample(unittest.TestCase):
    """Verifies Equations (10)-(11) against the paper's Fig. 6 and Fig. 7."""

    def test_fig6_convolutional_subtraction(self):
        """
        Paper Fig. 6 (Page 13):
        X1: P(10) = 0.8, P(20) = 0.2
        comm_{1,2} = 5
        L_{2,2}: P(85) = 0.1, P(95) = 0.9
        Formula Eq. (10):
            P(L_{1,3} = 60) = P(X1=20)*P(L_{2,2}=85) = 0.2 * 0.1 = 0.02
            P(L_{1,3} = 70) = P(X1=10)*P(L_{2,2}=85) + P(X1=20)*P(L_{2,2}=95) = 0.08 + 0.18 = 0.26
            P(L_{1,3} = 80) = P(X1=10)*P(L_{2,2}=95) = 0.72
        """
        D = 100
        # Represent L_{2,2} shifted by communication time (5)
        # LFT at successor = L_{2,2} - comm = 80 (0.1) and 90 (0.9)
        lft = np.zeros(D + 1)
        lft[80] = 0.1
        lft[90] = 0.9

        # C1 distribution
        c1 = np.zeros(21)
        c1[10] = 0.8
        c1[20] = 0.2

        # Correlate: LST = LFT - C
        corr = np.correlate(lft, c1, mode="full")
        z = len(c1) - 1
        lst = np.zeros(D + 1)
        lst[0] = corr[: z + 1].sum()
        end = min(len(corr), z + D + 1)
        lst[1: end - z] = corr[z + 1: end]

        self.assertAlmostEqual(lst[60], 0.02)
        self.assertAlmostEqual(lst[70], 0.26)
        self.assertAlmostEqual(lst[80], 0.72)
        self.assertAlmostEqual(lst.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
