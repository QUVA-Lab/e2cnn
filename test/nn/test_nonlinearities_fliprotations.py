import unittest
from unittest import TestCase

from e2cnn.nn import *
from e2cnn.gspaces import *

import random


class TestNonLinearitiesFlipRotations(TestCase):
    
    def test_dihedral_norm_relu(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values()) * 4)
        
        nnl = NormNonLinearity(r, function='n_relu')
        
        nnl.check_equivariance()
    
    def test_dihedral_norm_sigmoid(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values()) * 4)
        
        nnl = NormNonLinearity(r, function='n_sigmoid')
        
        nnl.check_equivariance()
    
    def test_dihedral_pointwise_relu(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs)
        
        nnl = PointwiseNonLinearity(r, function='p_relu')
        
        nnl.check_equivariance()
    
    def test_dihedral_pointwise_sigmoid(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs)
        
        nnl = PointwiseNonLinearity(r, function='p_sigmoid')
        
        nnl.check_equivariance()
    
    def test_dihedral_gated_one_input_shuffled_gated(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 3
        ngates = len(reprs)
        reprs += [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        r = FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()
    
    def test_dihedral_gated_one_input_sorted_gated(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 3
        
        r = FieldType(g, reprs).sorted()
        
        ngates = len(r)
        
        reprs = [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        r = r + FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()
    
    def test_dihedral_gated_one_input_all_shuffled(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 2
        
        ngates = len(reprs)
        
        reprs += [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        t = list(zip(reprs, gates))
        
        random.shuffle(t)
        
        reprs, gates = zip(*t)
        
        r = FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_dihedral_gated_two_inputs_shuffled_gated(self):
        N = 8
        g = FlipRot2dOnR2(N)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 3
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates

        gates = FieldType(g, gates)
        gated = FieldType(g, gated)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_dihedral_gated_two_inputs_sorted_gated(self):
        N = 8
        g = FlipRot2dOnR2(N)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 2
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates
    
        gates = FieldType(g, gates)
        gated = FieldType(g, gated).sorted()
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_dihedral_concat_relu(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'concatenated' in r.supported_nonlinearities]
        
        for rep in reprs:
            r = FieldType(g, [rep])
            nnl = ConcatenatedNonLinearity(r, function='c_relu')
            nnl.check_equivariance()

    def test_dihedral_induced_norm_relu(self):
    
        N = 9
        g = FlipRot2dOnR2(N)
    
        sg_id = (None, N)
        so2, _, _ = g.fibergroup.subgroup(sg_id)
        
        r = FieldType(g, [g.induced_repr(sg_id, so2.irrep(k)) for k in range(1, int(N // 2))] * 4).sorted()
        nnl = InducedNormNonLinearity(r, function='n_relu')
        nnl.check_equivariance()

    def test_o2_induced_norm_relu(self):
    
        g = FlipRot2dOnR2(-1, 10)
        
        sg_id = (None, -1)
        so2, _, _ = g.fibergroup.subgroup(sg_id)
        r = FieldType(g, [g.induced_repr(sg_id, so2.irrep(k)) for k in range(1, 7)] * 4).sorted()
        nnl = InducedNormNonLinearity(r, function='n_relu')
        nnl.check_equivariance()

    def test_o2_induced_gated(self):
    
        g = FlipRot2dOnR2(-1, 10)
    
        sg_id = (None, -1)
        so2, _, _ = g.fibergroup.subgroup(sg_id)

        reprs = [g.induced_repr(sg_id, so2.irrep(k)) for k in range(1, 3)] * 5
        ngates = len(reprs)
        reprs += [g.induced_repr(sg_id, so2.trivial_representation)] * ngates

        gates = ['gated'] * ngates + ['gate'] * ngates

        r = FieldType(g, reprs)

        nnl = InducedGatedNonLinearity1(r, gates=gates)
        
        nnl.check_equivariance()

    def test_o2_norm_relu(self):
        
        g = FlipRot2dOnR2(-1, 10)
    
        r = FieldType(g, list(g.representations.values()) * 4)
    
        nnl = NormNonLinearity(r, function='n_relu')
    
        nnl.check_equivariance()

    def test_o2_norm_sigmoid(self):
        g = FlipRot2dOnR2(-1, 10)
    
        r = FieldType(g, list(g.representations.values()) * 4)
    
        nnl = NormNonLinearity(r, function='n_sigmoid')
    
        nnl.check_equivariance()

    def test_o2_pointwise_relu(self):
        g = FlipRot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
    
        r = FieldType(g, reprs)
    
        nnl = PointwiseNonLinearity(r, function='p_relu')
    
        nnl.check_equivariance()

    def test_o2_pointwise_sigmoid(self):
        g = FlipRot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
    
        r = FieldType(g, reprs)
    
        nnl = PointwiseNonLinearity(r, function='p_sigmoid')
    
        nnl.check_equivariance()

    def test_o2_gated_one_input_shuffled_gated(self):
        g = FlipRot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 3
        ngates = len(reprs)
        reprs += [g.trivial_repr] * ngates
    
        gates = ['gated'] * ngates + ['gate'] * ngates
    
        r = FieldType(g, reprs)
    
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_o2_gated_one_input_sorted_gated(self):
        g = FlipRot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 2
    
        r = FieldType(g, reprs).sorted()
    
        ngates = len(r)
    
        reprs = [g.trivial_repr] * ngates
    
        gates = ['gated'] * ngates + ['gate'] * ngates
    
        r = r + FieldType(g, reprs)
    
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_o2_gated_one_input_all_shuffled(self):
        g = FlipRot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 3
    
        ngates = len(reprs)
    
        reprs += [g.trivial_repr] * ngates
    
        gates = ['gated'] * ngates + ['gate'] * ngates
    
        t = list(zip(reprs, gates))
    
        random.shuffle(t)
    
        reprs, gates = zip(*t)
    
        r = FieldType(g, reprs)
    
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_o2_gated_two_inputs_shuffled_gated(self):
        g = FlipRot2dOnR2(-1, 10)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 3
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates

        gates = FieldType(g, gates)
        gated = FieldType(g, gated)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_o2_gated_two_inputs_sorted_gated(self):
        g = FlipRot2dOnR2(-1, 10)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 2
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates
        
        gated = FieldType(g, gated).sorted()
        gates = FieldType(g, gates)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_dihedral_gated1_error(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        for r in g.representations.values():
            if 'gated' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r, g.trivial_repr])
                gates = ['gated', 'gate']
                self.assertRaises(AssertionError, GatedNonLinearity1, r1, gates=gates)
    
        for r in g.representations.values():
            if 'gate' not in r.supported_nonlinearities:
                r1 = FieldType(g, [g.trivial_repr, r])
                gates = ['gated', 'gate']
                self.assertRaises(AssertionError, GatedNonLinearity1, r1, gates=gates)

    def test_dihedral_gated2_error(self):
        N = 8
        g = FlipRot2dOnR2(N)
    
        for r in g.representations.values():
            if 'gated' not in r.supported_nonlinearities:
                gates = FieldType(g, [g.trivial_repr])
                gated = FieldType(g, [r])
                
                self.assertRaises(AssertionError, GatedNonLinearity2, (gates, gated))
    
        for r in g.representations.values():
            if 'gate' not in r.supported_nonlinearities:
                gates = FieldType(g, [r])
                gated = FieldType(g, [g.trivial_repr])
    
                self.assertRaises(AssertionError, GatedNonLinearity2, (gates, gated))

    def test_dihedral_norm_error(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        for r in g.representations.values():
        
            if 'norm' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r])
                self.assertRaises(AssertionError, NormNonLinearity, r1)

    def test_dihedral_pointwise_error(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        for r in g.representations.values():
        
            if 'pointwise' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r])
                self.assertRaises(AssertionError, PointwiseNonLinearity, r1)

    def test_dihedral_concat_error(self):
        N = 8
        g = FlipRot2dOnR2(N)
        
        for r in g.representations.values():

            if 'concatenated' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r])
                self.assertRaises(AssertionError, ConcatenatedNonLinearity, r1)


if __name__ == '__main__':
    unittest.main()
