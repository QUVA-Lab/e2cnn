import unittest
from unittest import TestCase

from e2cnn.nn import *
from e2cnn.gspaces import *

import torch

import numpy as np

import random


class TestNonLinearitiesRotations(TestCase):
    
    def test_cyclic_norm_relu(self):
        N = 8
        g = Rot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values()) * 4)
        
        nnl = NormNonLinearity(r, function='n_relu')

        nnl.check_equivariance()
    
    def test_cyclic_norm_sigmoid(self):
        N = 8
        g = Rot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values()) * 4)
        
        nnl = NormNonLinearity(r, function='n_sigmoid')
        
        nnl.check_equivariance()
    
    def test_cyclic_pointwise_relu(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs)
        
        # nnl = PointwiseNonLinearity(r, function='p_relu')
        nnl = ReLU(r)
        
        nnl.check_equivariance()
    
    def test_cyclic_pointwise_sigmoid(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs)
        
        nnl = PointwiseNonLinearity(r, function='p_sigmoid')
        
        nnl.check_equivariance()
    
    def test_cyclic_gated_one_input_shuffled_gated(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        ngates = len(reprs)
        reprs += [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        r = FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()
    
    def test_cyclic_gated_one_input_sorted_gated(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs).sorted()
        
        ngates = len(r)
        
        reprs = [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        r = r + FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()
    
    def test_cyclic_gated_one_input_all_shuffled(self):
        N = 8
        g = Rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        
        ngates = len(reprs)
        
        reprs += [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        t = list(zip(reprs, gates))
        
        random.shuffle(t)
        
        reprs, gates = zip(*t)
        
        r = FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_cyclic_gated_two_inputs_shuffled_gated(self):
        N = 8
        g = Rot2dOnR2(N)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates
    
        gates = FieldType(g, gates)
        gated = FieldType(g, gated)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_cyclic_gated_two_inputs_sorted_gated(self):
        N = 8
        g = Rot2dOnR2(N)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]

        gated = FieldType(g, gated).sorted()
    
        ngates = len(gated)
    
        gates = [g.trivial_repr] * ngates
        gates = FieldType(g, gates)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_cyclic_concat_relu(self):
        N = 8
        g = Rot2dOnR2(N)
    
        reprs = [r for r in g.representations.values() if 'concatenated' in r.supported_nonlinearities]
    
        for rep in reprs:
            print(rep.name)
            r = FieldType(g, [rep])
            nnl = ConcatenatedNonLinearity(r, function='c_relu')
            nnl.check_equivariance()

    def test_cyclic_vectorfield(self):
        N = 8
        g = Rot2dOnR2(N)
    
        reprs = [r for r in g.representations.values() if 'vectorfield' in r.supported_nonlinearities] * 8
    
        r = FieldType(g, reprs)
        nnl = VectorFieldNonLinearity(r)
        nnl.check_equivariance(atol=2e-6)

    def test_cyclic_induced_norm_relu(self):
    
        N = 15
        g = Rot2dOnR2(N)
    
        sg_id = 5
        sg, _, _ = g.fibergroup.subgroup(sg_id)
    
        r = FieldType(g, [g.induced_repr(sg_id, sg.irrep(k)) for k in range(1, int(sg.order() // 2))] * 4).sorted()
        nnl = InducedNormNonLinearity(r, function='n_relu')
        nnl.check_equivariance()

    def test_so2_norm_relu(self):
        
        g = Rot2dOnR2(-1, 10)
    
        r = FieldType(g, list(g.representations.values()) * 4)
    
        nnl = NormNonLinearity(r, function='n_relu')
    
        nnl.check_equivariance()

    def test_so2_norm_sigmoid(self):
        g = Rot2dOnR2(-1, 10)
    
        r = FieldType(g, list(g.representations.values()) * 4)
    
        nnl = NormNonLinearity(r, function='n_sigmoid')
    
        nnl.check_equivariance()

    def test_so2_pointwise_relu(self):
        g = Rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
    
        r = FieldType(g, reprs)
    
        nnl = PointwiseNonLinearity(r, function='p_relu')
    
        nnl.check_equivariance()

    def test_so2_pointwise_sigmoid(self):
        g = Rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
    
        r = FieldType(g, reprs)
    
        nnl = PointwiseNonLinearity(r, function='p_sigmoid')
    
        nnl.check_equivariance()

    def test_so2_gated_one_input_shuffled_gated(self):
        g = Rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        ngates = len(reprs)
        reprs += [g.trivial_repr] * ngates
    
        gates = ['gated'] * ngates + ['gate'] * ngates
    
        r = FieldType(g, reprs)
    
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_so2_gated_one_input_sorted_gated(self):
        g = Rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
    
        r = FieldType(g, reprs).sorted()
    
        ngates = len(r)
    
        reprs = [g.trivial_repr] * ngates
    
        gates = ['gated'] * ngates + ['gate'] * ngates
    
        r = r + FieldType(g, reprs)
    
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_so2_gated_one_input_all_shuffled(self):
        g = Rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
    
        ngates = len(reprs)
    
        reprs += [g.trivial_repr] * ngates
    
        gates = ['gated'] * ngates + ['gate'] * ngates
    
        t = list(zip(reprs, gates))
    
        random.shuffle(t)
    
        reprs, gates = zip(*t)
    
        r = FieldType(g, reprs)
    
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_so2_gated_two_inputs_shuffled_gated(self):
        g = Rot2dOnR2(-1, 10)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates

        gates = FieldType(g, gates)
        gated = FieldType(g, gated)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_so2_gated_two_inputs_sorted_gated(self):
        g = Rot2dOnR2(-1, 10)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]

        gated = FieldType(g, gated).sorted()
    
        ngates = len(gated)

        gates = [g.trivial_repr] * ngates
        gates = FieldType(g, gates)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_cyclic_gated1_error(self):
        N = 8
        g = Rot2dOnR2(N)
        
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

    def test_cyclic_gated2_error(self):
        N = 8
        g = Rot2dOnR2(N)
    
        for r in g.representations.values():
            if 'gated' not in r.supported_nonlinearities:
                gated = FieldType(g, [r])
                gates = FieldType(g, [g.trivial_repr])
                self.assertRaises(AssertionError, GatedNonLinearity2, (gates, gated))
    
        for r in g.representations.values():
            if 'gate' not in r.supported_nonlinearities:
                gated = FieldType(g, [g.trivial_repr])
                gates = FieldType(g, [r])
                self.assertRaises(AssertionError, GatedNonLinearity2, (gates, gated))

    def test_cyclic_norm_error(self):
        N = 8
        g = Rot2dOnR2(N)
        
        for r in g.representations.values():
        
            if 'norm' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r])
                self.assertRaises(AssertionError, NormNonLinearity, r1)

    def test_cyclic_pointwise_error(self):
        N = 8
        g = Rot2dOnR2(N)
        
        for r in g.representations.values():
        
            if 'pointwise' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r])
                self.assertRaises(AssertionError, PointwiseNonLinearity, r1)

    def test_cyclic_concat_error(self):
        N = 8
        g = Rot2dOnR2(N)
        
        for r in g.representations.values():

            if 'concatenated' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r])
                self.assertRaises(AssertionError, ConcatenatedNonLinearity, r1)


if __name__ == '__main__':
    unittest.main()
